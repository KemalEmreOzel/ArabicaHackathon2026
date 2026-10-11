"""
Arabica x YourEye Hackathon 2026 - Asama 2:
Sehre Giren Arac Sayimi, Takibi, Evrensel/Uluslararasi Plaka Tespiti ve OCR Sistemi

MIMARI VE OZELLIKLER:
1. Vehicle Detection & Tracking:
   - YOLO11m ile COCO arac siniflari (2: Otomobil, 3: Motosiklet, 5: Otobus/Minibus, 7: Kamyon).
   - ByteTrack ile kesintisiz takip ID'si ve yon takibi (tekrar sayimi onler).
2. Sehir Giris Sanal Cizgisi & Yonlu Cok Seritli / Yan Yol Sayimi:
   - Genisletilmis sanal giris cizgisi (x: 180 -> 1850, y = 750) uzerinden hem ana yoldan hem de
     sagdaki baglanti/yan yoldan sehre giren tum araclar tekil olarak sayilir.
3. Evrensel ve Uluslararasi Plaka Tespiti & Format Siniflandirma:
   - Turkiye (TR), Avrupa Birligi (EU: Almanya, Fransa, Italya, Polonya vb.), Birlesik Krallik (UK),
     Amerika (US) ve genel uluslararasi alfanumerik plaka formatlari desteklenir.
   - Coklu on isleme (CLAHE kontrast esitleme, Otsu ikili esikleme, renkli olcekleme).
   - Ham gorselden (HUD kirliligi olmadan) temiz tampon ve plaka kesiti cikarimi.
4. Dinamik Kayit Guncelleme:
   - Arac cizgiyi gectikten sonra kameraya yaklasip plakasi daha net okundugunda gecis kaydi ve HUD
     canli olarak en guncel yuksek guvenilirlikli plaka ile guncellenir.
5. Disa Aktarma & Raporlama:
   - outputs/arac_gecis_kayitlari.csv
   - outputs/arac_gecis_kayitlari.json
   - outputs/arac_gecis_kayitlari.txt
   - outputs/crops/plates/ ve outputs/crops/vehicles/
"""

import os
import sys
import time
import json
import csv
import re
import argparse
import logging
from datetime import datetime
from typing import Tuple, Dict, Set, Optional, List
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import easyocr

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Asama2_Arac")

# COCO Sinif Id -> Turkce Isim ve Renk
VEHICLE_CLASSES = {
    2: {"name": "Otomobil", "color": (0, 215, 255)},      # Sari/Turuncu
    3: {"name": "Motosiklet", "color": (255, 100, 50)},   # Mavi
    5: {"name": "Otobus/Minibus", "color": (50, 205, 50)}, # Yesil
    7: {"name": "Kamyon", "color": (230, 80, 180)}         # Mor
}


# ==============================================================================
# 1. EVRENSEL / ULUSLARARASI PLAKA DUZENLEME VE FORMAT SINIFLANDIRMA
# ==============================================================================
def clean_plate_text(raw_text: str) -> str:
    """Metindeki ozel sembol ve bosluklari temizler, standart buyuk harfe cevirir."""
    return re.sub(r"[^A-Z0-9]", "", raw_text.upper())


def classify_and_format_universal_plate(raw_text: str) -> Tuple[Optional[str], str]:
    """
    Plakayi temizler; Turkiye (TR), AB (EU), UK, US veya genel uluslararasi (EVRENSEL)
    formatlara gore analiz edip duzenler.
    Dondurur: (bicimlendirilmis_plaka, format_turu)
    """
    clean = clean_plate_text(raw_text)
    if len(clean) < 4 or len(clean) > 10:
        return None, "GECERSIZ"

    # 1. Turkiye (TR) Plaka: 01-81 (Il Kodu) + 1-3 Harf + 2-4 Rakam
    tr_m = re.match(r"^(0[1-9]|[1-7][0-9]|8[01])([A-Z]{1,3})([0-9]{2,4})$", clean)
    if tr_m:
        return f"{tr_m.group(1)} {tr_m.group(2)} {tr_m.group(3)}", "TR"

    # 2. Avrupa Birligi (EU) - Fransa / Italya (SIV Formati: AB 123 CD)
    fr_m = re.match(r"^([A-Z]{2})([0-9]{3})([A-Z]{2})$", clean)
    if fr_m:
        return f"{fr_m.group(1)} {fr_m.group(2)} {fr_m.group(3)}", "EU (FR/IT)"

    # 3. Birlesik Krallik (UK Formati: BD51 SMR)
    uk_m = re.match(r"^([A-Z]{2}[0-9]{2})([A-Z]{3})$", clean)
    if uk_m:
        return f"{uk_m.group(1)} {uk_m.group(2)}", "UK"

    # 4. Avrupa Birligi (EU) - Polonya / Orta Avrupa (Orn: WA 4869A, KR 12345)
    pl_m = re.match(r"^([A-Z]{2,3})([0-9]{1,4}[A-Z]{1,2}|[0-9]{4,5})$", clean)
    if pl_m:
        return f"{pl_m.group(1)} {pl_m.group(2)}", "EU (PL/CE)"

    # 5. Almanya (DE) / Benzeri (Sehir Kodu + Harfler + Rakamlar: B MW 1234)
    de_m = re.match(r"^([A-Z]{1,3})([A-Z]{1,2})([0-9]{1,4})$", clean)
    if de_m:
        return f"{de_m.group(1)} {de_m.group(2)} {de_m.group(3)}", "EU (DE)"

    # 6. Amerika / Kanada / Uluslararasi (US Formati: ABC 1234)
    us_m = re.match(r"^([A-Z]{3})([0-9]{3,4})$", clean)
    if us_m:
        return f"{us_m.group(1)} {us_m.group(2)}", "US/INTL"

    # 7. Evrensel Harf ve Rakam Bloklari (Orn: 21 OTO, 60 OTE, 12 AB 34)
    blocks = re.findall(r"[A-Z]+|[0-9]+", clean)
    if len(blocks) > 1 and len(clean) >= 4:
        return " ".join(blocks), "EVRENSEL"

    # 8. Alfanumerik genel plaka (Orn: STECBUD, SELBUD)
    if len(clean) >= 5:
        return clean, "EVRENSEL"

    return None, "GECERSIZ"


# ==============================================================================
# 2. PLAKA TESPİTİ VE COKLU ON ISLEMELI EASYOCR BILESENI
# ==============================================================================
class PlateOCRProcessor:
    """
    Arac gorselinden tampon/plaka ROI crop'u alir, coklu on islemeden (CLAHE, Otsu, Orijinal)
    gecirir ve evrensel plaka algoritmalarini calistirir.
    """

    def __init__(self, device: str = "cuda"):
        use_gpu = torch.cuda.is_available() and (device == "cuda" or device == "0" or device == 0)
        logger.info(f"EasyOCR Evrensel Okuyucu Baslatiliyor (GPU: {use_gpu})...")
        self.reader = easyocr.Reader(["en"], gpu=use_gpu, download_enabled=False)
        logger.info("EasyOCR basariyla hazirlandi!")

    def extract_plate_crop(self, frame: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Aracin tampon ve alt plaka montaj alanini net bir sekilde crop alir.
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1

        # Plaka bolgesi dikey: [y1 + 0.50*h : y2 - 0.02*h]
        # Plaka bolgesi yatay: [x1 + 0.08*w : x2 - 0.08*w]
        py1 = int(y1 + 0.50 * h)
        py2 = int(y2 - 0.02 * h)
        px1 = int(x1 + 0.08 * w)
        px2 = int(x2 - 0.08 * w)

        px1 = max(0, min(w_frame - 1, px1))
        py1 = max(0, min(h_frame - 1, py1))
        px2 = max(0, min(w_frame, px2))
        py2 = max(0, min(h_frame, py2))

        if py2 <= py1 or px2 <= px1:
            return frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)], (x1, y1, x2, y2)

        crop = frame[py1:py2, px1:px2]
        return crop, (px1, py1, px2, py2)

    def preprocess_variants(self, img_crop: np.ndarray) -> List[np.ndarray]:
        """Goruntuyu farkli kontrast ve ikili esikleme varyasyonlarina cevirir."""
        if img_crop.size == 0:
            return []

        # 2x olcekleme (kucuk plaka piksellerini buyutur)
        scaled = cv2.resize(img_crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # 1. CLAHE Adaptif Histogram Esitleme
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2. Otsu Adaptif Ikili Esikleme (Binary Binarization)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Varyantlar sirasi: CLAHE Gri, Otsu Ikili, Orijinal Olcekli
        return [enhanced, thresh, scaled]

    def read_plate(self, plate_crop: np.ndarray) -> Tuple[str, float, str]:
        """
        Crop uzerinde evrensel OCR calistirir.
        Dondurur: (bicimlendirilmis_plaka, guven_skoru, format_tipi)
        """
        if plate_crop.shape[0] < 12 or plate_crop.shape[1] < 20:
            return "", 0.0, "BILINMIYOR"

        try:
            variants = self.preprocess_variants(plate_crop)
            best_plate = ""
            best_conf = 0.0
            best_fmt = "BILINMIYOR"

            noise_words = {"OTOMOBIL", "KAMYON", "OTOBUS", "MOTOR", "VEHICLE", "CAR", "TRUCK"}

            for var_img in variants:
                results = self.reader.readtext(var_img, detail=1, paragraph=False)
                if not results:
                    continue

                # 1. Aday: Tespit edilen parcalari birlestir (orn: '38' + 'AB 123' veya 'WA' + '4869A')
                combined_tokens = []
                total_conf = 0.0
                for bbox, text, conf in results:
                    c_up = text.strip().upper()
                    if any(nw in c_up for nw in noise_words):
                        continue
                    combined_tokens.append(text)
                    total_conf += float(conf)

                if combined_tokens:
                    comb_text = " ".join(combined_tokens)
                    comb_conf = total_conf / len(combined_tokens)
                    fmt_p, fmt_t = classify_and_format_universal_plate(comb_text)
                    if fmt_p and comb_conf > best_conf:
                        best_plate = fmt_p
                        best_conf = comb_conf
                        best_fmt = fmt_t

                # 2. Aday: Tek tek metin kutularini kontrol et
                for bbox, text, conf in results:
                    c_up = text.strip().upper()
                    if any(nw in c_up for nw in noise_words):
                        continue
                    fmt_p, fmt_t = classify_and_format_universal_plate(text)
                    if fmt_p and float(conf) > best_conf:
                        best_plate = fmt_p
                        best_conf = float(conf)
                        best_fmt = fmt_t

                if best_conf > 0.85:
                    break

            return best_plate, best_conf, best_fmt
        except Exception as e:
            logger.debug(f"OCR hatasi: {e}")
            return "", 0.0, "BILINMIYOR"


# ==============================================================================
# 3. GENISLETILMIS SEHIR GIRIS SANAL CIZGISI VE ARAC SAYACI
# ==============================================================================
class CityEntranceCounter:
    """
    Sanal sehir giris cizgisini (ana yol + sag yan yol / tum seritler) gecen araclari
    sayar ve ID tekillestirmesi yapar.
    """

    def __init__(
        self,
        line_p1: Tuple[int, int] = (180, 750),
        line_p2: Tuple[int, int] = (1850, 750),
        cooldown_frames: int = 40
    ):
        self.p1 = np.array(line_p1, dtype=float)
        self.p2 = np.array(line_p2, dtype=float)
        self.cooldown_frames = cooldown_frames

        self.prev_positions: Dict[int, Tuple[float, float]] = {}
        self.last_cross_frame: Dict[int, int] = {}
        self.counted_ids: Set[int] = set()

        self.total_count: int = 0
        self.class_counts: Dict[str, int] = {
            "Otomobil": 0,
            "Motosiklet": 0,
            "Otobus/Minibus": 0,
            "Kamyon": 0
        }

    def update(
        self,
        track_id: int,
        center_pos: Tuple[float, float],
        frame_idx: int,
        veh_class_name: str
    ) -> bool:
        """Cizgiyi yukaridan asagiya (sehre giris) gecen araci sayar."""
        cx, cy = center_pos
        is_counted = False

        if track_id in self.prev_positions:
            prev_x, prev_y = self.prev_positions[track_id]

            line_y = self.p1[1]
            min_x = min(self.p1[0], self.p2[0]) - 50
            max_x = max(self.p1[0], self.p2[0]) + 50

            if min_x <= cx <= max_x:
                # Yukaridan asagiya dogru gecis (sehre giris: prev_y < line_y <= cy)
                if (prev_y < line_y <= cy) or (prev_y <= line_y < cy):
                    last_f = self.last_cross_frame.get(track_id, -999)
                    if (frame_idx - last_f > self.cooldown_frames) and (track_id not in self.counted_ids):
                        self.counted_ids.add(track_id)
                        self.last_cross_frame[track_id] = frame_idx
                        self.total_count += 1
                        self.class_counts[veh_class_name] = self.class_counts.get(veh_class_name, 0) + 1
                        is_counted = True
                        road_type = "Sag Yan Yol" if cx > 1250 else "Ana Yol"
                        logger.info(
                            f"[SEHRE GIRIS] Arac #{track_id} ({veh_class_name}) | "
                            f"Konum: {road_type} (X={cx:.0f}) | Toplam Giris: {self.total_count}"
                        )

        self.prev_positions[track_id] = (cx, cy)
        return is_counted

    def cleanup(self, active_ids: Set[int]):
        stale = [tid for tid in self.prev_positions if tid not in active_ids]
        for tid in stale:
            del self.prev_positions[tid]


# ==============================================================================
# 4. GORSELLESTIRME VE HUD PANELI
# ==============================================================================
class VehicleRenderer:

    COLOR_BG = (18, 18, 22)
    COLOR_BORDER = (70, 70, 85)
    COLOR_ACCENT = (0, 215, 255)
    COLOR_LINE = (0, 255, 120)

    def __init__(self, counter: CityEntranceCounter):
        self.counter = counter
        # Son gecen araclar kaydi: List[dict]
        self.recent_passes: List[dict] = []

    def add_recent_pass(self, info: dict):
        self.recent_passes.insert(0, info)
        if len(self.recent_passes) > 5:
            self.recent_passes.pop()

    def draw_scene(
        self,
        frame: np.ndarray,
        device_name: str,
        fps: float
    ):
        # 1. Sanal Sehir Giris Cizgisi (Ana Yol + Sag Baglanti Yolu)
        p1 = tuple(self.counter.p1.astype(int))
        p2 = tuple(self.counter.p2.astype(int))

        overlay = frame.copy()
        cv2.line(overlay, p1, p2, self.COLOR_LINE, 4, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.line(frame, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)

        line_y = p1[1]

        # Ana Yol Ok ve Etiketi (x ~ 680)
        cv2.arrowedLine(frame, (680, line_y - 30), (680, line_y + 30), (0, 255, 255), 3, tipLength=0.35)
        cv2.putText(frame, "ANA YOL GIRISI v", (700, line_y + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

        # Sag Yan Yol Ok ve Etiketi (x ~ 1550)
        cv2.arrowedLine(frame, (1550, line_y - 30), (1550, line_y + 30), (0, 255, 255), 3, tipLength=0.35)
        cv2.putText(frame, "YAN YOL GIRISI v", (1390, line_y - 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

        # 2. Sol Ust Dashboard Paneli
        self._draw_stats_dashboard(frame, device_name, fps)

        # 3. Sag Ust Canli Plaka Akisi
        self._draw_recent_feed(frame)

    def _draw_stats_dashboard(self, frame: np.ndarray, device_name: str, fps: float):
        pw, ph = 540, 220
        x1, y1 = 20, 20
        x2, y2 = x1 + pw, y1 + ph

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLOR_BG, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLOR_BORDER, 2)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.putText(frame, "ARABICA x YOUREYE", (x1 + 16, y1 + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, self.COLOR_ACCENT, 2, cv2.LINE_AA)
        cv2.putText(frame, "Asama 2: Sehre Giris Sayimi & Evrensel Plaka OCR", (x1 + 16, y1 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Model: YOLO11m + EasyOCR | {device_name} | {fps:.1f} FPS", (x1 + 16, y1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 170), 1, cv2.LINE_AA)

        cv2.line(frame, (x1 + 16, y1 + 70), (x2 - 16, y1 + 70), (60, 60, 70), 1)

        # Toplam Giris Kutusu
        box_y = y1 + 82
        cv2.rectangle(frame, (x1 + 16, box_y), (x1 + 175, box_y + 60), (35, 45, 25), -1)
        cv2.rectangle(frame, (x1 + 16, box_y), (x1 + 175, box_y + 60), (0, 255, 120), 2)
        cv2.putText(frame, "TOPLAM GIRIS", (x1 + 24, box_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter.total_count), (x1 + 55, box_y + 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 120), 2, cv2.LINE_AA)

        # Sinif Dagilimi
        cx_base = x1 + 195
        cls_y = box_y + 14
        for cls_name, cnt in self.counter.class_counts.items():
            cv2.putText(frame, f"- {cls_name}: {cnt}", (cx_base, cls_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
            cls_y += 18

        # Alt Bilgi Notu
        note = "* Evrensel Plaka (TR, EU, UK, US, INTL) + Cok Seritli / Yan Yol Takibi Aktif"
        cv2.putText(frame, note, (x1 + 16, y1 + 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 200, 220), 1, cv2.LINE_AA)
        csv_note = "Kayit: outputs/arac_gecis_kayitlari.csv, json, txt"
        cv2.putText(frame, csv_note, (x1 + 16, y1 + 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140, 180, 190), 1, cv2.LINE_AA)

    def _draw_recent_feed(self, frame: np.ndarray):
        if not self.recent_passes:
            return

        fw, fh = 420, 25 + len(self.recent_passes) * 32
        x1 = frame.shape[1] - fw - 20
        y1 = 20
        x2, y2 = x1 + fw, y1 + fh

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLOR_BG, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLOR_BORDER, 2)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.putText(frame, "SON GECEN ARACLAR & EVRENSEL PLAKALAR", (x1 + 14, y1 + 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

        item_y = y1 + 38
        for item in self.recent_passes:
            tid = item["track_id"]
            vcls = item["class"]
            plate = item.get("plate", "OKUNUYOR")
            fmt = item.get("fmt", "")
            conf = item.get("plate_conf", 0.0)

            p_color = (0, 255, 120) if plate not in ["OKUNUYOR", "PLAKA_BELIRSIZ"] else (180, 180, 180)
            fmt_str = f"[{fmt}] " if fmt and fmt != "YOK" else ""
            txt = f"#{tid} {vcls} | {fmt_str}{plate} (%{conf*100:.0f})"
            cv2.putText(frame, txt, (x1 + 14, item_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, p_color, 1, cv2.LINE_AA)
            item_y += 30

    def draw_vehicle(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        track_id: Optional[int],
        veh_class: str,
        veh_conf: float,
        plate_text: Optional[str] = None,
        plate_fmt: Optional[str] = None,
        plate_conf: float = 0.0,
        is_counted: bool = False
    ):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 120) if is_counted else (0, 215, 255)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Arac Etiketi
        tag = f"#{track_id} {veh_class} (%{veh_conf*100:.0f})" if track_id is not None else veh_class
        (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        lbl_y1 = max(0, y1 - th - bl - 4)
        lbl_y2 = y1
        cv2.rectangle(frame, (x1, lbl_y1), (x1 + tw + 8, lbl_y2), color, -1)
        cv2.putText(frame, tag, (x1 + 4, y1 - bl - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        # Plaka Etiketi (varsa kutunun altinda)
        if plate_text and plate_text != "PLAKA_BELIRSIZ":
            fmt_prefix = f"[{plate_fmt}] " if plate_fmt and plate_fmt != "YOK" else ""
            p_lbl = f"PLAKA: {fmt_prefix}{plate_text} (%{plate_conf*100:.0f})"
            (pw, ph), pbl = cv2.getTextSize(p_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            py1 = y2
            py2 = min(frame.shape[0], y2 + ph + pbl + 4)
            cv2.rectangle(frame, (x1, py1), (x1 + pw + 8, py2), (0, 0, 0), -1)
            cv2.rectangle(frame, (x1, py1), (x1 + pw + 8, py2), (0, 255, 120), 1)
            cv2.putText(frame, p_lbl, (x1 + 4, y2 + ph + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1, cv2.LINE_AA)


# ==============================================================================
# 5. ANA ARAC SAYIMI VE EVRENSEL PLAKA BORU HATTI
# ==============================================================================
class AdvancedVehiclePipeline:

    def __init__(
        self,
        video_path: str = "videos/arac_sayimi.mp4",
        output_path: str = "outputs/asama2_arac_plaka_sonuc.mp4",
        output_csv: str = "outputs/arac_gecis_kayitlari.csv",
        output_json: str = "outputs/arac_gecis_kayitlari.json",
        crops_dir: str = "outputs/crops",
        model_path: str = "yolo11m.pt",
        conf_thresh: float = 0.30,
        iou_thresh: float = 0.50,
        imgsz: int = 1280,
        device: Optional[str] = None,
        max_frames: Optional[int] = None
    ):
        self.video_path = video_path
        self.output_path = output_path
        self.output_csv = output_csv
        self.output_json = output_json
        self.crops_dir = crops_dir
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.max_frames = max_frames

        if device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"
            self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        else:
            self.device = device
            self.device_name = str(device)

        os.makedirs(self.crops_dir + "/plates", exist_ok=True)
        os.makedirs(self.crops_dir + "/vehicles", exist_ok=True)
        os.makedirs(os.path.dirname(self.output_csv) or ".", exist_ok=True)

        self.plate_ocr = PlateOCRProcessor(device="cuda" if torch.cuda.is_available() else "cpu")
        # Genisletilmis cizgi: x: 180 -> 1850 (Ana yol + sag yan yol / tum seritler)
        self.counter = CityEntranceCounter(line_p1=(180, 750), line_p2=(1850, 750))
        self.renderer = VehicleRenderer(self.counter)

        # Takip ID -> En iyi plaka verisi: {"plate": str, "conf": float, "fmt": str, "crop_path": str}
        self.track_plates: Dict[int, dict] = {}
        # Kaydedilmis gecis kayitlari listesi ve ID tablosu
        self.passage_records: List[dict] = []
        self.records_by_id: Dict[int, dict] = {}

        logger.info(f"YOLO11m Yukleniyor: {self.model_path} | Cihaz: {self.device} ({self.device_name})")
        self.model = YOLO(self.model_path)

    def run(self):
        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Video bulunamadi: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video acilamadi: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Asama 2 Video: {width}x{height} @ {fps:.1f} FPS | Toplam Kare: {total_frames}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cikti video yazicisi acilamadi: {self.output_path}")

        frame_count = 0
        start_time = time.time()
        last_calc_time = start_time
        cur_fps = 0.0

        try:
            while cap.isOpened():
                if self.max_frames and frame_count >= self.max_frames:
                    logger.info(f"Belirtilen maksimum kare sayisina ulasildi: {self.max_frames}")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # Ham goruntu - HUD ve cizim kirliligi olmadan
                clean_frame = frame.copy()

                # 1. YOLO11m ile Arac Tespiti ve ByteTrack Takibi (2: car, 3: moto, 5: bus, 7: truck)
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[2, 3, 5, 7],
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False
                )

                active_ids = set()

                if results and results[0].boxes:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    track_ids = (
                        results[0].boxes.id.cpu().numpy().astype(int)
                        if results[0].boxes.id is not None
                        else None
                    )

                    if track_ids is not None:
                        active_ids = set(track_ids)

                    for idx in range(len(boxes)):
                        box = boxes[idx]
                        conf = float(confs[idx])
                        cls_id = cls_ids[idx]
                        veh_info = VEHICLE_CLASSES.get(cls_id, {"name": "Arac", "color": (0, 255, 255)})
                        veh_name = veh_info["name"]
                        tid = track_ids[idx] if track_ids is not None else None

                        cx = (box[0] + box[2]) / 2.0
                        cy = float(box[3])  # Alt temas noktasi

                        # Plaka Tespiti & Evrensel OCR (Arac kameraya yakinken)
                        if tid is not None:
                            plate_data = self.track_plates.get(tid, None)
                            in_roi_plate_zone = 500 <= cy <= 950
                            should_ocr = (plate_data is None) or (plate_data["conf"] < 0.85 and in_roi_plate_zone and frame_count % 6 == 0)

                            if should_ocr:
                                # HAM TEMIZ kareden crop al (HUD kirliligi olmadan)
                                p_crop, p_coords = self.plate_ocr.extract_plate_crop(clean_frame, box)
                                p_text, p_conf, p_fmt = self.plate_ocr.read_plate(p_crop)

                                if p_text:
                                    if (plate_data is None) or (p_conf > plate_data["conf"]):
                                        p_crop_fn = f"{self.crops_dir}/plates/plate_ID{tid}_f{frame_count}.jpg"
                                        v_crop_fn = f"{self.crops_dir}/vehicles/veh_ID{tid}_f{frame_count}.jpg"
                                        cv2.imwrite(p_crop_fn, p_crop)
                                        x1, y1, x2, y2 = map(int, box)
                                        cv2.imwrite(v_crop_fn, clean_frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)])

                                        self.track_plates[tid] = {
                                            "plate": p_text,
                                            "conf": p_conf,
                                            "fmt": p_fmt,
                                            "crop_path": p_crop_fn
                                        }

                                        # Eger arac daha once cizgiyi gectiyse kaydini dinamik guncelle!
                                        if tid in self.records_by_id:
                                            rec = self.records_by_id[tid]
                                            rec["Plaka"] = str(p_text)
                                            rec["Plaka_Guven"] = float(round(p_conf, 3))
                                            rec["Plaka_Formati"] = str(p_fmt)
                                            rec["Plaka_Gorsel_Yolu"] = str(p_crop_fn)
                                            for r_item in self.renderer.recent_passes:
                                                if r_item["track_id"] == tid:
                                                    r_item["plate"] = str(p_text)
                                                    r_item["fmt"] = str(p_fmt)
                                                    r_item["plate_conf"] = float(p_conf)

                            # Sehir Giris Cizgisi Sayimi (Ana yol + Yan yol)
                            just_crossed = self.counter.update(tid, (cx, cy), frame_count, veh_name)
                            if just_crossed:
                                p_info = self.track_plates.get(tid, {"plate": "PLAKA_BELIRSIZ", "conf": 0.0, "fmt": "YOK", "crop_path": ""})
                                road_type = "Sag Yan Yol" if cx > 1250 else "Ana Yol"
                                record = {
                                    "Zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Kare": int(frame_count),
                                    "Arac_ID": int(tid),
                                    "Arac_Turu": str(veh_name),
                                    "Arac_Guven": float(round(conf, 3)),
                                    "Gecis_Yolu": str(road_type),
                                    "Plaka": str(p_info["plate"]),
                                    "Plaka_Guven": float(round(p_info["conf"], 3)),
                                    "Plaka_Formati": str(p_info["fmt"]),
                                    "Gecis_Yonu": "Sehre Giris (IN)",
                                    "Plaka_Gorsel_Yolu": str(p_info["crop_path"])
                                }
                                self.passage_records.append(record)
                                self.records_by_id[tid] = record
                                self.renderer.add_recent_pass({
                                    "track_id": int(tid),
                                    "class": str(veh_name),
                                    "plate": str(p_info["plate"]),
                                    "fmt": str(p_info["fmt"]),
                                    "plate_conf": float(p_info["conf"])
                                })

                        # Araci ve Plakasini Ciz
                        p_display = self.track_plates.get(tid, None)
                        p_txt = p_display["plate"] if p_display else None
                        p_fmt_txt = p_display["fmt"] if p_display else None
                        p_cnf = p_display["conf"] if p_display else 0.0
                        is_cnt = tid in self.counter.counted_ids if tid is not None else False

                        self.renderer.draw_vehicle(
                            frame=frame,
                            box=box,
                            track_id=tid,
                            veh_class=veh_name,
                            veh_conf=conf,
                            plate_text=p_txt,
                            plate_fmt=p_fmt_txt,
                            plate_conf=p_cnf,
                            is_counted=is_cnt
                        )

                self.counter.cleanup(active_ids)

                # Anlik FPS
                if frame_count % 15 == 0:
                    now = time.time()
                    cur_fps = 15.0 / max(1e-5, now - last_calc_time)
                    last_calc_time = now

                # Cizgi ve Dashboard Cizimi
                self.renderer.draw_scene(
                    frame=frame,
                    device_name=self.device_name,
                    fps=cur_fps
                )

                writer.write(frame)

                if frame_count % 150 == 0 or frame_count == total_frames:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / max(1e-5, elapsed)
                    prog = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    logger.info(
                        f"Asama 2 | Kare {frame_count:4d}/{total_frames} ({prog:5.1f}%) | "
                        f"Hiz: {avg_fps:4.1f} FPS | Sehre Giris Yapan Arac: {self.counter.total_count}"
                    )

        finally:
            cap.release()
            writer.release()

        # Sonuclari CSV, JSON ve TXT formatinda disari aktar
        self._export_records()

        total_elapsed = time.time() - start_time
        avg_fps = frame_count / max(1e-5, total_elapsed)

        logger.info("=" * 60)
        logger.info("ASAMA 2 (SEHIR GIRISI ARAC SAYIMI & EVRENSEL PLAKA) TAMAMLANDI!")
        logger.info(f"Toplam Islenen Kare       : {frame_count}")
        logger.info(f"Toplam Sure               : {total_elapsed:.2f} saniye")
        logger.info(f"Ortalama Islem Hizi       : {avg_fps:.2f} FPS")
        logger.info(f"Toplam Sehre Giren Arac   : {self.counter.total_count}")
        for cname, ccnt in self.counter.class_counts.items():
            logger.info(f"  - {cname:15s}: {ccnt}")
        logger.info(f"Cikti Video Dosyasi       : {os.path.abspath(self.output_path)}")
        logger.info(f"Cikti CSV Dosyasi         : {os.path.abspath(self.output_csv)}")
        logger.info(f"Cikti JSON Dosyasi        : {os.path.abspath(self.output_json)}")
        logger.info("=" * 60)

    def _export_records(self):
        """Sonuclari CSV, JSON ve TXT olarak kaydeder."""
        # 1. CSV Kaydi
        if self.passage_records:
            keys = self.passage_records[0].keys()
            with open(self.output_csv, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.passage_records)
            logger.info(f"CSV gecis kayitlari kaydedildi: {self.output_csv}")

        # 2. JSON Kaydi
        clean_counts = {str(k): int(v) for k, v in self.counter.class_counts.items()}
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "toplam_gecis": int(self.counter.total_count),
                "sinif_dagilimi": clean_counts,
                "gecisler": self.passage_records
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON gecis kayitlari kaydedildi: {self.output_json}")

        # 3. TXT Ozeti
        txt_path = self.output_csv.replace(".csv", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("ARABICA x YOUREYE HACKATHON 2026 - ASAMA 2 GECIS RAPORU\n")
            f.write(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Toplam Sehre Giren Arac: {self.counter.total_count}\n")
            f.write("Sinif Dagilimi:\n")
            for cname, ccnt in self.counter.class_counts.items():
                f.write(f"  - {cname}: {ccnt}\n")
            f.write("=" * 70 + "\n")
            f.write(f"{'ID':<6} {'Sinif':<14} {'Gecis Yolu':<14} {'Plaka':<16} {'Format':<10} {'Guven':<8}\n")
            f.write("-" * 70 + "\n")
            for r in self.passage_records:
                f.write(f"{r['Arac_ID']:<6} {r['Arac_Turu']:<14} {r.get('Gecis_Yolu', 'Ana Yol'):<14} {r['Plaka']:<16} {r['Plaka_Formati']:<10} {r['Plaka_Guven']:<8}\n")
        logger.info(f"TXT gecis ozeti kaydedildi: {txt_path}")


def main():
    args = parse_args()
    pipeline = AdvancedVehiclePipeline(
        video_path=args.video,
        output_path=args.output,
        output_csv=args.csv,
        output_json=args.json,
        crops_dir=args.crops,
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        max_frames=args.max_frames
    )
    pipeline.run()


def parse_args():
    parser = argparse.ArgumentParser(description="Arabica x YourEye Hackathon 2026 - Asama 2 (Evrensel Plaka & Yan Yol)")
    parser.add_argument("--video", type=str, default="videos/arac_sayimi.mp4", help="Giris video dosyasi")
    parser.add_argument("--output", type=str, default="outputs/asama2_arac_plaka_sonuc.mp4", help="Cikti video dosyasi")
    parser.add_argument("--csv", type=str, default="outputs/arac_gecis_kayitlari.csv", help="Cikti CSV dosyasi")
    parser.add_argument("--json", type=str, default="outputs/arac_gecis_kayitlari.json", help="Cikti JSON dosyasi")
    parser.add_argument("--crops", type=str, default="outputs/crops", help="Kesitlerin kaydedilecegi ana dizin")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="YOLO model dosyasi")
    parser.add_argument("--conf", type=float, default=0.30, help="Guven skoru esigi")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU NMS esigi")
    parser.add_argument("--imgsz", type=int, default=1280, help="Giris cozunurlugu")
    parser.add_argument("--max-frames", type=int, default=None, help="Islenecek maksimum kare sayisi (None: tum video)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
