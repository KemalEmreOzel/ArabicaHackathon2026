"""
Arabica x YourEye Hackathon 2026 - Asama 1:
Gelismis Kisi Tespiti, Takibi, Kapi Esigi Sayimi ve OpenAI CLIP Zero-Shot Cinsiyet Siniflandirma Sistemi

MIMARI VE YENILIKCI YONTEM:
1. YOLO11m + ByteTrack:
   - Yuksek dogruluklu kisi tespiti ve kesintisiz takip ID'si.
   - Nested box suppression ile otururken olusan ust/alt govde cift tespitleri elenir.
   - SpatialROIFilter: Sag ust cam arkasindaki dis alan kisileri elenir, yalnizca kafe icindekiler sayilir.
   - DoorLineCounter: Gercek kapi esiginde ([740, 275] -> [1060, 275]) cift yonlu (Giris/Cikis) sayim.

2. OpenAI CLIP Zero-Shot Cinsiyet Siniflandirma (SOTA):
   - Standart eski siniflandiricilar yerine OpenAI CLIP ViT-B/16 modeli kullanilir.
   - Model istemleri: "a photo of a man" ve "a photo of a woman".
   - Biyometrik kimlik tespiti degil, gorsel semantik benzerlik olasiligi uretilir.

3. Coklu Cerceve Agirlikli Oylama Sistemi (GenderVoter):
   - Kisi hareket ederken veya otururken coklu cercevelerden kaliteli crop kesitleri alinir.
   - Goruntu cozunurlugu, boyutu ve yakinligina gore dinamik oy agirligi (weight) atanir.
   - Zamansal agirlikli oylama ile anlik golge/isik yanilmalari elenir, titresimsiz karar sabitlenir.

4. Gorsellestirme Standartlari (Kullanici & Sartname Istegi):
   - Kadinlar: PEMBE KARE (Pink Bounding Box: (203, 108, 255))
   - Erkekler: MAVI KARE (Blue Bounding Box: (255, 140, 0))
   - Kutu uzerinde: ID, Cinsiyet, Guven Skoru (%)
   - Dashboard: Toplam Iceride, Erkek (Mavi), Kadin (Pembe), Kapi Giris/Cikis, FPS
   - Zorunlu Teknik Not Uyarisi HUD uzerinde gosterilir.
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Tuple, Dict, Set, Optional, List
from collections import deque
import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# Loglama yapilandirmasi
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Asama1_CLIP")

DEFAULT_CONFIG_PATH = "config/zone_config.json"


# ==============================================================================
# 1. KUTU BASTIRMA (NESTED BOX SUPPRESSION)
# ==============================================================================
def suppress_nested_boxes(
    boxes: np.ndarray,
    confs: np.ndarray,
    ids: Optional[np.ndarray] = None,
    containment_thresh: float = 0.72
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Oturan kisilerde ayni kisiye ait tam govde ve ust govde cift tespitlerini eler."""
    if len(boxes) <= 1:
        return boxes, confs, ids

    keep = [True] * len(boxes)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(boxes)):
            if not keep[j]:
                continue

            xx1 = max(boxes[i, 0], boxes[j, 0])
            yy1 = max(boxes[i, 1], boxes[j, 1])
            xx2 = min(boxes[i, 2], boxes[j, 2])
            yy2 = min(boxes[i, 3], boxes[j, 3])

            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h

            if inter > 0:
                min_area = min(areas[i], areas[j])
                containment = inter / (min_area + 1e-6)

                if containment >= containment_thresh:
                    if areas[i] < areas[j]:
                        keep[i] = False
                        break
                    else:
                        keep[j] = False

    keep_indices = np.array(keep, dtype=bool)
    filtered_boxes = boxes[keep_indices]
    filtered_confs = confs[keep_indices]
    filtered_ids = ids[keep_indices] if ids is not None else None

    return filtered_boxes, filtered_confs, filtered_ids


# ==============================================================================
# 2. BÖLGE VE KAPI EŞİĞİ ÇİZGİSİ YAPILANDIRMASI
# ==============================================================================
class ZoneConfig:
    """Kafe ici ROI cokgeni ve kapi esigi koordinatlarini yonetir."""

    DEFAULT_ROI = [
        [50, 150],
        [640, 150],
        [740, 275],
        [1060, 275],
        [1090, 100],
        [1360, 100],
        [1480, 150],
        [1630, 240],
        [1710, 370],
        [1770, 520],
        [1900, 720],
        [1900, 1070],
        [50, 1070]
    ]
    DEFAULT_DOOR_LINE = [
        [740, 275],
        [1060, 275]
    ]
    DEFAULT_DIR_REF = [900, 420]

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.roi_polygon: np.ndarray = np.array(self.DEFAULT_ROI, dtype=np.int32)
        self.door_p1: Tuple[int, int] = tuple(self.DEFAULT_DOOR_LINE[0])
        self.door_p2: Tuple[int, int] = tuple(self.DEFAULT_DOOR_LINE[1])
        self.dir_ref_pt: Tuple[int, int] = tuple(self.DEFAULT_DIR_REF)
        self.load()

    def load(self):
        if os.path.isfile(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.roi_polygon = np.array(data.get("roi_polygon", self.DEFAULT_ROI), dtype=np.int32)
                dline = data.get("door_line", self.DEFAULT_DOOR_LINE)
                self.door_p1 = tuple(dline[0])
                self.door_p2 = tuple(dline[1])
                self.dir_ref_pt = tuple(data.get("door_direction_ref", self.DEFAULT_DIR_REF))
                logger.info(f"Bolge yapilandirmasi yuklendi: {self.config_path}")
            except Exception as e:
                logger.warning(f"Yapilandirma okunamadi, varsayilan yuklendi: {e}")
                self.save()
        else:
            self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.config_path) or ".", exist_ok=True)
        data = {
            "roi_polygon": self.roi_polygon.tolist(),
            "door_line": [list(self.door_p1), list(self.door_p2)],
            "door_direction_ref": list(self.dir_ref_pt)
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# ==============================================================================
# 3. İÇ MEKAN / DIŞ MEKAN KONUM DEĞERLENDİRME (ROI FİLTRESİ)
# ==============================================================================
class SpatialROIFilter:
    """
    Kisinin kafe icinde mi yoksa cam arkasindaki dis alanda mi oldugunu filtreler.
    Histerezis (gecikmeli oylama) ile sinir titremesini engeller.
    """

    def __init__(self, polygon: np.ndarray, smoothing_frames: int = 5):
        self.polygon = polygon
        self.smoothing_frames = smoothing_frames
        self.history: Dict[int, deque] = {}
        self.confirmed_inside: Dict[int, bool] = {}

    def is_box_inside(self, box: np.ndarray) -> bool:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        h = y2 - y1

        pt_lower = (float(cx), float(y1 + 0.78 * h))
        pt_feet = (float(cx), float(y2))
        pt_center = (float(cx), float((y1 + y2) / 2.0))

        in_lower = cv2.pointPolygonTest(self.polygon, pt_lower, False) >= 0
        in_feet = cv2.pointPolygonTest(self.polygon, pt_feet, False) >= 0
        in_center = cv2.pointPolygonTest(self.polygon, pt_center, False) >= 0

        return (in_lower or in_feet) or in_center

    def update_track_presence(self, track_id: int, box: np.ndarray) -> bool:
        raw_inside = self.is_box_inside(box)

        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.smoothing_frames)
            self.confirmed_inside[track_id] = raw_inside

        self.history[track_id].append(raw_inside)

        inside_votes = sum(self.history[track_id])
        total_votes = len(self.history[track_id])

        current_status = self.confirmed_inside[track_id]
        if not current_status and inside_votes >= int(total_votes * 0.60):
            self.confirmed_inside[track_id] = True
        elif current_status and inside_votes <= int(total_votes * 0.30):
            self.confirmed_inside[track_id] = False

        return self.confirmed_inside[track_id]

    def cleanup(self, active_ids: Set[int]):
        stale = [tid for tid in self.history if tid not in active_ids]
        for tid in stale:
            del self.history[tid]
            if tid in self.confirmed_inside:
                del self.confirmed_inside[tid]


# ==============================================================================
# 4. KAPI EŞİĞİ GİRİŞ / ÇIKIŞ SAYACI
# ==============================================================================
class DoorLineCounter:
    """Gercek kapi esiginde dikey gecisleri cift yonlu sayar."""

    def __init__(
        self,
        line_p1: Tuple[int, int] = (740, 275),
        line_p2: Tuple[int, int] = (1060, 275),
        dir_ref_pt: Tuple[int, int] = (900, 420),
        cooldown_frames: int = 60,
        min_displacement: float = 35.0,
        min_dy: float = 25.0
    ):
        self.p1 = np.array(line_p1, dtype=float)
        self.p2 = np.array(line_p2, dtype=float)
        self.ref_pt = np.array(dir_ref_pt, dtype=float)
        self.cooldown_frames = cooldown_frames
        self.min_displacement = min_displacement
        self.min_dy = min_dy

        self.prev_positions: Dict[int, Tuple[float, float]] = {}
        self.trajectory_buffer: Dict[int, List[Tuple[float, float]]] = {}
        self.last_cross_frame: Dict[int, int] = {}

        self.counted_in_ids: Set[int] = set()
        self.counted_out_ids: Set[int] = set()

        self.in_count: int = 0
        self.out_count: int = 0

    @staticmethod
    def _ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @classmethod
    def _segments_intersect(cls, A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float], D: Tuple[float, float]) -> bool:
        return cls._ccw(A, C, D) != cls._ccw(B, C, D) and cls._ccw(A, B, C) != cls._ccw(A, B, D)

    def _cross_product_sign(self, pt: Tuple[float, float]) -> float:
        line_vec = self.p2 - self.p1
        return line_vec[0] * (pt[1] - self.p1[1]) - line_vec[1] * (pt[0] - self.p1[0])

    def update(self, track_id: int, current_pos: Tuple[float, float], frame_idx: int) -> Optional[str]:
        curr_x, curr_y = current_pos
        event = None

        if track_id not in self.trajectory_buffer:
            self.trajectory_buffer[track_id] = []
        self.trajectory_buffer[track_id].append((curr_x, curr_y))
        if len(self.trajectory_buffer[track_id]) > 30:
            self.trajectory_buffer[track_id].pop(0)

        if track_id in self.prev_positions:
            prev_pt = self.prev_positions[track_id]
            L1 = (self.p1[0], self.p1[1])
            L2 = (self.p2[0], self.p2[1])

            if self._segments_intersect(prev_pt, (curr_x, curr_y), L1, L2):
                last_f = self.last_cross_frame.get(track_id, -999)
                if (frame_idx - last_f) > self.cooldown_frames:
                    buf = self.trajectory_buffer[track_id]
                    if len(buf) >= 3:
                        tot_disp = np.hypot(buf[-1][0] - buf[0][0], buf[-1][1] - buf[0][1])
                        dy = buf[-1][1] - buf[0][1]
                    else:
                        tot_disp = np.hypot(curr_x - prev_pt[0], curr_y - prev_pt[1])
                        dy = curr_y - prev_pt[1]

                    if tot_disp >= self.min_displacement:
                        cp_prev = self._cross_product_sign(prev_pt)
                        cp_curr = self._cross_product_sign((curr_x, curr_y))
                        cp_ref = self._cross_product_sign(self.ref_pt)

                        # GIRIS (IN): Dikey iceri hareket (dy > 0)
                        if (cp_prev * cp_ref < 0) and (cp_curr * cp_ref >= 0) and (dy >= self.min_dy):
                            if track_id not in self.counted_in_ids:
                                self.counted_in_ids.add(track_id)
                                self.in_count += 1
                                self.last_cross_frame[track_id] = frame_idx
                                event = "IN"
                                logger.info(f"-> GIRIS (IN): Kisi #{track_id} | Toplam Giris: {self.in_count}")
                        # CIKIS (OUT): Dikey disari hareket (dy < 0)
                        elif (cp_prev * cp_ref > 0) and (cp_curr * cp_ref <= 0) and (dy <= -self.min_dy):
                            if track_id not in self.counted_out_ids:
                                self.counted_out_ids.add(track_id)
                                self.out_count += 1
                                self.last_cross_frame[track_id] = frame_idx
                                event = "OUT"
                                logger.info(f"<- CIKIS (OUT): Kisi #{track_id} | Toplam Cikis: {self.out_count}")

        self.prev_positions[track_id] = (curr_x, curr_y)
        return event

    def cleanup(self, active_ids: Set[int], current_frame: int):
        stale = [
            tid for tid, f in self.last_cross_frame.items()
            if tid not in active_ids and (current_frame - f > 150)
        ]
        for tid in stale:
            if tid in self.prev_positions:
                del self.prev_positions[tid]
            if tid in self.trajectory_buffer:
                del self.trajectory_buffer[tid]


# ==============================================================================
# 5. OPENAI CLIP ZERO-SHOT CINSIYET VE ÇOKLU ÇERÇEVE OYLAMA SİSTEMİ (GenderVoter)
# ==============================================================================
class CLIPZeroShotGenderClassifier:
    """
    OpenAI CLIP ViT-B/16 modeli ile SOTA Zero-Shot Cinsiyet Siniflandirici.
    Metin istemleri: 'a photo of a man', 'a photo of a woman'.
    """

    PROMPTS = ["a photo of a man", "a photo of a woman"]

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", device: str = "cuda"):
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        logger.info(f"OpenAI CLIP Modeli Yukleniyor ({model_name}) | Cihaz: {self.device}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=False).to(self.device)
        self.model.eval()

        logger.info("OpenAI CLIP Zero-Shot siniflandirici basariyla yuklendi!")

    def classify_crop(self, pil_crop: Image.Image) -> Tuple[float, float]:
        """Tek bir kisi kesiti icin (prob_man, prob_woman) dondurur."""
        try:
            inputs = self.processor(
                text=self.PROMPTS,
                images=pil_crop,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            return float(probs[0]), float(probs[1])
        except Exception as e:
            logger.debug(f"CLIP crop siniflandirma hatasi: {e}")
            return 0.5, 0.5


class GenderVoter:
    """
    Coklu Cerceve Agirlikli Oylama Algoritmasi (State-of-the-Art Temporal Voting):
    - Her takip ID'si icin coklu cercevelerden (5-10 crop) foto alir.
    - Buyuk, net ve kameraya yakin (yuksek cozunurluklu) fotoğraflarin oy agirligini artirir.
    - Agirlikli toplama ile titresimsiz nihai cinsiyet karari ve guven skoru belirler.
    """

    def __init__(self, clip_classifier: CLIPZeroShotGenderClassifier, max_votes_per_track: int = 10):
        self.classifier = clip_classifier
        self.max_votes = max_votes_per_track

        # Her track_id icin oy kayitlari: List[Tuple[p_man, p_woman, weight]]
        self.history: Dict[int, List[Tuple[float, float, float]]] = {}
        # Kilitlenmis/Kesinlesmis karar: Dict[int, Tuple[str, float]] -> ("Erkek"/"Kadin", confidence)
        self.locked_decisions: Dict[int, Tuple[str, float]] = {}
        # Son analiz karesi
        self.last_analyzed_frame: Dict[int, int] = {}

    def should_sample(self, track_id: int, frame_idx: int) -> bool:
        """Kisinin yeni bir cercevesinin oylamaya alinip alinmayacagini kontrol eder."""
        if track_id in self.locked_decisions:
            return False  # Karar kesinlesti

        votes_so_far = len(self.history.get(track_id, []))
        if votes_so_far >= self.max_votes:
            return False

        last_f = self.last_analyzed_frame.get(track_id, -999)
        # Her 6 karede bir yeni kaliteli ornek al (yaklasik 4 Hz)
        return (frame_idx - last_f) >= 6

    def add_sample(self, track_id: int, crop_bgr: np.ndarray, frame_idx: int):
        """Kisinin kesitini CLIP ile analiz edip agirlikli oya donusturur."""
        h, w = crop_bgr.shape[:2]
        if w < 24 or h < 45:
            return  # Cok kucuk / bulanik kesitler oylamaya dahil edilmez

        # Agirlik hesaplama: Alan (cozunurluk) karekoku * boy/en orani kalite carpani
        area = float(w * h)
        aspect_ratio = float(h) / max(1.0, float(w))
        quality = 1.3 if (1.4 <= aspect_ratio <= 3.8) else 0.7
        weight = np.sqrt(area) * quality

        # PIL RGB cevirimi
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)

        p_man, p_woman = self.classifier.classify_crop(pil_crop)

        if track_id not in self.history:
            self.history[track_id] = []

        self.history[track_id].append((p_man, p_woman, weight))
        self.last_analyzed_frame[track_id] = frame_idx

        # Eger yeterli sayida oy (>= 5) toplandiysa ve net bir fark varsa karari kilitle
        if len(self.history[track_id]) >= 5:
            gender, conf = self.get_gender(track_id)
            if conf >= 0.70:
                self.locked_decisions[track_id] = (gender, conf)

    def get_gender(self, track_id: Optional[int]) -> Tuple[str, float]:
        """Kisinin agirlikli oylama sonucundaki nihai cinsiyetini ve guven skorunu verir."""
        if track_id is None:
            return "Bilinmiyor", 0.50

        if track_id in self.locked_decisions:
            return self.locked_decisions[track_id]

        votes = self.history.get(track_id, [])
        if not votes:
            return "Bilinmiyor", 0.50

        total_weight = sum(w for _, _, w in votes)
        if total_weight <= 1e-6:
            return "Bilinmiyor", 0.50

        w_man = sum(pm * w for pm, _, w in votes) / total_weight
        w_woman = sum(pw * w for _, pw, w in votes) / total_weight

        if w_woman > w_man:
            gender = "Kadin"
            conf = w_woman
        else:
            gender = "Erkek"
            conf = w_man

        return gender, conf

    def get_distribution(self, total_inside_count: int, inside_track_ids: List[int]) -> Dict[str, int]:
        """Icerideki kisiler arasindaki Erkek, Kadin, Bilinmiyor sayilarini tam olarak hesaplar."""
        males = 0
        females = 0
        unknowns = 0

        for tid in inside_track_ids:
            g, _ = self.get_gender(tid)
            if g == "Erkek":
                males += 1
            elif g == "Kadin":
                females += 1
            else:
                unknowns += 1

        assigned = males + females + unknowns
        if assigned < total_inside_count:
            unknowns += (total_inside_count - assigned)
        elif assigned > total_inside_count:
            diff = assigned - total_inside_count
            unknowns = max(0, unknowns - diff)

        return {"Erkek": males, "Kadin": females, "Bilinmiyor": unknowns}

    def cleanup(self, active_ids: Set[int], current_frame: int):
        """Gorunmeyen ID'leri temizler (bellek yonetimi)."""
        stale = [
            tid for tid, f in self.last_analyzed_frame.items()
            if tid not in active_ids and (current_frame - f > 300)
        ]
        for tid in stale:
            self.history.pop(tid, None)
            self.locked_decisions.pop(tid, None)
            self.last_analyzed_frame.pop(tid, None)


# ==============================================================================
# 6. GÖRSELLEŞTİRME VE ÇERÇEVELER (KADIN PEMBE, ERKEK MAVİ)
# ==============================================================================
class FrameRenderer:
    """
    Sartname ve kullanici isteklerine uygun cizim:
    - Kadin: PEMBE KARE (BGR: (203, 108, 255))
    - Erkek: MAVI KARE (BGR: (255, 140, 0))
    - Dis alan: GRI (BGR: (120, 120, 120))
    """

    COLOR_BG = (18, 18, 22)
    COLOR_BORDER = (70, 70, 85)
    COLOR_ACCENT = (0, 215, 255)

    COLOR_MALE = (255, 140, 0)      # Canli Mavi (BGR)
    COLOR_FEMALE = (203, 108, 255)  # Canli Pembe (BGR)
    COLOR_UNKNOWN = (180, 180, 180) # Bilinmiyor
    COLOR_OUTSIDE = (110, 110, 120) # Dis Alan Gri

    COLOR_DOOR = (255, 120, 0)
    COLOR_IN = (46, 204, 113)
    COLOR_OUT = (52, 73, 235)

    def __init__(self, zone_config: ZoneConfig, door_counter: DoorLineCounter, gender_voter: GenderVoter):
        self.zone = zone_config
        self.door = door_counter
        self.voter = gender_voter

    def draw_overlays(
        self,
        frame: np.ndarray,
        inside_count: int,
        outside_count: int,
        inside_track_ids: List[int],
        device_name: str,
        fps: float
    ):
        self._draw_zones(frame)
        self._draw_dashboard(frame, inside_count, outside_count, inside_track_ids, device_name, fps)

    def _draw_zones(self, frame: np.ndarray):
        # 1. Kafe Ici ROI Siniri
        overlay = frame.copy()
        cv2.polylines(frame, [self.zone.roi_polygon], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.fillPoly(overlay, [self.zone.roi_polygon], (0, 45, 0))
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        cv2.putText(frame, "[Kafe Ici Alan Siniri (ROI)]", (self.zone.roi_polygon[0][0] + 10, self.zone.roi_polygon[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # 2. Kapi Esigi Cizgisi
        p1 = self.door.p1.astype(int)
        p2 = self.door.p2.astype(int)
        cv2.line(frame, tuple(p1), tuple(p2), self.COLOR_DOOR, 4, cv2.LINE_AA)
        cv2.circle(frame, tuple(p1), 6, (255, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(p2), 6, (255, 200, 0), -1, cv2.LINE_AA)

        # Yon Oklari
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.arrowedLine(frame, (mid[0], mid[1] - 40), (mid[0], mid[1] + 40), (0, 255, 255), 3, tipLength=0.35)
        cv2.putText(frame, "GIRIS (IN) v", (mid[0] + 12, mid[1] + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.COLOR_IN, 2, cv2.LINE_AA)
        cv2.putText(frame, "^ CIKIS (OUT)", (mid[0] + 12, mid[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.COLOR_OUT, 2, cv2.LINE_AA)
        cv2.putText(frame, "[Kapi Esigi Cizgisi]", (mid[0] - 80, mid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.COLOR_DOOR, 1, cv2.LINE_AA)

    def _draw_dashboard(
        self,
        frame: np.ndarray,
        inside_count: int,
        outside_count: int,
        inside_track_ids: List[int],
        device_name: str,
        fps: float
    ):
        panel_w, panel_h = 560, 250
        x1, y1 = 20, 20
        x2, y2 = x1 + panel_w, y1 + panel_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLOR_BG, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLOR_BORDER, 2)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Baslik
        cv2.putText(frame, "ARABICA x YOUREYE", (x1 + 16, y1 + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, self.COLOR_ACCENT, 2, cv2.LINE_AA)
        cv2.putText(frame, "Asama 1: OpenAI CLIP Zero-Shot Cinsiyet & Kisi Sayimi", (x1 + 16, y1 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"YOLO11m + CLIP ViT-B/16 | {device_name} | {fps:.1f} FPS", (x1 + 16, y1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 170), 1, cv2.LINE_AA)

        cv2.line(frame, (x1 + 16, y1 + 70), (x2 - 16, y1 + 70), (60, 60, 70), 1)

        # ANLIK KAFE DURUMU
        dist = self.voter.get_distribution(inside_count, inside_track_ids)

        cv2.putText(frame, "KAFE ICI ANLIK SAYIM VE CINSIYET DAGILIMI", (x1 + 16, y1 + 88),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

        c_box_y = y1 + 98
        c_w = 118
        c_h = 56
        gap = 14

        # 1. Toplam Iceride
        b1_x = x1 + 16
        cv2.rectangle(frame, (b1_x, c_box_y), (b1_x + c_w, c_box_y + c_h), (45, 38, 20), -1)
        cv2.rectangle(frame, (b1_x, c_box_y), (b1_x + c_w, c_box_y + c_h), (255, 180, 40), 2)
        cv2.putText(frame, "TOPLAM ICERIDE", (b1_x + 8, c_box_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, str(inside_count), (b1_x + 42, c_box_y + 46),
                    cv2.FONT_HERSHEY_DUPLEX, 0.90, (255, 180, 40), 2, cv2.LINE_AA)

        # 2. Erkek (Mavi Kutu)
        b2_x = b1_x + c_w + gap
        cv2.rectangle(frame, (b2_x, c_box_y), (b2_x + c_w, c_box_y + c_h), (35, 25, 15), -1)
        cv2.rectangle(frame, (b2_x, c_box_y), (b2_x + c_w, c_box_y + c_h), self.COLOR_MALE, 2)
        cv2.putText(frame, "ERKEK (MAVI)", (b2_x + 16, c_box_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, str(dist["Erkek"]), (b2_x + 44, c_box_y + 46),
                    cv2.FONT_HERSHEY_DUPLEX, 0.90, self.COLOR_MALE, 2, cv2.LINE_AA)

        # 3. Kadin (Pembe Kutu)
        b3_x = b2_x + c_w + gap
        cv2.rectangle(frame, (b3_x, c_box_y), (b3_x + c_w, c_box_y + c_h), (35, 18, 30), -1)
        cv2.rectangle(frame, (b3_x, c_box_y), (b3_x + c_w, c_box_y + c_h), self.COLOR_FEMALE, 2)
        cv2.putText(frame, "KADIN (PEMBE)", (b3_x + 14, c_box_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 180, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, str(dist["Kadin"]), (b3_x + 44, c_box_y + 46),
                    cv2.FONT_HERSHEY_DUPLEX, 0.90, self.COLOR_FEMALE, 2, cv2.LINE_AA)

        # 4. Bilinmiyor
        b4_x = b3_x + c_w + gap
        cv2.rectangle(frame, (b4_x, c_box_y), (b4_x + c_w, c_box_y + c_h), (30, 30, 35), -1)
        cv2.rectangle(frame, (b4_x, c_box_y), (b4_x + c_w, c_box_y + c_h), self.COLOR_UNKNOWN, 1)
        cv2.putText(frame, "BILINMIYOR", (b4_x + 20, c_box_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 210), 1, cv2.LINE_AA)
        cv2.putText(frame, str(dist["Bilinmiyor"]), (b4_x + 44, c_box_y + 46),
                    cv2.FONT_HERSHEY_DUPLEX, 0.90, self.COLOR_UNKNOWN, 2, cv2.LINE_AA)

        # KAPI GECISLERI
        g_y = y1 + 176
        cv2.putText(frame, "KAPI ESIK GECISLERI", (x1 + 16, g_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, (200, 220, 255), 1, cv2.LINE_AA)

        k_txt = (
            f"Giris (IN): {self.door.in_count}   |   "
            f"Cikis (OUT): {self.door.out_count}   |   "
            f"Net Degisim: {self.door.in_count - self.door.out_count:+d}   |   "
            f"Dis Alan: {outside_count}"
        )
        cv2.putText(frame, k_txt, (x1 + 16, g_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # SARTNAME ZORUNLU TEKNIK NOT UYARISI
        not_txt = "* Teknik Not: Cinsiyet tahmini gorsel CLIP model ciktisidir; kesin kimlik tespiti degildir."
        cv2.putText(frame, not_txt, (x1 + 16, g_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 200, 220), 1, cv2.LINE_AA)

    def draw_person(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        track_id: Optional[int],
        conf: float,
        is_inside: bool
    ):
        x1, y1, x2, y2 = map(int, box)
        cx = int((x1 + x2) / 2)

        if not is_inside:
            color = self.COLOR_OUTSIDE
            label = f"ID:#{track_id} [Dis Alan]" if track_id is not None else "[Dis Alan]"
            thickness = 1
        else:
            gender, g_conf = self.voter.get_gender(track_id)
            if gender == "Kadin":
                color = self.COLOR_FEMALE  # PEMBE KARE
                label = f"ID:#{track_id} | Kadin (%{g_conf*100:.0f})"
            elif gender == "Erkek":
                color = self.COLOR_MALE    # MAVI KARE
                label = f"ID:#{track_id} | Erkek (%{g_conf*100:.0f})"
            else:
                color = self.COLOR_UNKNOWN
                label = f"ID:#{track_id} | Analiz..." if track_id is not None else "Person"
            thickness = 2

        # Bounding box cizimi
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        cv2.circle(frame, (cx, y2), 4, color, -1, cv2.LINE_AA)

        # Etiket arkaplani ve metin
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        lbl_y1 = max(0, y1 - th - bl - 4)
        lbl_y2 = y1
        cv2.rectangle(frame, (x1, lbl_y1), (x1 + tw + 8, lbl_y2), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - bl - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0) if is_inside else (255, 255, 255), 1, cv2.LINE_AA)


# ==============================================================================
# 7. ANA VİDEO İŞLEME BORU HATTI
# ==============================================================================
class AdvancedPersonPipeline:

    def __init__(
        self,
        video_path: str = "videos/kisi_sayimi.mp4",
        output_path: str = "outputs/asama1_clip_sonuc.mp4",
        config_path: str = DEFAULT_CONFIG_PATH,
        model_path: str = "yolo11m.pt",
        conf_thresh: float = 0.28,
        iou_thresh: float = 0.55,
        imgsz: int = 1280,
        device: Optional[str] = None
    ):
        self.video_path = video_path
        self.output_path = output_path
        self.config_path = config_path
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz

        if device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"
            self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        else:
            self.device = device
            self.device_name = str(device)

        self.zone_cfg = ZoneConfig(config_path)
        self.roi_filter = SpatialROIFilter(self.zone_cfg.roi_polygon)
        self.door_counter = DoorLineCounter(
            line_p1=self.zone_cfg.door_p1,
            line_p2=self.zone_cfg.door_p2,
            dir_ref_pt=self.zone_cfg.dir_ref_pt
        )

        # Zero-Shot CLIP ve Coklu Cerceve Oylama
        self.clip_classifier = CLIPZeroShotGenderClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
        self.gender_voter = GenderVoter(self.clip_classifier)
        self.renderer = FrameRenderer(self.zone_cfg, self.door_counter, self.gender_voter)

        logger.info(f"YOLO11m Yukleniyor: {self.model_path} | Cihaz: {self.device} ({self.device_name})")
        self.model = YOLO(self.model_path)

    def run(self):
        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Video bulunamadi: {self.video_path}")

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video acilamadi: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {width}x{height} @ {fps:.1f} FPS | Toplam Kare: {total_frames}")

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
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # 1. YOLO11m + ByteTrack Takip
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False
                )

                inside_count = 0
                outside_count = 0
                active_ids = set()
                inside_track_ids = []

                if results and results[0].boxes:
                    raw_b = results[0].boxes.xyxy.cpu().numpy()
                    raw_c = results[0].boxes.conf.cpu().numpy()
                    raw_ids = (
                        results[0].boxes.id.cpu().numpy().astype(int)
                        if results[0].boxes.id is not None
                        else None
                    )

                    boxes, confs, track_ids = suppress_nested_boxes(
                        raw_b, raw_c, raw_ids, containment_thresh=0.72
                    )

                    if track_ids is not None:
                        active_ids = set(track_ids)

                    for idx in range(len(boxes)):
                        box = boxes[idx]
                        conf = confs[idx]
                        tid = track_ids[idx] if track_ids is not None else None

                        if tid is not None:
                            bx = (box[0] + box[2]) / 2.0
                            by = float(box[3])
                            self.door_counter.update(tid, (bx, by), frame_count)

                        if tid is not None:
                            is_inside = self.roi_filter.update_track_presence(tid, box)
                        else:
                            is_inside = self.roi_filter.is_box_inside(box)

                        if is_inside:
                            inside_count += 1
                            if tid is not None:
                                inside_track_ids.append(tid)

                                # CLIP Zero-Shot Cinsiyet Ornekleme
                                if self.gender_voter.should_sample(tid, frame_count):
                                    x1, y1, x2, y2 = map(int, box)
                                    x1, y1 = max(0, x1), max(0, y1)
                                    x2, y2 = min(width, x2), min(height, y2)
                                    crop = frame[y1:y2, x1:x2]
                                    if crop.size > 0:
                                        self.gender_voter.add_sample(tid, crop, frame_count)
                        else:
                            outside_count += 1

                        self.renderer.draw_person(frame, box, tid, conf, is_inside)

                self.roi_filter.cleanup(active_ids)
                self.door_counter.cleanup(active_ids, frame_count)
                self.gender_voter.cleanup(active_ids, frame_count)

                # Anlik FPS
                if frame_count % 10 == 0:
                    now = time.time()
                    cur_fps = 10.0 / max(1e-5, now - last_calc_time)
                    last_calc_time = now

                self.renderer.draw_overlays(
                    frame=frame,
                    inside_count=inside_count,
                    outside_count=outside_count,
                    inside_track_ids=inside_track_ids,
                    device_name=self.device_name,
                    fps=cur_fps
                )

                writer.write(frame)

                if frame_count % 150 == 0 or frame_count == total_frames:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / max(1e-5, elapsed)
                    prog = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    dist = self.gender_voter.get_distribution(inside_count, inside_track_ids)
                    logger.info(
                        f"Kare {frame_count:4d}/{total_frames} ({prog:5.1f}%) | "
                        f"Hiz: {avg_fps:4.1f} FPS | Iceride: {inside_count:2d} (Erkek: {dist['Erkek']}, Kadin: {dist['Kadin']}) | "
                        f"Giris: {self.door_counter.in_count} | Cikis: {self.door_counter.out_count}"
                    )

        finally:
            cap.release()
            writer.release()

        total_elapsed = time.time() - start_time
        avg_fps = frame_count / max(1e-5, total_elapsed)

        logger.info("=" * 60)
        logger.info("ASAMA 1 (OPENAI CLIP SOTA) ISLEMI BASARIYLA TAMAMLANDI!")
        logger.info(f"Toplam Islenen Kare       : {frame_count}")
        logger.info(f"Toplam Sure               : {total_elapsed:.2f} saniye")
        logger.info(f"Ortalama Islem Hizi       : {avg_fps:.2f} FPS")
        logger.info(f"Toplam GIRIS Sayisi       : {self.door_counter.in_count}")
        logger.info(f"Toplam CIKIS Sayisi       : {self.door_counter.out_count}")
        logger.info(f"Cikti Video Dosyasi       : {os.path.abspath(self.output_path)}")
        logger.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Arabica x YourEye Hackathon 2026 - Asama 1 (CLIP)")
    parser.add_argument("--video", type=str, default="videos/kisi_sayimi.mp4", help="Giris video dosyasi")
    parser.add_argument("--output", type=str, default="outputs/asama1_clip_sonuc.mp4", help="Cikti video dosyasi")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Bolge yapilandirma JSON dosyasi")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="YOLO model dosyasi")
    parser.add_argument("--conf", type=float, default=0.28, help="Guven skoru esigi")
    parser.add_argument("--iou", type=float, default=0.55, help="IoU NMS esigi")
    parser.add_argument("--imgsz", type=int, default=1280, help="Giris cozunurlugu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = AdvancedPersonPipeline(
        video_path=args.video,
        output_path=args.output,
        config_path=args.config,
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz
    )
    pipeline.run()
