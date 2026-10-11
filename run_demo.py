"""
Arabica x YourEye Hackathon 2026 - Ana Calistirici (Demo Runner)
Bu script, her iki asamanin da tek komutla veya ayri ayri calistirilmasini saglar.

Kullanim:
  python run_demo.py --asama 1    # Sadece Asama 1 (Kafe Kisi Sayimi & Zero-Shot CLIP)
  python run_demo.py --asama 2    # Sadece Asama 2 (Sehir Girisi Arac Sayimi & Plaka OCR)
  python run_demo.py --asama all  # Her iki asamayi sirasiyla calistirir
"""

import sys
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("HackathonDemo")


def run_asama1():
    logger.info("=" * 60)
    logger.info("ASAMA 1: KAFE ICERISINDE KISI SAYIMI & CLIP ZERO-SHOT CINSIYET BASLATILIYOR...")
    logger.info("=" * 60)
    import asama1_kisi_sayimi
    pipeline = asama1_kisi_sayimi.AdvancedPersonPipeline(
        video_path="videos/kisi_sayimi.mp4",
        output_path="outputs/asama1_clip_sonuc.mp4"
    )
    pipeline.run()


def run_asama2(max_frames=1200):
    logger.info("=" * 60)
    logger.info("ASAMA 2: SEHRE GIREN ARAC SAYIMI & PLAKA TESPITI BASLATILIYOR...")
    logger.info("=" * 60)
    import asama2_arac_sayimi_plaka
    pipeline = asama2_arac_sayimi_plaka.AdvancedVehiclePipeline(
        video_path="videos/arac_sayimi.mp4",
        output_path="outputs/asama2_arac_plaka_sonuc.mp4",
        max_frames=max_frames
    )
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabica x YourEye Hackathon Demo Runner")
    parser.add_argument("--asama", type=str, choices=["1", "2", "all"], default="all",
                        help="Calistirilacak asama: '1', '2' veya 'all'")
    parser.add_argument("--max-frames-asama2", type=int, default=1200,
                        help="Asama 2 icin islenecek maksimum kare sayisi (varsayilan: 1200)")
    args = parser.parse_args()

    if args.asama in ["1", "all"]:
        run_asama1()

    if args.asama in ["2", "all"]:
        run_asama2(max_frames=args.max_frames_asama2)

    logger.info("=" * 60)
    logger.info("TUM GOREVLER BASARIYLA TAMAMLANDI! CIKTILAR 'outputs/' KLASORUNDE.")
    logger.info("=" * 60)
