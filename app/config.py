"""Central configuration for the fryum quality inspection pipeline."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "Images"
NORMAL_DIR = IMAGES_DIR / "Normal"
ANOMALY_DIR = IMAGES_DIR / "Anomaly"
ANNOTATION_FILE = DATA_DIR / "image_anno.csv"
MASK_DIR = DATA_DIR / "Masks"

OUTPUT_ROOT = BASE_DIR / "output"
RUN_IMAGE_DIR = OUTPUT_ROOT / "classified"
REPORT_DIR = OUTPUT_ROOT / "reports"

AGGREGATED_CLASS_MAPPING = {
    "normal": "Normal",
    "burnt": "Farbfehler",
    "different colour spot": "Farbfehler",
    "different colour spot,similar colour spot": "Farbfehler",
    "similar colour spot": "Farbfehler",
    "similar colour spot,other": "Farbfehler",
    "similar colour spot,small scratches": "Farbfehler",
    "small scratches": "Farbfehler",
    "corner or edge breakage": "Bruch",
    "corner or edge breakage,small scratches": "Bruch",
    "middle breakage": "Bruch",
    "middle breakage,small scratches": "Bruch",
    "middle breakage,similar colour spot": "Bruch",
    "fryum stuck together": "Rest",
}

TARGET_CLASSES = ("Normal", "Farbfehler", "Bruch", "Rest")
