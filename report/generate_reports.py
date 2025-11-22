"""
Manueller Report-Generator.

Erstellt bei Bedarf die vorher automatischen Dateien:
- report/complexity_report.png (aus Rest-Klasse, Kanten-Bereinigung)
- report/report.png (Beispiele aus Bruch/Normal)
"""

import argparse
import random
import sys
from pathlib import Path

import cv2

# Projektwurzel in den Pfad legen, damit "scripts" importiert werden kann
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import bruch, rest  # noqa: E402


def build_complexity_report(sorted_dir: Path, output_file: Path) -> None:
    rest_dir = sorted_dir / "Rest"
    if not rest_dir.exists():
        print(f"[report] Kein Rest-Ordner gefunden unter {rest_dir}")
        return

    report_data = []
    for img_path in rest_dir.rglob("*"):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        _, edges_orig, binary_orig = rest.calculate_edge_sum(image)
        binary_clean = rest.remove_small_artifacts(binary_orig, rest.MIN_OBJECT_AREA)
        edges_clean = cv2.Canny(binary_clean, 50, 150)
        report_data.append((str(img_path), edges_orig, edges_clean))

    if not report_data:
        print(f"[report] Keine geeigneten Bilder f�r complexity_report in {rest_dir}")
        return

    random.shuffle(report_data)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rest.create_edge_report(report_data, str(output_file))


def build_bruch_report(sorted_dir: Path, output_file: Path) -> None:
    candidates = []
    for cls in ("Bruch", "Normal"):
        class_dir = sorted_dir / cls
        if not class_dir.exists():
            continue
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                candidates.append(str(img_path))

    if not candidates:
        print(f"[report] Keine Bilder f�r report.png in {sorted_dir}")
        return

    random.shuffle(candidates)
    samples = candidates[:5]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    bruch.create_visual_report(samples, str(output_file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manuelle Report-Erstellung f�r die Pipeline.")
    parser.add_argument(
        "--sorted-dir",
        default=str(ROOT / "data" / "sorted"),
        help="Pfad zum Sortier-Output (Standard: data/sorted)",
    )
    parser.add_argument(
        "--skip-complexity",
        action="store_true",
        help="complexity_report.png auslassen",
    )
    parser.add_argument(
        "--skip-bruch",
        action="store_true",
        help="report.png auslassen",
    )
    args = parser.parse_args()

    sorted_dir = Path(args.sorted_dir)
    report_dir = ROOT / "report"

    if not args.skip_complexity:
        build_complexity_report(sorted_dir, report_dir / "complexity_report.png")
    if not args.skip_bruch:
        build_bruch_report(sorted_dir, report_dir / "report.png")


if __name__ == "__main__":
    main()
