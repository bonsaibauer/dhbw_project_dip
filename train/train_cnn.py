import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import bruch, ergebnis, farb, rest


def load_ground_truth(csv_path: str) -> Dict[str, str]:
    """Laedt die Ground-Truth-Labels aus der CSV und normalisiert die Keys."""
    gt = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        idx_image = header.index("image")
        idx_label = header.index("label")
        for line in f:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) <= max(idx_image, idx_label):
                continue
            full_path = cols[idx_image]
            parts = full_path.split("/")
            key = f"{parts[-2]}/{parts[-1]}".lower() if len(parts) >= 2 else full_path.lower()
            gt[key] = ergebnis.get_true_label(cols[idx_label])
    return gt


def normalize_filename(filename: str) -> str:
    """Passt Dateinamen an die Logik aus ergebnis.py an (Score-Prefix entfernen)."""
    name = filename.lower()
    if "_" in name:
        prefix, rest_name = name.split("_", 1)
        try:
            float(prefix)
            name = rest_name
        except ValueError:
            pass
    return name


def evaluate_sorted_dir(sorted_dir: str, ground_truth: Dict[str, str]) -> Dict[str, object]:
    """Berechnet die Genauigkeit der Sortierung gegen die Ground-Truth-CSV."""
    categories = ["Normal", "Bruch", "Farbfehler", "Rest"]
    stats = {c: {"soll": 0, "hits": 0} for c in categories}
    for true_label in ground_truth.values():
        if true_label in stats:
            stats[true_label]["soll"] += 1

    total_hits = 0
    processed = 0

    for cat in categories:
        folder_path = os.path.join(sorted_dir, cat)
        if not os.path.exists(folder_path):
            continue
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                fname_norm = normalize_filename(filename)
                reconstructed_key = fname_norm.replace("_", "/", 1)

                true_cat = None
                if reconstructed_key in ground_truth:
                    true_cat = ground_truth[reconstructed_key]
                else:
                    matches = [
                        val
                        for k, val in ground_truth.items()
                        if k.endswith(f"/{fname_norm}") or k == fname_norm
                    ]
                    if matches:
                        true_cat = matches[0]

                if true_cat is None:
                    continue

                processed += 1
                if true_cat == cat:
                    total_hits += 1
                    stats[cat]["hits"] += 1

    total_soll = sum(v["soll"] for v in stats.values())
    accuracy = (total_hits / total_soll * 100) if total_soll > 0 else 0.0

    per_class = {
        c: {
            "soll": stats[c]["soll"],
            "hits": stats[c]["hits"],
            "acc": (stats[c]["hits"] / stats[c]["soll"] * 100) if stats[c]["soll"] else 0.0,
        }
        for c in categories
    }
    return {
        "accuracy": accuracy,
        "total_hits": total_hits,
        "total_soll": total_soll,
        "processed": processed,
        "per_class": per_class,
    }


def make_color_detector(params: Dict[str, float]):
    """Erzeugt eine parametrische Variante von farb.detect_defects."""

    def detect_defects(image, spot_threshold=None, debug=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask_obj = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        kernel_erode = np.ones((13, 13), np.uint8)
        mask_analysis = cv2.erode(mask_obj, kernel_erode, iterations=1)
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        blackhat_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_morph)
        _, mask_defects_contrast = cv2.threshold(
            blackhat_img, params["contrast_threshold"], 255, cv2.THRESH_BINARY
        )

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_burn = np.array([0, params["burn_sat_min"], 0])
        upper_burn = np.array([180, 255, params["burn_value_max"]])
        mask_burn = cv2.inRange(hsv, lower_burn, upper_burn)

        combined_defects = cv2.bitwise_or(mask_defects_contrast, mask_burn)
        valid_defects = cv2.bitwise_and(combined_defects, combined_defects, mask=mask_analysis)
        valid_defects = cv2.morphologyEx(valid_defects, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(valid_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        significant_contours = []
        total_defect_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > params["min_spot_size"]:
                significant_contours.append(cnt)
                total_defect_area += area

        thr = params["spot_threshold"] if spot_threshold is None else spot_threshold
        is_defective = total_defect_area > thr

        return {
            "is_defective": is_defective,
            "spot_area": total_defect_area,
            "contours": significant_contours,
        }

    def run_color_check(sorted_dir):
        defect_dir = os.path.join(sorted_dir, "Farbfehler")
        os.makedirs(defect_dir, exist_ok=True)
        moved = 0

        for cls in ["Normal"]:
            class_path = os.path.join(sorted_dir, cls)
            if not os.path.exists(class_path):
                continue
            for root, _, files in os.walk(class_path):
                for file_name in files:
                    if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    file_path = os.path.join(root, file_name)
                    image = cv2.imread(file_path)
                    if image is None:
                        continue
                    result = detect_defects(image)
                    if result["is_defective"]:
                        cv2.drawContours(image, result["contours"], -1, (0, 0, 255), 2)
                        for cnt in result["contours"]:
                            (x, y), radius = cv2.minEnclosingCircle(cnt)
                            cv2.circle(image, (int(x), int(y)), int(radius) + 8, (0, 0, 255), 2)
                        target_path = os.path.join(defect_dir, file_name)
                        cv2.imwrite(target_path, image)
                        try:
                            os.remove(file_path)
                            moved += 1
                        except OSError:
                            pass
        return moved

    farb.detect_defects = detect_defects
    farb.run_color_check = run_color_check


def apply_params(params: Dict[str, float]):
    """Schreibt die gewaehlten Parameter in die Module (Monkey-Patching)."""
    bruch.OUTER_BREAK_SENSITIVITY = params["bruch_outer_sens"]
    bruch.MAX_RADIUS_JUMP = params["bruch_max_radius_jump"]
    bruch.LOCAL_VARIANCE_THRESHOLD = params["bruch_local_var"]
    bruch.MIN_OBJECT_AREA = int(params["bruch_min_obj_area"])
    bruch.MIN_WINDOWS_FOR_BRUCH = int(params["bruch_min_windows"])
    bruch.MIN_WINDOW_AREA = int(params["bruch_min_window_area"])
    bruch.MAX_ALLOWED_CORNERS = int(params["bruch_max_corners"])
    bruch.MIN_PEAK_DISTANCE = int(params["bruch_min_peak_dist"])

    rest.MAX_EDGE_SUM = int(params["rest_max_edge_sum"])
    rest.MIN_EDGE_SUM = int(params["rest_min_edge_sum"])
    rest.MIN_OBJECT_AREA = int(params["rest_min_object_area"])

    make_color_detector(params)


def sample_params(rng: random.Random) -> Dict[str, float]:
    """Erzeugt eine zufaellige Parameter-Kombination innerhalb sinnvoller Grenzen."""
    return {
        # bruch.py
        "bruch_outer_sens": rng.uniform(0.55, 0.95),
        "bruch_max_radius_jump": rng.uniform(2.0, 16.0),
        "bruch_local_var": rng.uniform(1.5, 8.5),
        "bruch_min_obj_area": rng.randint(2500, 12000),
        "bruch_min_windows": rng.randint(2, 8),
        "bruch_min_window_area": rng.randint(3, 35),
        "bruch_max_corners": rng.randint(2, 6),
        "bruch_min_peak_dist": rng.randint(30, 140),
        # rest.py
        "rest_max_edge_sum": rng.randint(2400, 5200),
        "rest_min_edge_sum": rng.randint(1500, 3600),
        "rest_min_object_area": rng.randint(100, 900),
        # farb.py
        "spot_threshold": rng.randint(10, 120),
        "burn_sat_min": rng.randint(10, 80),
        "burn_value_max": rng.randint(60, 150),
        "min_spot_size": rng.randint(15, 110),
        "contrast_threshold": rng.randint(20, 80),
    }


def write_best_result(best: Dict[str, object], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    best_file = output_dir / "best_result.txt"
    lines = [
        f"Beste Genauigkeit: {best['metrics']['accuracy']:.2f}% (Run: {best['run_id']})",
        f"Treffer: {best['metrics']['total_hits']} / {best['metrics']['total_soll']} (verarbeitete Bilder: {best['metrics']['processed']})",
        "",
        "Empfohlene Parameter-Updates:",
    ]
    p = best["params"]
    lines += [
        f"- bruch.OUTER_BREAK_SENSITIVITY = {p['bruch_outer_sens']:.3f}",
        f"- bruch.MAX_RADIUS_JUMP = {p['bruch_max_radius_jump']:.2f}",
        f"- bruch.LOCAL_VARIANCE_THRESHOLD = {p['bruch_local_var']:.2f}",
        f"- bruch.MIN_OBJECT_AREA = {int(p['bruch_min_obj_area'])}",
        f"- bruch.MIN_WINDOWS_FOR_BRUCH = {int(p['bruch_min_windows'])}",
        f"- bruch.MIN_WINDOW_AREA = {int(p['bruch_min_window_area'])}",
        f"- bruch.MAX_ALLOWED_CORNERS = {int(p['bruch_max_corners'])}",
        f"- bruch.MIN_PEAK_DISTANCE = {int(p['bruch_min_peak_dist'])}",
        f"- rest.MAX_EDGE_SUM = {int(p['rest_max_edge_sum'])}",
        f"- rest.MIN_EDGE_SUM = {int(p['rest_min_edge_sum'])}",
        f"- rest.MIN_OBJECT_AREA = {int(p['rest_min_object_area'])}",
        f"- farb spot_threshold = {int(p['spot_threshold'])}",
        f"- farb burn_sat_min = {int(p['burn_sat_min'])}",
        f"- farb burn_value_max = {int(p['burn_value_max'])}",
        f"- farb min_spot_size = {int(p['min_spot_size'])}",
        f"- farb contrast_threshold = {int(p['contrast_threshold'])}",
        "",
        f"Ergebnisordner: {best['run_dir']}",
    ]
    best_file.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(run_id: int, params: Dict[str, float], cfg: argparse.Namespace, ground_truth):
    run_dir = Path(cfg.output_dir) / f"run_{run_id:02d}"
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    apply_params(params)

    bruch.sort_images(cfg.processed_dir, str(run_dir))
    rest.run_complexity_check(str(run_dir))
    farb.run_color_check(str(run_dir))

    metrics = evaluate_sorted_dir(str(run_dir), ground_truth)

    (run_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"run_id": run_id, "params": params, "metrics": metrics, "run_dir": str(run_dir)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random Search fuer Parameter in bruch.py, rest.py, farb.py (30 Laeufe fuer >90.7% Zielgenauigkeit).",
    )
    parser.add_argument("--processed-dir", default="data/processed", help="Pfad zu den vorverarbeiteten Bildern.")
    parser.add_argument("--anno-file", default="data/image_anno.csv", help="CSV mit Ground-Truth-Labels.")
    parser.add_argument("--output-dir", default="train/runs", help="Basisordner fuer Zwischenergebnisse.")
    parser.add_argument("--runs", type=int, default=30, help="Anzahl der Versuche.")
    parser.add_argument("--seed", type=int, default=42, help="Seed fuer die Stichprobe.")
    return parser.parse_args()


def main():
    cfg = parse_args()
    rng = random.Random(cfg.seed)
    ground_truth = load_ground_truth(cfg.anno_file)

    processed_root = Path(cfg.processed_dir)
    has_images = False
    if processed_root.exists():
        for _, _, files in os.walk(processed_root):
            if any(fn.lower().endswith((".jpg", ".jpeg", ".png")) for fn in files):
                has_images = True
                break
    if not has_images:
        print(
            f"Kein verarbeitetes Dataset gefunden unter {cfg.processed_dir}. "
            "Bitte zuerst segmentierung.prepare_dataset oder main.py ausfuehren."
        )
        return

    best = None
    for run_id in range(1, cfg.runs + 1):
        params = sample_params(rng)
        result = run_experiment(run_id, params, cfg, ground_truth)
        acc = result["metrics"]["accuracy"]
        print(f"[Run {run_id:02d}] Accuracy: {acc:.2f}%")
        if best is None or acc > best["metrics"]["accuracy"]:
            best = result

    if best:
        write_best_result(best, Path(cfg.output_dir))
        print(
            f"\nBestes Ergebnis: {best['metrics']['accuracy']:.2f}% "
            f"(Run {best['run_id']:02d}) -> Details in {cfg.output_dir}"
        )
        if best["metrics"]["accuracy"] >= 90.7:
            print("Zielwert 90.7% erreicht oder ueberschritten.")
        else:
            print("Zielwert 90.7% nicht erreicht. Parameterbereiche erweitern oder Laeufe erhoehen.")


if __name__ == "__main__":
    main()
