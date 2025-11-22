import cv2
import numpy as np
import os


def get_symmetry_score(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    total_pixels = cv2.countNonZero(mask)
    if total_pixels == 0:
        return 0.0

    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return 0.0

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    h, w = mask.shape[:2]

    symmetric_core = mask.copy()

    for angle in range(60, 360, 60):
        rot_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        rotated_mask = cv2.warpAffine(mask, rot_matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

        symmetric_core = cv2.bitwise_and(symmetric_core, rotated_mask)

    asymmetric_mask = cv2.subtract(mask, symmetric_core)
    asymmetric_pixel_count = cv2.countNonZero(asymmetric_mask)

    error_ratio = asymmetric_pixel_count / total_pixels

    score = (1.0 - error_ratio) * 100.0

    return max(0.0, min(100.0, round(score, 2)))


def run_symmetry_check(sorted_dir):
    print("\n[symmetrie.py] Starte Symmetrie-Analyse f�r Klasse 'Normal'...")

    normal_path = os.path.join(sorted_dir, "Normal")

    if not os.path.exists(normal_path):
        print(f"Warnung: Ordner {normal_path} existiert nicht. �berspringe Symmetrie-Check.")
        return

    count = 0
    scores = []

    for root, _, files in os.walk(normal_path):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            file_path = os.path.join(root, filename)
            image = cv2.imread(file_path)

            if image is None:
                continue

            score = get_symmetry_score(image)
            scores.append(score)

            new_filename = f"{score:05.2f}_{filename}"
            new_path = os.path.join(root, new_filename)

            try:
                os.rename(file_path, new_path)
                count += 1
            except OSError as e:
                print(f"Fehler beim Umbenennen von {filename}: {e}")

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"[symmetrie.py] Abgeschlossen. {count} Bilder bewertet und umbenannt.")
    print(f"   -> Durchschnittlicher Symmetrie-Score: {avg_score:.2f}")
