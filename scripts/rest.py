import cv2
import numpy as np
import os
import shutil

MAX_EDGE_SUM = 3031
MIN_EDGE_SUM = 2740
MIN_OBJECT_AREA = 250


def remove_small_artifacts(binary_img, min_area):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    clean_binary = np.zeros_like(binary_img)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_binary[labels == i] = 255

    return clean_binary


def calculate_edge_sum(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(binary, 50, 150)
    total_edge_length = cv2.countNonZero(edges)

    return total_edge_length, edges, binary


def run_complexity_check(sorted_dir):
    print("\n[rest.py] Starte intelligente Komplexitäts-Prüfung...")
    print(f"   - Limit: {MAX_EDGE_SUM} Kanten-Pixel")
    print(f"   - Artefakt-Filter: Objekte unter {MIN_OBJECT_AREA}px werden ignoriert")

    rest_dir = os.path.join(sorted_dir, "Rest")
    os.makedirs(rest_dir, exist_ok=True)

    check_classes = ["Normal", "Bruch"]
    moved_count = 0
    kept_count = 0

    for cls in check_classes:
        class_path = os.path.join(sorted_dir, cls)

        for root, _, files in os.walk(class_path):
            for file_name in files:
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                file_path = os.path.join(root, file_name)
                image = cv2.imread(file_path)
                if image is None:
                    continue

                edge_sum, edges_orig, binary_orig = calculate_edge_sum(image)

                if edge_sum < MIN_EDGE_SUM:
                    target_path = os.path.join(rest_dir, file_name)
                    shutil.move(file_path, target_path)
                    moved_count += 1
                    print(f"   -> REST (Fragment): {file_name} (Sum: {edge_sum} < {MIN_EDGE_SUM})")
                    continue

                if edge_sum > MAX_EDGE_SUM:
                    binary_clean = remove_small_artifacts(binary_orig, MIN_OBJECT_AREA)

                    edges_clean = cv2.Canny(binary_clean, 50, 150)
                    clean_edge_sum = cv2.countNonZero(edges_clean)

                    if clean_edge_sum > MAX_EDGE_SUM:
                        target_path = os.path.join(rest_dir, file_name)
                        if os.path.exists(target_path):
                            base, ext = os.path.splitext(file_name)
                            target_path = os.path.join(rest_dir, f"{base}_complex{ext}")

                        shutil.move(file_path, target_path)
                        moved_count += 1
                        print(f"   -> REST (Chaos): {file_name} (Clean Sum: {clean_edge_sum})")
                    else:
                        kept_count += 1
                        print(f"   -> BEHALTEN: {file_name} (Original: {edge_sum} -> Clean: {clean_edge_sum})")

    print(f"[rest.py] Fertig. {moved_count} verschoben. {kept_count} vor fälschlicher Verschiebung gerettet.")
