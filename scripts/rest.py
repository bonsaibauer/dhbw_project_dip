import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

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


def create_edge_report(image_data, output_file="complexity_report.png"):
    num_images = min(5, len(image_data))
    if num_images == 0:
        return

    print(f"[rest.py] Erstelle Bericht f�r {num_images} Bilder...")

    fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    if num_images == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for idx, (img_path, edges_orig, edges_clean) in enumerate(image_data[:num_images]):
        axes[0, idx].imshow(edges_orig, cmap='gray')
        axes[0, idx].set_title(f"Original\n{os.path.basename(img_path)}", fontsize=8)
        axes[0, idx].axis('off')

        axes[1, idx].imshow(edges_clean, cmap='gray')
        axes[1, idx].set_title("Bereinigt (Ohne Artefakte)", fontsize=8)
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=100)
    plt.close()
    print(f"[rest.py] Bericht gespeichert: {output_file}")


def run_complexity_check(sorted_dir):
    print("\n[rest.py] Starte intelligente Komplexit�ts-Pr�fung...")
    print(f"   - Limit: {MAX_EDGE_SUM} Kanten-Pixel")
    print(f"   - Artefakt-Filter: Objekte unter {MIN_OBJECT_AREA}px werden ignoriert")

    rest_dir = os.path.join(sorted_dir, "Rest")
    os.makedirs(rest_dir, exist_ok=True)

    check_classes = ["Normal", "Bruch"]
    moved_count = 0
    kept_count = 0

    for cls in check_classes:
        class_path = os.path.join(sorted_dir, cls)
        if not os.path.exists(class_path):
            continue

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

                elif edge_sum > MAX_EDGE_SUM:
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

    print(f"[rest.py] Fertig. {moved_count} verschoben. {kept_count} vor f�lschlicher Verschiebung gerettet.")
