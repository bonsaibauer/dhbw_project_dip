import os
import shutil

import cv2

from image_processing import prep_img


def prog_bar(prefix, current, total, bar_length=30):
    """Einfacher Fortschrittsbalken für die Segmentierung."""
    if total <= 0:
        return
    ratio = min(max(current / total, 0), 1)
    filled = int(bar_length * ratio)
    bar = "#" * filled + "-" * (bar_length - filled)
    percent = ratio * 100
    print(f"\r{prefix} [{bar}] {percent:5.1f}% ({current}/{total})", end="", flush=True)


def img_list(source_dir):
    """Listet rekursiv alle Bilddateien eines Quellordners auf."""
    image_files = []
    for root, _, files in os.walk(source_dir):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append((root, name))
    return image_files


def prep_set(source_dir, target_dir, preprocess_settings):
    """Segmentiert alle Rohbilder und speichert die Ergebnisse im Zielordner."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    image_files = img_list(source_dir)
    total_files = len(image_files)
    if total_files == 0:
        print(f"Keine Bilder in '{source_dir}' gefunden.")
        return

    print(f"Segmentierung ({total_files} Bilder):")
    class_dirs = {}

    for idx, (root, name) in enumerate(image_files, 1):
        full_path = os.path.join(root, name)
        image = cv2.imread(full_path)
        if image is None:
            continue

        results = []
        has_result = prep_img(image, results, preprocess_settings)
        if has_result:
            class_name = os.path.basename(root)
            if class_name not in class_dirs:
                save_path = os.path.join(target_dir, class_name)
                os.makedirs(save_path, exist_ok=True)
                class_dirs[class_name] = save_path
            save_path = class_dirs[class_name]

            for item in results:
                if item["name"] == "Result":
                    cv2.imwrite(os.path.join(save_path, name), item["data"])

        prog_bar("  Segmentierung", idx, total_files)

    print("\nSegmentierung abgeschlossen.")


def cnt_hier(image):
    """Liefert alle Konturen samt Hierarchie des segmentierten Bildes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
