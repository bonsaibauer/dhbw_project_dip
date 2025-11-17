import os
import shutil
import stat

import cv2
import numpy as np

from main import fetch_pipeline_paths, fetch_preprocessing_settings

PIPELINE_PATHS = fetch_pipeline_paths()
RAW_IMAGE_DIR = PIPELINE_PATHS["raw_image_directory"]
PROCESSED_IMAGE_DIR = PIPELINE_PATHS["processed_image_directory"]
PREPROCESSING_SETTINGS = fetch_preprocessing_settings()


def display_progress_bar(prefix, current, total, bar_len=30):
    if total <= 0:
        return
    ratio = min(max(current / total, 0), 1)
    filled = int(bar_len * ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    label = prefix.ljust(20)
    print(f"\r{label}[{bar}] {ratio * 100:5.1f}% ({current}/{total})", end="", flush=True)


# --- Dateiverwaltung ---------------------------------------------------------

def remove_directory_tree(folder):
    if not os.path.exists(folder):
        return

    def _on_rm_error(func, path, exc_info):
        if isinstance(exc_info[1], PermissionError):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        else:
            raise

    shutil.rmtree(folder, onerror=_on_rm_error)


def iterate_image_files(source_dir):
    valid_suffixes = (".jpg", ".jpeg", ".png")
    for root, _, files in os.walk(source_dir):
        for name in files:
            if name.lower().endswith(valid_suffixes):
                yield root, name


# --- Bildsegmentierung -------------------------------------------------------

def warp_image_segments(image, settings):
    """Gibt alle normalisierten Snacks eines Fotos zurück."""
    mask = cv2.bitwise_not(
        cv2.inRange(
            cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
            settings["preprocess_hsv_lower"],
            settings["preprocess_hsv_upper"],
        )
    )
    masked = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(
        cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1],
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )

    warp_w, warp_h = settings["warp_frame_size"]
    target_size = (settings["target_width"], settings["target_height"])
    dst_pts = np.float32([[0, warp_h - 1], [0, 0], [warp_w - 1, 0], [warp_w - 1, warp_h - 1]])

    outputs = []
    min_area = settings["minimum_contour_area"]
    for contour in (cnt for cnt in contours if cv2.contourArea(cnt) > min_area):
        rect = cv2.minAreaRect(contour)
        if rect[1][1] > rect[1][0]:
            rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)

        transform = cv2.getPerspectiveTransform(
            cv2.boxPoints(rect).astype("float32"),
            dst_pts,
        )

        mask_fill = np.zeros_like(gray)
        cv2.drawContours(mask_fill, [contour], -1, 255, cv2.FILLED)
        isolated = cv2.bitwise_and(masked, masked, mask=mask_fill)

        warped = cv2.warpPerspective(isolated, transform, (warp_w, warp_h), cv2.INTER_CUBIC)
        warped = cv2.resize(warped, target_size, interpolation=cv2.INTER_CUBIC)
        outputs.append(warped)

    return outputs


# --- Datensatzsteuerung ------------------------------------------------------

def segment_directory_images(source_dir, target_dir, preprocess_settings):
    """Segmentiert alle Rohbilder und speichert die Ergebnisse im Zielordner."""
    remove_directory_tree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    image_files = list(iterate_image_files(source_dir))
    total_files = len(image_files)
    if total_files == 0:
        print(f"Keine Bilder in '{source_dir}' gefunden.")
        return

    for idx, (root, name) in enumerate(image_files, 1):
        image = cv2.imread(os.path.join(root, name))
        if image is None:
            continue

        warped_snacks = warp_image_segments(image, preprocess_settings)
        if warped_snacks:
            class_dir = os.path.join(target_dir, os.path.basename(root))
            os.makedirs(class_dir, exist_ok=True)
            for snack in warped_snacks:
                cv2.imwrite(os.path.join(class_dir, name), snack)

        display_progress_bar("  Segmentierung", idx, total_files)

    if total_files > 0:
        print()


def run_segmentation_cli():
    if not os.path.exists(RAW_IMAGE_DIR):
        print(f"Fehler: Quellordner '{RAW_IMAGE_DIR}' nicht gefunden. Bitte Bilder bereitstellen.")
        return

    os.makedirs(os.path.dirname(PROCESSED_IMAGE_DIR), exist_ok=True)
    segment_directory_images(RAW_IMAGE_DIR, PROCESSED_IMAGE_DIR, PREPROCESSING_SETTINGS)


if __name__ == "__main__":
    run_segmentation_cli()
