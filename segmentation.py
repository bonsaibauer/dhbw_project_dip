import os
import shutil
import stat

import cv2
import numpy as np

from main import get_paths, get_preprocessing_params

PATHS = get_paths()
RAW_DIR = PATHS["RAW_DIR"]
PROC_DIR = PATHS["PROC_DIR"]
PREPROCESSING_PARAMS = get_preprocessing_params()


def progress_bar(prefix, current, total, bar_len=30):
    if total <= 0:
        return
    ratio = min(max(current / total, 0), 1)
    filled = int(bar_len * ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    label = prefix.ljust(20)
    print(f"\r{label}[{bar}] {ratio * 100:5.1f}% ({current}/{total})", end="", flush=True)


# --- Dateiverwaltung ---------------------------------------------------------

def clean_directory(folder):
    if not os.path.exists(folder):
        return

    def _on_rm_error(func, path, exc_info):
        if isinstance(exc_info[1], PermissionError):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        else:
            raise

    shutil.rmtree(folder, onerror=_on_rm_error)


def image_paths(source_dir):
    valid_suffixes = (".jpg", ".jpeg", ".png")
    for root, _, files in os.walk(source_dir):
        for name in files:
            if name.lower().endswith(valid_suffixes):
                yield root, name


# --- Bildsegmentierung -------------------------------------------------------

def warp_segments(image, settings):
    """Gibt alle normalisierten Snacks eines Fotos zurück."""
    mask = cv2.bitwise_not(
        cv2.inRange(
            cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
            settings["HSV_LO"],
            settings["HSV_HI"],
        )
    )
    masked = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(
        cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1],
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )

    warp_w, warp_h = settings["WARP_SZ"]
    target_size = (settings["TGT_W"], settings["TGT_H"])
    dst_pts = np.float32([[0, warp_h - 1], [0, 0], [warp_w - 1, 0], [warp_w - 1, warp_h - 1]])

    outputs = []
    for contour in (cnt for cnt in contours if cv2.contourArea(cnt) > settings["CNT_MINA"]):
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

def process_directory(source_dir, target_dir, preprocess_settings):
    """Segmentiert alle Rohbilder und speichert die Ergebnisse im Zielordner."""
    clean_directory(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    image_files = list(image_paths(source_dir))
    total_files = len(image_files)
    if total_files == 0:
        print(f"Keine Bilder in '{source_dir}' gefunden.")
        return

    for idx, (root, name) in enumerate(image_files, 1):
        image = cv2.imread(os.path.join(root, name))
        if image is None:
            continue

        warped_snacks = warp_segments(image, preprocess_settings)
        if warped_snacks:
            class_dir = os.path.join(target_dir, os.path.basename(root))
            os.makedirs(class_dir, exist_ok=True)
            for snack in warped_snacks:
                cv2.imwrite(os.path.join(class_dir, name), snack)

        progress_bar("  Segmentierung", idx, total_files)

    if total_files > 0:
        print()


def main():
    if not os.path.exists(RAW_DIR):
        print(f"Fehler: Quellordner '{RAW_DIR}' nicht gefunden. Bitte Bilder bereitstellen.")
        return

    os.makedirs(os.path.dirname(PROC_DIR), exist_ok=True)
    process_directory(RAW_DIR, PROC_DIR, PREPROCESSING_PARAMS)


if __name__ == "__main__":
    main()
