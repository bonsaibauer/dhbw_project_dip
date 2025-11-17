import os
import shutil
import stat

import cv2
import numpy as np


def remove_green_background(image, hsv_lo, hsv_hi):
    """Maskiert alles außer dem nicht-grünen Objekt."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, hsv_lo, hsv_hi)
    mask_object = cv2.bitwise_not(mask_green)
    return cv2.bitwise_and(image, image, mask=mask_object)


def extract_significant_contours(masked_image, min_area):
    """Findet alle Konturen oberhalb des Mindestflächen-Thresholds."""
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]


def normalize_rect(rect):
    """Erzwingt Querformat für MinAreaRect, damit der Warp stabil bleibt."""
    if rect[1][1] > rect[1][0]:
        return (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)
    return rect


def build_warp_matrix(contour, warp_size):
    """Erstellt die Projektionsmatrix für den perspektivischen Warp."""
    rect = normalize_rect(cv2.minAreaRect(contour))
    box_points = cv2.boxPoints(rect).astype("float32")
    dst_pts = np.array(
        [
            [0, warp_size[1] - 1],
            [0, 0],
            [warp_size[0] - 1, 0],
            [warp_size[0] - 1, warp_size[1] - 1],
        ],
        dtype="float32",
    )
    return cv2.getPerspectiveTransform(box_points, dst_pts)


def isolate_contour_region(masked_image, contour):
    """Nullt alles außerhalb der Kontur und liefert das zugeschnittene Bild."""
    isolated = masked_image.copy()
    mask = np.zeros(masked_image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    isolated[mask == 0] = (0, 0, 0)
    return isolated


def warp_and_resize(image, contour, warp_size, target_width, target_height):
    """Projiziert die Konturfläche in ein normiertes Koordinatensystem."""
    isolated = isolate_contour_region(image, contour)
    matrix = build_warp_matrix(contour, warp_size)
    warped = cv2.warpPerspective(isolated, matrix, warp_size, cv2.INTER_CUBIC)
    return cv2.resize(warped, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def prep_img(image, result, settings):
    """Zieht den grünen Hintergrund ab und liefert einen normalisierten Warp."""
    masked = remove_green_background(image, settings["HSV_LO"], settings["HSV_HI"])
    contours = extract_significant_contours(masked, settings["CNT_MINA"])

    processed = False
    for contour in contours:
        warped = warp_and_resize(
            masked,
            contour,
            settings["WARP_SZ"],
            settings["TGT_W"],
            settings["TGT_H"],
        )
        result.append({"name": "Result", "data": warped})
        processed = True

    return processed


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
        def _on_rm_error(func, path, exc_info):
            error = exc_info[1]
            if isinstance(error, PermissionError):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            else:
                raise

        shutil.rmtree(target_dir, onerror=_on_rm_error)
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
