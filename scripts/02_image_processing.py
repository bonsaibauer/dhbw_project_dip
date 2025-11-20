import csv
import json
import os
from functools import lru_cache

import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_dir = os.path.join(project_root, "config")
path_path = os.path.join(config_dir, "path.json")
imgproc_path = os.path.join(config_dir, "image_processing.json")


def load_json(path, error_msg):
    if not os.path.exists(path):
        raise FileNotFoundError(error_msg)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_path_config():
    return load_json(
        path_path,
        f"Pfaddatei '{path_path}' nicht gefunden.",
    )


@lru_cache(maxsize=1)
def load_image_config():
    return load_json(
        imgproc_path,
        f"Image-Processing-Datei '{imgproc_path}' nicht gefunden.",
    )


def norm_path(path_value):
    return os.path.normpath(path_value) if path_value else ""


def read_path_section(cfg_name):
    return load_path_config().get(cfg_name, {})


def read_image_section(cfg_name):
    return load_image_config().get(cfg_name, {})


def pull_value(cfg, key, default, transform=None):
    value = cfg.get(key, default)
    return transform(value) if transform else value


def load_paths():
    return {
        key: norm_path(value)
        for key, value in read_path_section("paths").items()
    }


def load_geometry():
    cfg = read_image_section("geometry")
    return {
        "polygon_epsilon_factor": pull_value(
            cfg,
            "polygon_epsilon_factor",
            0.0,
        ),
        "minimum_hole_area": pull_value(cfg, "minimum_hole_area", 0),
        "minimum_window_area": pull_value(cfg, "minimum_window_area", 0),
        "maximum_center_area": pull_value(cfg, "maximum_center_area", 0),
    }


def load_spot():
    cfg = read_image_section("spot")
    return {
        "erosion_kernel_size": tuple(
            pull_value(cfg, "erosion_kernel_size", [0, 0])
        ),
        "erosion_iterations": pull_value(cfg, "erosion_iterations", 0),
        "blackhat_kernel_size": tuple(
            pull_value(cfg, "blackhat_kernel_size", [0, 0])
        ),
        "blackhat_contrast_threshold": pull_value(
            cfg,
            "blackhat_contrast_threshold",
            0,
        ),
        "noise_kernel_size": tuple(
            pull_value(cfg, "noise_kernel_size", [0, 0])
        ),
        "minimum_spot_area": pull_value(cfg, "minimum_spot_area", 0),
        "dark_percentile": pull_value(cfg, "dark_percentile", 0),
    }


def normalize_path(path):
    """Normalisiert relative Pfade analog zu den Annotationen."""
    if not path:
        return ""
    normalized = path.replace("\\", "/")
    marker = "Data/Images/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    return normalized.lstrip("/")

path_map = load_paths()
proc_dir = path_map["processed_image_directory"]
pipe_csv = path_map["pipeline_csv_path"]
geo_cfg = load_geometry()
spot_cfg = load_spot()

def show_progress(prefix, current, total, bar_len=30):
    if total <= 0:
        return
    ratio = min(max(current / total, 0), 1)
    filled = int(bar_len * ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    label = prefix.ljust(20)
    print(f"\r{label}[{bar}] {ratio * 100:5.1f}% ({current}/{total})", end="", flush=True)


BOOL_TRUE = "true"
BOOL_FALSE = "false"


def bool_text(value):
    return BOOL_TRUE if value else BOOL_FALSE


def build_masks(image, ero_kernel, ero_iterations):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask_obj = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_analysis = cv2.erode(mask_obj, ero_kernel, iterations=ero_iterations)
    return gray, mask_analysis


def detect_spots(image, settings):
    """Reduzierte Farbanalyse: Blackhat + einfacher Schwellwert."""
    ero_kernel = np.ones(settings["erosion_kernel_size"], np.uint8)
    noise_kernel = np.ones(settings["noise_kernel_size"], np.uint8)
    gray, mask_analysis = build_masks(
        image,
        ero_kernel,
        settings["erosion_iterations"],
    )
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        settings["blackhat_kernel_size"],
    )
    blackhat_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
    _, mask_defects = cv2.threshold(
        blackhat_img,
        settings["blackhat_contrast_threshold"],
        255,
        cv2.THRESH_BINARY,
    )
    valid_defects = cv2.bitwise_and(mask_defects, mask_defects, mask=mask_analysis)
    valid_defects = cv2.morphologyEx(valid_defects, cv2.MORPH_OPEN, noise_kernel)

    spot_area = cv2.countNonZero(valid_defects)
    obj_pixels = gray[mask_analysis == 255]
    if obj_pixels.size == 0:
        texture_std = 0.0
        dark_delta = 0.0
    else:
        texture_std = float(np.std(obj_pixels))
        median_intensity = float(np.median(obj_pixels))
        dark_val = float(np.percentile(obj_pixels, settings["dark_percentile"]))
        dark_delta = median_intensity - dark_val

    min_spot = max(0, int(settings["minimum_spot_area"]))
    is_defective = spot_area >= max(min_spot, 120)

    return {
        "color_detection_flag": is_defective,
        "color_spot_area": spot_area,
        "color_texture_stddev": texture_std,
        "color_dark_delta": dark_delta,
    }


def symmetry_score(image):
    """Berechnet einen 6-fachen Rotationssymmetrie-Score (0-100%)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    total_pixels = cv2.countNonZero(mask)
    if total_pixels == 0:
        return 0.0

    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return 0.0

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    height, width = mask.shape[:2]
    core_mask = mask.copy()

    for angle in range(60, 360, 60):
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(mask, rot_mat, (width, height))
        core_mask = cv2.bitwise_and(core_mask, rotated)

    asymmetric = cv2.subtract(mask, core_mask)
    asymmetric_pixels = cv2.countNonZero(asymmetric)
    symmetry_ratio = 1.0 - (asymmetric_pixels / total_pixels)
    return max(0.0, min(100.0, round(symmetry_ratio * 100, 2)))


def extract_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def geometry_stats(contours, hierarchy, geo_cfg):
    """Zählt Fenster, Fragmente und liefert Geometriekennzahlen."""
    eps_fact = geo_cfg["polygon_epsilon_factor"]
    hole_min = geo_cfg["minimum_hole_area"]
    wind_min = geo_cfg["minimum_window_area"]
    ctr_maxa = geo_cfg["maximum_center_area"]

    stats = {
        "geometry_has_primary_object": False,
        "geometry_window_count": 0,
        "geometry_has_center_hole": False,
        "geometry_window_area_list": [],
    }

    if not contours:
        return stats

    main_cnt_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    main_cnt = contours[main_cnt_idx]
    stats["geometry_has_primary_object"] = True

    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            parent_idx = hierarchy[0][i][3]

            if parent_idx == main_cnt_idx:
                hole_area = cv2.contourArea(cnt)
                if hole_area < hole_min:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                approx_hole = cv2.approxPolyDP(cnt, eps_fact * perimeter, True)
                corners = len(approx_hole)

                if 3 <= corners <= 5 and hole_area > wind_min:
                    stats["geometry_window_count"] += 1
                    stats["geometry_window_area_list"].append(hole_area)
                elif corners > 5 and hole_area < ctr_maxa:
                    stats["geometry_has_center_hole"] = True

    if (
        not stats["geometry_has_center_hole"]
        and len(stats["geometry_window_area_list"]) >= 6
    ):
        min_idx = int(np.argmin(stats["geometry_window_area_list"]))
        min_area = stats["geometry_window_area_list"][min_idx]
        if min_area <= ctr_maxa:
            stats["geometry_has_center_hole"] = True
            stats["geometry_window_area_list"].pop(min_idx)
            stats["geometry_window_count"] = len(stats["geometry_window_area_list"])

    return stats


CSV_FIELDS = [
    "relative_path",
    "source_path",
    "filename",
    "geometry_window_count",
    "geometry_has_center_hole",
    "geometry_window_area_list",
    "color_detection_flag",
    "color_spot_area",
    "color_dark_delta",
]


def analyze_image(img_path, source_root, geo_cfg, spot_cfg):
    image = cv2.imread(img_path)
    if image is None:
        return None

    rel_path = normalize_path(os.path.relpath(img_path, source_root))

    contours, hierarchy = extract_contours(image)
    geo = geometry_stats(contours, hierarchy, geo_cfg)

    color_res = detect_spots(image, spot_cfg)

    return {
        "relative_path": rel_path,
        "source_path": os.path.abspath(img_path),
        "filename": os.path.basename(img_path),
        "geometry_window_count": f"{geo.get('geometry_window_count', 0)}",
        "geometry_has_center_hole": bool_text(
            geo.get("geometry_has_center_hole", False)
        ),
        "geometry_window_area_list": json.dumps(
            geo.get("geometry_window_area_list", [])
        ),
        "color_detection_flag": bool_text(color_res["color_detection_flag"]),
        "color_spot_area": f"{color_res.get('color_spot_area', 0)}",
        "color_dark_delta": f"{color_res.get('color_dark_delta', 0.0)}",
    }


def process_folder(source_dir, csv_path, geo_cfg, spot_cfg):
    if not os.path.exists(source_dir):
        print(f"Fehler: Verzeichnis '{source_dir}' nicht gefunden.")
        return

    image_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(source_dir)
        for file in files
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    total_files = len(image_files)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if total_files == 0:
        with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
        print("Keine Bilder für die Bildverarbeitung gefunden.")
        return

    rows = []
    for idx, img_path in enumerate(sorted(image_files), 1):
        data = analyze_image(img_path, source_dir, geo_cfg, spot_cfg)
        if data:
            rows.append(data)
        show_progress("  Bildverarbeitung", idx, total_files)

    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if total_files > 0:
        print()


def process_cli():
    process_folder(
        proc_dir,
        pipe_csv,
        geo_cfg,
        spot_cfg,
    )


if __name__ == "__main__":
    process_cli()
