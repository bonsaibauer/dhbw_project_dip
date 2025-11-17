import csv
import json
import os

import cv2
import numpy as np

from main import (
    fetch_geometry_settings,
    fetch_pipeline_paths,
    fetch_spot_detection_settings,
)
from validation import normalize_annotation_path

PIPELINE_PATHS = fetch_pipeline_paths()
PROCESSED_IMAGE_DIR = PIPELINE_PATHS["processed_image_directory"]
PIPELINE_CSV_PATH = PIPELINE_PATHS["pipeline_csv_path"]
GEOMETRY_SETTINGS = fetch_geometry_settings()
SPOT_SETTINGS = fetch_spot_detection_settings()


def display_progress_bar(prefix, current, total, bar_len=30):
    if total <= 0:
        return
    ratio = min(max(current / total, 0), 1)
    filled = int(bar_len * ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    label = prefix.ljust(20)
    print(f"\r{label}[{bar}] {ratio * 100:5.1f}% ({current}/{total})", end="", flush=True)


def convert_bool_to_text(value):
    return "true" if value else "false"


# --- Spot-/Farbprüfung Helfer ---

def create_object_masks(image, ero_kernel, ero_iterations):
    """Erzeugt Masken für Objekt und Analysebereich."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask_obj = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_analysis = cv2.erode(mask_obj, ero_kernel, iterations=ero_iterations)
    return gray, mask_obj, mask_analysis


def compute_blackhat_filter(gray, kernel):
    """Hebt dunkle Flecken über Blackhat-Filter hervor."""
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


def detect_defect_contrast(blackhat_img, contrast_threshold):
    """Segmentiert Defekte anhand eines Kontrastschwellwerts."""
    _, mask_defects = cv2.threshold(blackhat_img, contrast_threshold, 255, cv2.THRESH_BINARY)
    return mask_defects


def filter_defect_regions(mask_defects, mask_analysis, noise_kernel):
    """Begrenzt Defekte auf den Snack und filtert Kleinstrauschen."""
    valid = cv2.bitwise_and(mask_defects, mask_defects, mask=mask_analysis)
    return cv2.morphologyEx(valid, cv2.MORPH_OPEN, noise_kernel)


def calculate_texture_features(gray, mask_analysis, dark_percentile):
    """Berechnet Texturstreuung, Median und Dark-Delta."""
    object_pixels = gray[mask_analysis == 255]
    if object_pixels.size == 0:
        return 0.0, 0.0, 0.0
    texture_std = float(np.std(object_pixels))
    median_intensity = float(np.median(object_pixels))
    dark_percentile_val = float(np.percentile(object_pixels, dark_percentile))
    dark_delta = median_intensity - dark_percentile_val
    return texture_std, median_intensity, dark_delta


def calculate_lab_color_variance(image, mask_analysis):
    """Berechnet LAB-Standardabweichung innerhalb der Objektmaske."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    masked_values = a_channel[mask_analysis == 255]
    if masked_values.size == 0:
        return 0.0
    return float(np.std(masked_values))


def erode_mask_and_area(mask, kernel, iterations):
    """Erzeugt eine enger gefasste Maske und liefert Fläche zurück."""
    if iterations <= 0:
        return mask, cv2.countNonZero(mask)
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    return eroded, cv2.countNonZero(eroded)


def calculate_spot_ratio(spot_area, object_area):
    """Hilfsfunktion für robuste Quotientenberechnung."""
    return spot_area / max(1, object_area)


def meets_primary_defect_criteria(spot_area, object_area, inner_spot_area, ratio_limit, inner_ratio_limit, spot_threshold):
    """Prüft die Hauptbedingungen (Fläche + Verhältnis) für einen Defekt."""
    ratio = calculate_spot_ratio(spot_area, object_area)
    meets_ratio = (ratio >= ratio_limit) if ratio_limit > 0 else True
    inner_ratio = inner_spot_area / max(1, spot_area)
    meets_inner = (inner_ratio >= inner_ratio_limit) if spot_area > 0 else False
    return spot_area > spot_threshold and meets_ratio and meets_inner


def refine_defect_candidates(mask_obj, mask_defects, noise_kernel, ero_kernel, fine_iterations, inner_iterations, inner_ratio_limit, fine_ratio, spot_final_threshold):
    """Führt die feinere Erosionsvariante aus, um kleinere Defekte zu erkennen."""
    if fine_iterations <= 0:
        return False, 0

    mask_fine = cv2.erode(mask_obj, ero_kernel, iterations=fine_iterations)
    fine_area_obj = cv2.countNonZero(mask_fine)
    valid_fine = cv2.bitwise_and(mask_defects, mask_defects, mask=mask_fine)
    valid_fine = cv2.morphologyEx(valid_fine, cv2.MORPH_OPEN, noise_kernel)
    fine_spot_area = cv2.countNonZero(valid_fine)

    fine_ratio_val = calculate_spot_ratio(fine_spot_area, fine_area_obj)
    meets_fine_ratio = (fine_ratio_val >= fine_ratio) if fine_ratio > 0 else True

    if inner_iterations > 0:
        fine_inner_mask = cv2.erode(mask_fine, ero_kernel, iterations=inner_iterations)
        fine_inner_valid = cv2.bitwise_and(valid_fine, valid_fine, mask=fine_inner_mask)
        fine_inner_area = cv2.countNonZero(fine_inner_valid)
    else:
        fine_inner_area = fine_spot_area

    fine_inner_ratio = fine_inner_area / max(1, fine_spot_area) if fine_spot_area > 0 else 0
    meets_fine_inner = (fine_inner_ratio >= inner_ratio_limit) if fine_spot_area > 0 else False

    passes = (
        fine_spot_area > spot_final_threshold
        and meets_fine_ratio
        and meets_fine_inner
    )
    return passes, fine_spot_area


def render_debug_view(blackhat_img, mask_analysis):
    """Erzeugt das Debug-Bild mit maskiertem Blackhat-Result."""
    return cv2.bitwise_and(blackhat_img, blackhat_img, mask=mask_analysis)


def detect_surface_spots(image, settings, debug=False):
    """Führt die Farb- und Texturprüfung mit den übergebenen Parametern aus."""
    ero_kernel = np.ones(settings["erosion_kernel_size"], np.uint8)
    noise_kernel = np.ones(settings["noise_kernel_size"], np.uint8)

    gray, mask_obj, mask_analysis = create_object_masks(
        image,
        ero_kernel,
        settings["erosion_iterations"],
    )
    object_area = cv2.countNonZero(mask_analysis)

    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        settings["blackhat_kernel_size"],
    )
    blackhat_img = compute_blackhat_filter(gray, blackhat_kernel)
    mask_defects = detect_defect_contrast(
        blackhat_img,
        settings["blackhat_contrast_threshold"],
    )
    valid_defects = filter_defect_regions(mask_defects, mask_analysis, noise_kernel)

    spot_area = cv2.countNonZero(valid_defects)

    texture_std, median_intensity, dark_delta = calculate_texture_features(
        gray,
        mask_analysis,
        settings["dark_percentile"],
    )
    lab_std = calculate_lab_color_variance(image, mask_analysis)

    inner_mask, _ = erode_mask_and_area(
        mask_analysis,
        ero_kernel,
        settings["inner_erosion_iterations"],
    )
    inner_valid = cv2.bitwise_and(valid_defects, valid_defects, mask=inner_mask)
    inner_spot_area = cv2.countNonZero(inner_valid)

    is_defective = meets_primary_defect_criteria(
        spot_area,
        object_area,
        inner_spot_area,
        settings["spot_area_ratio"],
        settings["inner_spot_ratio"],
        settings["minimum_spot_area"],
    )

    if not is_defective:
        fine_passed, fine_area = refine_defect_candidates(
            mask_obj,
            mask_defects,
            noise_kernel,
            ero_kernel,
            settings["fine_erosion_iterations"],
            settings["inner_erosion_iterations"],
            settings["inner_spot_ratio"],
            settings["fine_spot_ratio"],
            settings["fine_spot_area"],
        )
        if fine_passed:
            is_defective = True
            spot_area = fine_area

    debug_image = None
    if debug:
        debug_image = render_debug_view(blackhat_img, mask_analysis)

    result = {
        "color_detection_flag": is_defective,
        "color_spot_area": spot_area,
        "color_texture_stddev": texture_std,
        "color_lab_stddev": lab_std,
        "color_dark_delta": dark_delta,
        "color_median_intensity": median_intensity,
    }
    if debug:
        result["debug_image"] = debug_image
    return result


def extract_contour_hierarchy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def compute_geometry_features(contours, hierarchy, geometry_settings):
    """Zählt Fenster, Fragmente und liefert Geometriekennzahlen."""
    eps_fact = geometry_settings["polygon_epsilon_factor"]
    hole_min = geometry_settings["minimum_hole_area"]
    wind_min = geometry_settings["minimum_window_area"]
    ctr_maxa = geometry_settings["maximum_center_area"]
    frag_min = geometry_settings["minimum_fragment_area"]

    stats = {
        "geometry_has_primary_object": False,
        "geometry_area": 0,
        "geometry_solidity": 0,
        "geometry_window_count": 0,
        "geometry_has_center_hole": False,
        "main_contour": None,
        "geometry_fragment_count": 0,
        "geometry_convex_area": 0,
        "geometry_window_area_list": [],
        "geometry_outer_contour_count": 0,
        "geometry_edge_damage_ratio": 0.0,
        "geometry_edge_segment_count": 0,
    }

    if not contours:
        return stats

    main_cnt_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    main_cnt = contours[main_cnt_idx]
    area = cv2.contourArea(main_cnt)

    stats["geometry_has_primary_object"] = True
    stats["geometry_area"] = area
    stats["main_contour"] = main_cnt
    hull = cv2.convexHull(main_cnt)
    stats["geometry_convex_area"] = cv2.contourArea(hull)
    hull_peri = cv2.arcLength(hull, True)
    main_peri = cv2.arcLength(main_cnt, True)
    stats["geometry_edge_damage_ratio"] = (
        (hull_peri / max(1.0, main_peri)) if main_peri > 0 else 0.0
    )
    approx = cv2.approxPolyDP(main_cnt, eps_fact * main_peri, True)
    stats["geometry_edge_segment_count"] = len(approx)

    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            parent_idx = hierarchy[0][i][3]
            cnt_area = cv2.contourArea(cnt)

            if parent_idx == -1:
                stats["geometry_outer_contour_count"] += 1
                if i != main_cnt_idx and cnt_area > frag_min:
                    stats["geometry_fragment_count"] += 1
                    continue

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
    "pipeline_has_anomaly_flag",
    "geometry_has_primary_object",
    "geometry_area",
    "geometry_convex_area",
    "geometry_edge_damage_ratio",
    "geometry_edge_segment_count",
    "geometry_window_count",
    "geometry_has_center_hole",
    "geometry_fragment_count",
    "geometry_outer_contour_count",
    "geometry_window_area_list",
    "color_detection_flag",
    "color_spot_area",
    "color_texture_stddev",
    "color_lab_stddev",
    "color_dark_delta",
    "color_median_intensity",
]


def analyze_image_features(img_path, source_root, geometry_settings, spot_settings):
    image = cv2.imread(img_path)
    if image is None:
        return None

    rel_path = normalize_annotation_path(os.path.relpath(img_path, source_root))
    is_anomaly = "Anomaly" in img_path

    contours, hierarchy = extract_contour_hierarchy(image)
    geo = compute_geometry_features(contours, hierarchy, geometry_settings)

    if is_anomaly:
        color_res = detect_surface_spots(image, spot_settings)
    else:
        color_res = {
            "color_detection_flag": False,
            "color_spot_area": 0,
            "color_texture_stddev": 0.0,
            "color_lab_stddev": 0.0,
            "color_dark_delta": 0.0,
            "color_median_intensity": 0.0,
        }

    return {
        "relative_path": rel_path,
        "source_path": os.path.abspath(img_path),
        "filename": os.path.basename(img_path),
        "pipeline_has_anomaly_flag": convert_bool_to_text(is_anomaly),
        "geometry_has_primary_object": convert_bool_to_text(
            geo["geometry_has_primary_object"]
        ),
        "geometry_area": f"{geo.get('geometry_area', 0)}",
        "geometry_convex_area": f"{geo.get('geometry_convex_area', 0)}",
        "geometry_edge_damage_ratio": f"{geo.get('geometry_edge_damage_ratio', 0.0)}",
        "geometry_edge_segment_count": f"{geo.get('geometry_edge_segment_count', 0)}",
        "geometry_window_count": f"{geo.get('geometry_window_count', 0)}",
        "geometry_has_center_hole": convert_bool_to_text(
            geo.get("geometry_has_center_hole", False)
        ),
        "geometry_fragment_count": f"{geo.get('geometry_fragment_count', 0)}",
        "geometry_outer_contour_count": f"{geo.get('geometry_outer_contour_count', 0)}",
        "geometry_window_area_list": json.dumps(
            geo.get("geometry_window_area_list", [])
        ),
        "color_detection_flag": convert_bool_to_text(color_res["color_detection_flag"]),
        "color_spot_area": f"{color_res.get('color_spot_area', 0)}",
        "color_texture_stddev": f"{color_res.get('color_texture_stddev', 0.0)}",
        "color_lab_stddev": f"{color_res.get('color_lab_stddev', 0.0)}",
        "color_dark_delta": f"{color_res.get('color_dark_delta', 0.0)}",
        "color_median_intensity": f"{color_res.get('color_median_intensity', 0.0)}",
    }


def process_directory_to_csv(source_dir, csv_path, geometry_settings, spot_settings):
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
        data = analyze_image_features(img_path, source_dir, geometry_settings, spot_settings)
        if data:
            rows.append(data)
        display_progress_bar("  Bildverarbeitung", idx, total_files)

    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if total_files > 0:
        print()


def run_image_processing_cli():
    process_directory_to_csv(
        PROCESSED_IMAGE_DIR,
        PIPELINE_CSV_PATH,
        GEOMETRY_SETTINGS,
        SPOT_SETTINGS,
    )


if __name__ == "__main__":
    run_image_processing_cli()
