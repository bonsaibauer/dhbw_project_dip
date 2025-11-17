import cv2
import numpy as np


# --- Spot-/Farbprüfung Helfer ---

def create_masks(image, ero_kernel, ero_iterations):
    """Erzeugt Masken für Objekt und Analysebereich."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask_obj = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_analysis = cv2.erode(mask_obj, ero_kernel, iterations=ero_iterations)
    return gray, mask_obj, mask_analysis


def compute_blackhat(gray, kernel):
    """Hebt dunkle Flecken über Blackhat-Filter hervor."""
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


def detect_contrast(blackhat_img, contrast_threshold):
    """Segmentiert Defekte anhand eines Kontrastschwellwerts."""
    _, mask_defects = cv2.threshold(blackhat_img, contrast_threshold, 255, cv2.THRESH_BINARY)
    return mask_defects


def filter_defects(mask_defects, mask_analysis, noise_kernel):
    """Begrenzt Defekte auf den Snack und filtert Kleinstrauschen."""
    valid = cv2.bitwise_and(mask_defects, mask_defects, mask=mask_analysis)
    return cv2.morphologyEx(valid, cv2.MORPH_OPEN, noise_kernel)


def texture_features(gray, mask_analysis, dark_percentile):
    """Berechnet Texturstreuung, Median und Dark-Delta."""
    object_pixels = gray[mask_analysis == 255]
    if object_pixels.size == 0:
        return 0.0, 0.0, 0.0
    texture_std = float(np.std(object_pixels))
    median_intensity = float(np.median(object_pixels))
    dark_percentile_val = float(np.percentile(object_pixels, dark_percentile))
    dark_delta = median_intensity - dark_percentile_val
    return texture_std, median_intensity, dark_delta


def color_features(image, mask_analysis):
    """Berechnet LAB-Standardabweichung innerhalb der Objektmaske."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    masked_values = a_channel[mask_analysis == 255]
    if masked_values.size == 0:
        return 0.0
    return float(np.std(masked_values))


def erode_mask(mask, kernel, iterations):
    """Erzeugt eine enger gefasste Maske und liefert Fläche zurück."""
    if iterations <= 0:
        return mask, cv2.countNonZero(mask)
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    return eroded, cv2.countNonZero(eroded)


def spot_ratio(spot_area, object_area):
    """Hilfsfunktion für robuste Quotientenberechnung."""
    return spot_area / max(1, object_area)


def primary_defect(spot_area, object_area, inner_spot_area, ratio_limit, inner_ratio_limit, spot_threshold):
    """Prüft die Hauptbedingungen (Fläche + Verhältnis) für einen Defekt."""
    ratio = spot_ratio(spot_area, object_area)
    meets_ratio = (ratio >= ratio_limit) if ratio_limit > 0 else True
    inner_ratio = inner_spot_area / max(1, spot_area)
    meets_inner = (inner_ratio >= inner_ratio_limit) if spot_area > 0 else False
    return spot_area > spot_threshold and meets_ratio and meets_inner


def refine_defects(mask_obj, mask_defects, noise_kernel, ero_kernel, fine_iterations, inner_iterations, inner_ratio_limit, fine_ratio, spot_final_threshold):
    """Führt die feinere Erosionsvariante aus, um kleinere Defekte zu erkennen."""
    if fine_iterations <= 0:
        return False, 0

    mask_fine = cv2.erode(mask_obj, ero_kernel, iterations=fine_iterations)
    fine_area_obj = cv2.countNonZero(mask_fine)
    valid_fine = cv2.bitwise_and(mask_defects, mask_defects, mask=mask_fine)
    valid_fine = cv2.morphologyEx(valid_fine, cv2.MORPH_OPEN, noise_kernel)
    fine_spot_area = cv2.countNonZero(valid_fine)

    fine_ratio_val = spot_ratio(fine_spot_area, fine_area_obj)
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


def debug_view(blackhat_img, mask_analysis):
    """Erzeugt das Debug-Bild mit maskiertem Blackhat-Result."""
    return cv2.bitwise_and(blackhat_img, blackhat_img, mask=mask_analysis)


def detect_spots(image, settings, debug=False):
    """Führt die Farb- und Texturprüfung mit den übergebenen Parametern aus."""
    ero_kernel = np.ones(settings["ERO_KN"], np.uint8)
    noise_kernel = np.ones(settings["NOI_KN"], np.uint8)

    gray, mask_obj, mask_analysis = create_masks(
        image,
        ero_kernel,
        settings["ERO_ITER"],
    )
    object_area = cv2.countNonZero(mask_analysis)

    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, settings["BKH_KN"])
    blackhat_img = compute_blackhat(gray, blackhat_kernel)
    mask_defects = detect_contrast(blackhat_img, settings["BKH_CON"])
    valid_defects = filter_defects(mask_defects, mask_analysis, noise_kernel)

    spot_area = cv2.countNonZero(valid_defects)

    texture_std, median_intensity, dark_delta = texture_features(
        gray,
        mask_analysis,
        settings["DRK_PCT"],
    )
    lab_std = color_features(image, mask_analysis)

    inner_mask, _ = erode_mask(mask_analysis, ero_kernel, settings["INER_ITR"])
    inner_valid = cv2.bitwise_and(valid_defects, valid_defects, mask=inner_mask)
    inner_spot_area = cv2.countNonZero(inner_valid)

    is_defective = primary_defect(
        spot_area,
        object_area,
        inner_spot_area,
        settings["SPT_RAT"],
        settings["INSP_RAT"],
        settings["SPT_MIN"],
    )

    if not is_defective:
        fine_passed, fine_area = refine_defects(
            mask_obj,
            mask_defects,
            noise_kernel,
            ero_kernel,
            settings["FERO_ITR"],
            settings["INER_ITR"],
            settings["INSP_RAT"],
            settings["FSPT_RAT"],
            settings["SPT_FIN"],
        )
        if fine_passed:
            is_defective = True
            spot_area = fine_area

    debug_image = None
    if debug:
        debug_image = debug_view(blackhat_img, mask_analysis)

    result = {
        "is_defective": is_defective,
        "spot_area": spot_area,
        "texture_std": texture_std,
        "lab_std": lab_std,
        "dark_delta": dark_delta,
        "median_intensity": median_intensity,
    }
    if debug:
        result["debug_image"] = debug_image
    return result
