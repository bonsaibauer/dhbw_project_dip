import cv2
import numpy as np


def prep_img(image, result, settings):
    """Zieht den grünen Hintergrund ab und liefert einen normalisierten Warp."""
    hsv_lo = settings["HSV_LO"]
    hsv_hi = settings["HSV_HI"]
    min_area = settings["CNT_MINA"]
    warp_size = settings["WARP_SZ"]
    target_width = settings["TGT_W"]
    target_height = settings["TGT_H"]

    image_copy = image.copy()
    image_work = image.copy()

    hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, hsv_lo, hsv_hi)
    mask_object = cv2.bitwise_not(mask_green)
    image_work = cv2.bitwise_and(image_work, image_work, mask=mask_object)

    _, thresh = cv2.threshold(
        cv2.cvtColor(image_work, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    processed = False
    for contour in contours:
        if cv2.contourArea(contour) <= min_area:
            continue

        rect = cv2.minAreaRect(contour)
        if rect[1][1] > rect[1][0]:
            rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)

        boxf = cv2.boxPoints(rect)
        boxf = np.int64(boxf)

        mask = np.zeros((image_copy.shape[0], image_copy.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        image_work[mask == 0] = (0, 0, 0)

        dst_pts = np.array(
            [
                [0, warp_size[1] - 1],
                [0, 0],
                [warp_size[0] - 1, 0],
                [warp_size[0] - 1, warp_size[1] - 1],
            ],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(boxf.astype("float32"), dst_pts)

        warped = cv2.warpPerspective(image_work, matrix, warp_size, cv2.INTER_CUBIC)
        warped = cv2.resize(
            warped, (target_width, target_height), interpolation=cv2.INTER_CUBIC
        )

        result.append({"name": "Result", "data": warped})
        processed = True

    return processed


def spot_det(image, settings, debug=False):
    """Führt die Farb- und Texturprüfung mit den übergebenen Parametern aus."""
    ero_kernel = np.ones(settings["ERO_KN"], np.uint8)
    ero_iterations = settings["ERO_ITER"]
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, settings["BKH_KN"]
    )
    blackhat_contrast = settings["BKH_CON"]
    noise_kernel = np.ones(settings["NOI_KN"], np.uint8)
    inner_iterations = settings["INER_ITR"]
    ratio_limit = settings["SPT_RAT"]
    spot_threshold = settings["SPT_MIN"]
    fine_iterations = settings["FERO_ITR"]
    fine_ratio = settings["FSPT_RAT"]
    spot_final_threshold = settings["SPT_FIN"]
    inner_ratio_limit = settings["INSP_RAT"]
    dark_percentile = settings["DRK_PCT"]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask_obj = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    mask_analysis = cv2.erode(mask_obj, ero_kernel, iterations=ero_iterations)
    object_area = cv2.countNonZero(mask_analysis)

    blackhat_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
    _, mask_defects = cv2.threshold(blackhat_img, blackhat_contrast, 255, cv2.THRESH_BINARY)
    valid_defects = cv2.bitwise_and(mask_defects, mask_defects, mask=mask_analysis)
    valid_defects = cv2.morphologyEx(valid_defects, cv2.MORPH_OPEN, noise_kernel)

    spot_area = cv2.countNonZero(valid_defects)
    object_pixels = gray[mask_analysis == 255]
    texture_std = float(np.std(object_pixels)) if object_area else 0.0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    a_std = float(np.std(a_channel[mask_analysis == 255])) if object_area else 0.0
    dark_delta = 0.0
    median_intensity = 0.0
    if object_pixels.size > 0:
        median_intensity = float(np.median(object_pixels))
        dark_percentile_val = float(np.percentile(object_pixels, dark_percentile))
        dark_delta = median_intensity - dark_percentile_val

    inner_mask = mask_analysis
    inner_spot_area = spot_area
    if inner_iterations > 0:
        inner_mask = cv2.erode(mask_analysis, ero_kernel, iterations=inner_iterations)
        inner_valid = cv2.bitwise_and(valid_defects, valid_defects, mask=inner_mask)
        inner_spot_area = cv2.countNonZero(inner_valid)

    ratio = spot_area / max(1, object_area)
    meets_ratio = (ratio >= ratio_limit) if ratio_limit > 0 else True
    inner_ratio = inner_spot_area / max(1, spot_area)
    meets_inner = (inner_ratio >= inner_ratio_limit) if spot_area > 0 else False
    is_defective = spot_area > spot_threshold and meets_ratio and meets_inner

    if not is_defective and fine_iterations > 0:
        mask_fine = cv2.erode(mask_obj, ero_kernel, iterations=fine_iterations)
        fine_area_obj = cv2.countNonZero(mask_fine)
        valid_fine = cv2.bitwise_and(mask_defects, mask_defects, mask=mask_fine)
        valid_fine = cv2.morphologyEx(valid_fine, cv2.MORPH_OPEN, noise_kernel)
        fine_spot_area = cv2.countNonZero(valid_fine)
        fine_ratio_val = fine_spot_area / max(1, fine_area_obj)
        meets_fine_ratio = (fine_ratio_val >= fine_ratio) if fine_ratio > 0 else True
        fine_inner_area = fine_spot_area
        if inner_iterations > 0:
            fine_inner_mask = cv2.erode(mask_fine, ero_kernel, iterations=inner_iterations)
            fine_inner_valid = cv2.bitwise_and(valid_fine, valid_fine, mask=fine_inner_mask)
            fine_inner_area = cv2.countNonZero(fine_inner_valid)
        fine_inner_ratio = (
            fine_inner_area / max(1, fine_spot_area) if fine_spot_area > 0 else 0
        )
        meets_fine_inner = (fine_inner_ratio >= inner_ratio_limit) if fine_spot_area > 0 else False
        if fine_spot_area > spot_final_threshold and meets_fine_ratio and meets_fine_inner:
            is_defective = True
            spot_area = fine_spot_area

    if debug:
        debug_view = cv2.bitwise_and(blackhat_img, blackhat_img, mask=mask_analysis)
        return {
            "is_defective": is_defective,
            "spot_area": spot_area,
            "texture_std": texture_std,
            "lab_std": a_std,
            "dark_delta": dark_delta,
            "median_intensity": median_intensity,
            "debug_image": debug_view,
        }

    return {
        "is_defective": is_defective,
        "spot_area": spot_area,
        "texture_std": texture_std,
        "lab_std": a_std,
        "dark_delta": dark_delta,
        "median_intensity": median_intensity,
    }
