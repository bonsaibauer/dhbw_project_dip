import cv2
import numpy as np
import os
import shutil


def detect_defects(image, spot_threshold=43, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask_obj = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel_erode = np.ones((13, 13), np.uint8)
    mask_analysis = cv2.erode(mask_obj, kernel_erode, iterations=1)
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    blackhat_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_morph)
    _, mask_defects_contrast = cv2.threshold(blackhat_img, 45, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_burn = np.array([0, 30, 0])
    upper_burn = np.array([180, 255, 95])
    mask_burn = cv2.inRange(hsv, lower_burn, upper_burn)

    combined_defects = cv2.bitwise_or(mask_defects_contrast, mask_burn)
    valid_defects = cv2.bitwise_and(combined_defects, combined_defects, mask=mask_analysis)
    valid_defects = cv2.morphologyEx(valid_defects, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(valid_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    significant_contours = []
    total_defect_area = 0
    MIN_SPOT_SIZE = 35

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_SPOT_SIZE:
            significant_contours.append(cnt)
            total_defect_area += area

    is_defective = total_defect_area > spot_threshold

    return {
        "is_defective": is_defective,
        "spot_area": total_defect_area,
        "contours": significant_contours
    }


def run_color_check(sorted_dir):
    print("\n[farb.py] Starte Farbpr�fung (Strenge Filterung + Rand-Ignoranz)...")

    defect_dir = os.path.join(sorted_dir, "Farbfehler")
    os.makedirs(defect_dir, exist_ok=True)

    check_classes = ["Normal"]
    moved_count = 0

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

                result = detect_defects(image, spot_threshold=20)

                if result["is_defective"]:
                    cv2.drawContours(image, result["contours"], -1, (0, 0, 255), 2)

                    for cnt in result["contours"]:
                        (x, y), radius = cv2.minEnclosingCircle(cnt)
                        center = (int(x), int(y))
                        radius = int(radius) + 8
                        cv2.circle(image, center, radius, (0, 0, 255), 2)

                    target_path = os.path.join(defect_dir, file_name)
                    cv2.imwrite(target_path, image)

                    try:
                        os.remove(file_path)
                        moved_count += 1
                    except OSError as e:
                        print(f"Fehler beim L�schen von {file_path}: {e}")

    print(f"[farb.py] Farbpr�fung abgeschlossen. {moved_count} Bilder markiert und verschoben.")
