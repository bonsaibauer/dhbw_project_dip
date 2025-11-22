import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

OUTER_BREAK_SENSITIVITY = 0.78
MAX_RADIUS_JUMP = 6.0
LOCAL_VARIANCE_THRESHOLD = 3.2
MIN_OBJECT_AREA = 5000

MIN_WINDOWS_FOR_BRUCH = 5
MIN_WINDOW_AREA = 10
MAX_ALLOWED_CORNERS = 3
MIN_PEAK_DISTANCE = 60


def check_local_variance(distances, window_size=20):
    padded = np.pad(distances, (window_size // 2, window_size // 2), mode='wrap')
    local_std = []
    for i in range(len(distances)):
        segment = padded[i: i + window_size]
        local_std.append(np.std(segment))
    return np.array(local_std)


def get_radial_profile(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, (0, 0)

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    distances = []
    for point in contour:
        px, py = point[0]
        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        distances.append(dist)

    return np.array(distances), (cx, cy)


def count_peaks(values, window=10, min_dist=200):
    n = len(values)
    if n < window:
        return 0

    smoothed = np.convolve(values, np.ones(window) / window, mode='same')

    candidates = []
    lookahead = 5
    padded = np.pad(smoothed, (lookahead, lookahead), mode='wrap')

    for i in range(n):
        current_val = padded[i + lookahead]
        segment = padded[i: i + 2 * lookahead + 1]

        if current_val == np.max(segment) and current_val > np.min(segment) + 2:
            if np.argmax(segment) == lookahead:
                candidates.append((i, current_val))

    if not candidates:
        return 0

    candidates.sort(key=lambda x: x[1], reverse=True)

    final_peaks = []

    for idx, val in candidates:
        is_too_close = False
        for kept_idx in final_peaks:
            dist = abs(idx - kept_idx)
            dist = min(dist, n - dist)

            if dist < min_dist:
                is_too_close = True
                break

        if not is_too_close:
            final_peaks.append(idx)

    return len(final_peaks)


def analyze_snack_geometry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours_ext, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours_ext:
        return "Rest", "Kein Objekt"
    outer_contour = max(contours_ext, key=cv2.contourArea)

    (x_fl, y_fl), _ = cv2.minEnclosingCircle(outer_contour)
    cX, cY = int(x_fl), int(y_fl)

    dists_outer = []
    for p in outer_contour:
        dists_outer.append(np.sqrt((p[0][0] - cX) ** 2 + (p[0][1] - cY) ** 2))
    dists_outer = np.array(dists_outer)

    if len(dists_outer) > 0:
        w = 15
        d_smooth = np.convolve(dists_outer, np.ones(w) / w, mode='same')
        median_r = np.median(dists_outer)
        if len(np.where(d_smooth < median_r * OUTER_BREAK_SENSITIVITY)[0]) > 10:
            return "Bruch", "�u�erer Bruch: Tiefe"
        grad = np.abs(np.gradient(d_smooth))
        if len(grad) > 20 and np.max(grad[10:-10]) > MAX_RADIUS_JUMP:
            return "Bruch", "�u�erer Bruch: Kante"
        loc_var = check_local_variance(dists_outer, window_size=15)
        if np.max(loc_var) > LOCAL_VARIANCE_THRESHOLD:
            return "Bruch", f"�u�erer Bruch: Unruhig (Var {np.max(loc_var):.1f})"

    contours_all, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    valid_windows = []

    if hierarchy is not None:
        for i, cnt in enumerate(contours_all):
            if hierarchy[0][i][3] != -1 and cv2.contourArea(cnt) > MIN_WINDOW_AREA:
                Mh = cv2.moments(cnt)
                if Mh["m00"] != 0:
                    hx, hy = int(Mh["m10"] / Mh["m00"]), int(Mh["m01"] / Mh["m00"])
                    if np.sqrt((hx - cX) ** 2 + (hy - cY) ** 2) > 30:
                        valid_windows.append(cnt)

    num_windows = len(valid_windows)
    if num_windows < MIN_WINDOWS_FOR_BRUCH:
        return "Rest", f"Fragmentiert ({num_windows})"
    if num_windows < 6:
        return "Bruch", f"Zu wenig Fenster ({num_windows})"
    if num_windows > 6:
        return "Rest", f"Zu viele Fenster ({num_windows})"

    for idx, w_cnt in enumerate(valid_windows):
        radii, _ = get_radial_profile(w_cnt)
        if radii is None or len(radii) < 10:
            continue

        corners = count_peaks(radii, window=8, min_dist=MIN_PEAK_DISTANCE)

        if corners > MAX_ALLOWED_CORNERS:
            return "Bruch", f"Innerer Bruch: Fenster {idx + 1} hat {corners} Ecken (Max 3)"

    return "Normal", "OK"


def create_visual_report(image_paths, output_file="analysis_report.png"):
    num_images = min(5, len(image_paths))
    if num_images == 0:
        return
    print(f"[bruch.py] Erstelle grafischen Bericht...")

    fig, axes = plt.subplots(4, num_images, figsize=(4 * num_images, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    if num_images == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]], [axes[3]]])

    for idx, img_path in enumerate(image_paths[:num_images]):
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours_ext, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        d_smooth_outer = []
        var_outer = []
        cX, cY = 0, 0
        if contours_ext:
            outer = max(contours_ext, key=cv2.contourArea)
            (x_fl, y_fl), _ = cv2.minEnclosingCircle(outer)
            cX, cY = int(x_fl), int(y_fl)
            do = [np.sqrt((p[0][0] - cX) ** 2 + (p[0][1] - cY) ** 2) for p in outer]
            if do:
                d_smooth_outer = np.convolve(do, np.ones(15) / 15, mode='same')
                var_outer = check_local_variance(do, window_size=15)

        contours_all, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        inner_profiles = []

        if hierarchy is not None:
            for i, c in enumerate(contours_all):
                if hierarchy[0][i][3] != -1 and cv2.contourArea(c) > MIN_WINDOW_AREA:
                    Mh = cv2.moments(c)
                    if Mh["m00"] != 0:
                        hx, hy = int(Mh["m10"] / Mh["m00"]), int(Mh["m01"] / Mh["m00"])
                        if np.sqrt((hx - cX) ** 2 + (hy - cY) ** 2) > 30:
                            rads, _ = get_radial_profile(c)
                            if len(rads) > 0:
                                x_old = np.linspace(0, 1, len(rads))
                                x_new = np.linspace(0, 1, 100)
                                rads_interp = np.interp(x_new, x_old, rads)
                                corners = count_peaks(rads, window=8, min_dist=MIN_PEAK_DISTANCE)
                                inner_profiles.append((rads_interp, corners))

        axes[0, idx].imshow(mask, cmap='gray')
        axes[0, idx].set_title(os.path.basename(img_path), fontsize=8)
        axes[0, idx].axis('off')

        if len(d_smooth_outer) > 0:
            axes[1, idx].plot(d_smooth_outer, color='blue')
            med = np.median(d_smooth_outer)
            axes[1, idx].axhline(med * OUTER_BREAK_SENSITIVITY, color='red', linestyle='--')
        axes[1, idx].set_title("Au�en: Radius", fontsize=8)

        if len(var_outer) > 0:
            axes[2, idx].plot(var_outer, color='green')
            axes[2, idx].axhline(LOCAL_VARIANCE_THRESHOLD, color='red', linestyle='--')
        axes[2, idx].set_title("Au�en: Varianz", fontsize=8)

        if inner_profiles:
            for prof, corners in inner_profiles:
                color = 'red' if corners > MAX_ALLOWED_CORNERS else 'tab:blue'
                alpha = 0.8 if corners > MAX_ALLOWED_CORNERS else 0.3
                lw = 2 if corners > MAX_ALLOWED_CORNERS else 1
                axes[3, idx].plot(prof, color=color, alpha=alpha, linewidth=lw)
            axes[3, idx].set_title(f"Innen: Profile ({len(inner_profiles)})", fontsize=8)
            axes[3, idx].grid(True, alpha=0.3)
        else:
            axes[3, idx].text(0.5, 0.5, "Keine Fenster", ha='center')

    plt.tight_layout()
    plt.savefig(output_file, dpi=100)
    plt.close()


def sort_images(source_dir, target_dir):
    print("\n[bruch.py] Starte Analyse (Geometrie + Peak Merging)...")
    CLASSES = ["Normal", "Bruch", "Rest"]
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    for c in CLASSES:
        os.makedirs(os.path.join(target_dir, c), exist_ok=True)

    stats = {k: 0 for k in CLASSES}
    collected_files = {"Normal": [], "Bruch": [], "Rest": []}

    for root, dirs, files in os.walk(source_dir):
        for file_name in files:
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            src_path = os.path.join(root, file_name)
            img = cv2.imread(src_path)
            if img is None:
                continue

            cat, reason = analyze_snack_geometry(img)
            if cat not in CLASSES:
                cat = "Rest"

            parent = os.path.basename(root)
            if os.path.abspath(root) == os.path.abspath(source_dir):
                name = file_name
            else:
                name = f"{parent}_{file_name}"

            dst = os.path.join(target_dir, cat, name)
            shutil.copy(src_path, dst)
            stats[cat] += 1
            collected_files[cat].append(src_path)
            if cat == "Bruch":
                print(f"   [Bruch] {name} -> {reason}")

    print(f"[bruch.py] Fertig: {stats}")
