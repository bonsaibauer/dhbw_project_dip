import os
import shutil
import cv2
import numpy as np

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
            return "Bruch", "Äußerer Bruch: Tiefe"
        grad = np.abs(np.gradient(d_smooth))
        if len(grad) > 20 and np.max(grad[10:-10]) > MAX_RADIUS_JUMP:
            return "Bruch", "Äußerer Bruch: Kante"
        loc_var = check_local_variance(dists_outer, window_size=15)
        if np.max(loc_var) > LOCAL_VARIANCE_THRESHOLD:
            return "Bruch", f"Äußerer Bruch: Unruhig (Var {np.max(loc_var):.1f})"

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


def sort_images(source_dir, target_dir):
    print("\n[bruch.py] Starte Analyse (Geometrie + Peak Merging)...")
    classes = ["Normal", "Bruch", "Rest"]
    shutil.rmtree(target_dir, ignore_errors=True)
    for c in classes:
        os.makedirs(os.path.join(target_dir, c), exist_ok=True)

    stats = {k: 0 for k in classes}
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
            if cat not in classes:
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
