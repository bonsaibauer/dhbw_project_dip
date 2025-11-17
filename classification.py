import os
import shutil
import stat

import cv2
import numpy as np

from image_processing import detect_spots
from validation import render_table, normalize_path


def progress_bar(prefix, current, total, bar_len=30):
    if total <= 0:
        return
    ratio = min(max(current / total, 0), 1)
    filled = int(bar_len * ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix} [{bar}] {ratio * 100:5.1f}% ({current}/{total})", end="", flush=True)


def extract_hierarchy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def compute_geometry(contours, hierarchy, geometry_settings):
    """Zählt Fenster, Fragmente und liefert Geometriekennzahlen."""
    eps_fact = geometry_settings["EPS_FACT"]
    hole_min = geometry_settings["HOLE_MIN"]
    wind_min = geometry_settings["WIND_MIN"]
    ctr_maxa = geometry_settings["CTR_MAXA"]
    frag_min = geometry_settings["FRAG_MIN"]

    stats = {
        "has_object": False,
        "area": 0,
        "solidity": 0,
        "num_windows": 0,
        "has_center_hole": False,
        "main_contour": None,
        "fragment_count": 0,
        "convex_area": 0,
        "window_areas": [],
        "outer_count": 0,
        "edge_damage": 0.0,
    }

    if not contours:
        return stats

    main_cnt_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    main_cnt = contours[main_cnt_idx]
    area = cv2.contourArea(main_cnt)

    stats["has_object"] = True
    stats["area"] = area
    stats["main_contour"] = main_cnt
    hull = cv2.convexHull(main_cnt)
    stats["convex_area"] = cv2.contourArea(hull)
    hull_peri = cv2.arcLength(hull, True)
    main_peri = cv2.arcLength(main_cnt, True)
    stats["edge_damage"] = (hull_peri / max(1.0, main_peri)) if main_peri > 0 else 0.0
    approx = cv2.approxPolyDP(main_cnt, eps_fact * main_peri, True)
    stats["edge_segments"] = len(approx)

    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            parent_idx = hierarchy[0][i][3]
            cnt_area = cv2.contourArea(cnt)

            if parent_idx == -1:
                stats["outer_count"] += 1
                if i != main_cnt_idx and cnt_area > frag_min:
                    stats["fragment_count"] += 1
                    continue

            if parent_idx == main_cnt_idx:
                hole_area = cv2.contourArea(cnt)
                if hole_area < hole_min:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                approx_hole = cv2.approxPolyDP(cnt, eps_fact * perimeter, True)
                corners = len(approx_hole)

                if 3 <= corners <= 5 and hole_area > wind_min:
                    stats["num_windows"] += 1
                    stats["window_areas"].append(hole_area)
                elif corners > 5 and hole_area < ctr_maxa:
                    stats["has_center_hole"] = True

    if not stats["has_center_hole"] and len(stats["window_areas"]) >= 6:
        min_idx = int(np.argmin(stats["window_areas"]))
        min_area = stats["window_areas"][min_idx]
        if min_area <= ctr_maxa:
            stats["has_center_hole"] = True
            stats["window_areas"].pop(min_idx)
            stats["num_windows"] = len(stats["window_areas"])

    return stats


def run_pipeline(
    source_data_dir,
    sorted_data_dir,
    geometry_settings,
    spot_settings,
    classifier_rules,
    sort_log,
):
    """Komplette Klassifikationspipeline inkl. Dateikopie."""
    rwa_base = classifier_rules["RWA_BASE"]
    rwa_strg = classifier_rules["RWA_STRG"]
    rwa_cmp = classifier_rules["RWA_CMP"]
    rwa_lrg = classifier_rules["RWA_LRG"]
    rhl_base = classifier_rules["RHL_BASE"]
    rhl_strg = classifier_rules["RHL_STRG"]
    rwr_base = classifier_rules["RWR_BASE"]
    rwr_strg = classifier_rules["RWR_STRG"]
    rmult_sp = classifier_rules["RMULT_SP"]
    col_str = classifier_rules["COL_STR"]
    col_spt = classifier_rules["COL_SPT"]
    txt_std = classifier_rules["TXT_STD"]
    col_sym = classifier_rules["COL_SYM"]
    col_lab = classifier_rules["COL_LAB"]
    lab_std = classifier_rules["LAB_STD"]
    drk_dlt = classifier_rules["DRK_DLT"]
    sym_sen = classifier_rules["SYM_SEN"]
    edge_dmg = classifier_rules["EDGE_DMG"]
    edge_seg = classifier_rules["EDGE_SEG"]
    brk_sym = classifier_rules["BRK_SYM"]

    print("\nStarte Sortierung und Symmetrie-Berechnung...")
    classes = ["Normal", "Bruch", "Farbfehler", "Rest"]

    if os.path.exists(sorted_data_dir):
        def _on_rm_error(func, path, exc_info):
            error = exc_info[1]
            if isinstance(error, PermissionError):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            else:
                raise

        shutil.rmtree(sorted_data_dir, onerror=_on_rm_error)
    for cls in classes:
        os.makedirs(os.path.join(sorted_data_dir, cls), exist_ok=True)

    stats_counter = {k: 0 for k in classes}
    predictions = []

    image_files = [
        (root, file)
        for root, _, files in os.walk(source_data_dir)
        for file in files
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    total_files = len(image_files)
    if sort_log and total_files > 0:
        print(f"Sortierung ({total_files} Bilder):")

    for idx, (root, file) in enumerate(image_files, 1):
        img_path = os.path.join(root, file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rel_path = normalize_path(os.path.relpath(img_path, source_data_dir))
        contours, hierarchy = extract_hierarchy(image)
        geo = compute_geometry(contours, hierarchy, geometry_settings)

        target_class = "Normal"
        reason = "OK"
        file_prefix = ""

        total_holes = geo["num_windows"] + (1 if geo["has_center_hole"] else 0)
        source_is_anomaly = "Anomaly" in img_path

        areas = geo["window_areas"]
        avg_window = np.mean(areas) if areas else 0
        hull_ratio = (
            geo.get("convex_area", 0) / max(1, geo.get("area", 0))
            if geo.get("area")
            else 0
        )
        edge_damage = geo.get("edge_damage", 0.0)
        edge_segments = geo.get("edge_segments", 0)
        window_ratio = (
            (max(areas) / max(1, min(areas))) if areas and min(areas) > 0 else 1
        )

        symmetry_score = 0.0
        if len(areas) > 0:
            mean_a = np.mean(areas)
            std_a = np.std(areas)
            cv_value = std_a / mean_a if mean_a > 0 else 0
            raw_score = 100 * (1 - (cv_value * sym_sen))
            symmetry_score = max(0.0, min(100.0, round(raw_score, 1)))

        rest_reason = None
        rest_strength = 0
        rest_window_hint = False
        rest_multi_hint = False
        rest_structural_hint = False
        if source_is_anomaly:
            rest_hints = []
            if geo["fragment_count"] > 0:
                rest_hints.append((3, f"Fragmente: {geo['fragment_count']}"))
                rest_structural_hint = True
            if geo.get("outer_count", 0) > 1:
                strength = 2 if geo["outer_count"] > 2 else 1
                rest_hints.append((strength, f"Mehrfachobj.: {geo['outer_count']}"))
                rest_multi_hint = True
                rest_structural_hint = True
            if hull_ratio >= rhl_strg:
                rest_hints.append((2, f"Hülle: {hull_ratio:.2f}"))
                rest_structural_hint = True
            elif hull_ratio >= rhl_base:
                rest_hints.append((1, f"Hülle: {hull_ratio:.2f}"))
            if areas and avg_window > 0:
                if avg_window <= rwa_base and window_ratio >= rwr_base:
                    strong = avg_window <= rwa_strg and window_ratio >= rwr_strg
                    rest_hints.append(
                        (2 if strong else 1, f"Fensterverh.: {window_ratio:.1f}")
                    )
                    rest_window_hint = True
                if avg_window <= rwa_cmp:
                    rest_hints.append(
                        (1, f"Fenster klein: {avg_window:.0f}")
                    )
                    rest_window_hint = True
                if avg_window >= rwa_lrg and window_ratio <= 1.3:
                    rest_hints.append(
                        (1, f"Fenster groß: {avg_window:.0f}")
                    )
                    rest_window_hint = True
            if rest_hints:
                rest_strength, rest_reason = max(rest_hints, key=lambda item: item[0])

        if not rest_structural_hint:
            rest_strength = min(rest_strength, 1)

        if source_is_anomaly:
            col_res = detect_spots(image, spot_settings)
        else:
            col_res = {
                "is_defective": False,
                "spot_area": 0,
                "texture_std": 0,
                "lab_std": 0,
                "dark_delta": 0,
                "median_intensity": 0,
            }

        color_candidate = None
        color_strength = 0
        if source_is_anomaly:
            def assign_color(reason_text, strength):
                nonlocal color_candidate, color_strength
                if strength > color_strength:
                    color_candidate = ("Farbfehler", reason_text)
                    color_strength = strength

            if col_res["is_defective"]:
                assign_color(f"Fleck: {col_res['spot_area']}px", 2)
            if col_res["spot_area"] >= col_str:
                assign_color(f"Fleck: {col_res['spot_area']}px", 2)
            if (
                col_res["spot_area"] >= col_spt
                and col_res.get("texture_std", 0) > txt_std
                and symmetry_score >= col_sym
            ):
                assign_color(f"Textur: {col_res['texture_std']:.1f}", 1)
            if (
                col_res["spot_area"] >= col_lab
                and col_res.get("lab_std", 0) > lab_std
                and symmetry_score >= col_sym
            ):
                assign_color(f"Farbe: {col_res['lab_std']:.1f}", 1)
            if col_res.get("dark_delta", 0) > drk_dlt and symmetry_score >= col_sym:
                assign_color(f"Dunkelanteil: {col_res['dark_delta']:.1f}", 1)

        if color_candidate and rest_strength > 1 and not rest_multi_hint and geo["fragment_count"] == 0:
            rest_strength = 1

        multi_outer_spot = (
            source_is_anomaly
            and geo.get("outer_count", 0) > 1
            and col_res.get("spot_area", 0) >= rmult_sp
        )
        if multi_outer_spot:
            rest_strength = max(rest_strength, 2)
            rest_reason = rest_reason or f"Mehrfachobj.: {geo['outer_count']}"

        if color_candidate and rest_strength > 1 and not multi_outer_spot:
            rest_strength = 1

        if not geo["has_object"]:
            target_class = "Rest"
            reason = "Kein Objekt"
        elif total_holes < 7:
            if color_candidate and color_strength >= 2:
                target_class, reason = color_candidate
            elif rest_strength >= 2:
                target_class = "Rest"
                reason = rest_reason or f"Unklare Form ({total_holes})"
            else:
                target_class = "Bruch"
                reason = f"Zu wenig Löcher: {total_holes}"
        elif total_holes > 7:
            target_class = "Rest"
            reason = f"Zu viele Fragmente: {total_holes}"
        else:
            if source_is_anomaly and rest_strength >= 2:
                target_class = "Rest"
                reason = rest_reason or "Starker Resthinweis"
            else:
                classified = False
                if color_candidate and (color_strength >= 2 or rest_strength <= 1):
                    target_class, reason = color_candidate
                    classified = True
                if (
                    not classified
                    and (edge_damage >= edge_dmg or edge_segments >= edge_seg)
                    and color_strength < 2
                ):
                    target_class = "Bruch"
                    reason = f"Kante: {edge_damage:.2f}"
                    classified = True
                if not classified and color_candidate:
                    target_class, reason = color_candidate
                    classified = True
                else:
                    if target_class == "Normal":
                        reason = f"Symmetrie: {symmetry_score:.2f}%"

                if not classified:
                    if target_class == "Normal" and rest_strength >= 1 and rest_reason:
                        target_class = "Rest"
                        reason = rest_reason
                    elif source_is_anomaly and symmetry_score < brk_sym:
                        target_class = "Bruch"
                        reason = f"Asymmetrie: {symmetry_score:.2f}%"
                    elif target_class == "Normal":
                        reason = f"Symmetrie: {symmetry_score:.2f}%"

        if target_class == "Normal":
            file_prefix = f"{symmetry_score:06.2f}_"
            if not reason or reason == "OK":
                reason = f"Symmetrie: {symmetry_score:.2f}%"
        else:
            file_prefix = ""

        new_filename = f"{file_prefix}{file}"
        dest_path = os.path.join(sorted_data_dir, target_class, new_filename)
        shutil.copy(img_path, dest_path)

        stats_counter[target_class] += 1

        predictions.append(
            {
                "relative_path": rel_path,
                "predicted": target_class,
                "source_path": img_path,
                "destination_path": dest_path,
                "reason": reason,
                "original_name": file,
            }
        )

        if sort_log and total_files > 0:
            progress_bar("  Sortierung", idx, total_files)

    if sort_log and total_files > 0:
        print("\nSortierung abgeschlossen.")

    total_sorted = sum(stats_counter.values())
    print("\nErgebnisübersicht:\n")
    headers = ["Klasse", "Anzahl", "Anteil %"]
    rows = []
    for cls in classes:
        amount = stats_counter[cls]
        share = (amount / total_sorted * 100) if total_sorted else 0
        rows.append([cls, str(amount), f"{share:.1f}"])
    render_table(headers, rows)

    return predictions
