import shutil
import cv2
import numpy as np
import os
import csv

# ==========================================
# ANPASSBARE PARAMETER & VERZEICHNISSE
# ==========================================
# --- Pfad-Setup (FIXED mit BASE_DIR für Robustheit) ---
# Dieser Block stellt sicher, dass die Pfade immer stimmen, egal von wo du das Skript startest
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

RAW_DIR = os.path.join(BASE_DIR, "data", "Images")  # Sucht "data" neben dem Skript
OUT_DIR = os.path.join(BASE_DIR, "output") # Erstellt "output" neben dem Skript
PROC_DIR = os.path.join(OUT_DIR, "processed")
SORT_DIR = os.path.join(OUT_DIR, "sorted")
FAIL_DIR = os.path.join(SORT_DIR, "Falsch")
ANNO_FILE = os.path.join(BASE_DIR, "data", "image_anno.csv") # Findet die CSV
SORT_LOG = True  # True = ausführliche Konsolenlogs

# Neuer Pfad für dein Symmetrie-Ranking
RANKING_DIR = os.path.join(OUT_DIR, "Symmetrie-Ranking-V2")


# --- Preproc-Pipeline (prep_img, prep_set) ---
HSV_LO = np.array([35, 40, 30])
HSV_HI = np.array([85, 255, 255])
CNT_MINA = 30000
WARP_SZ = (600, 400)
TGT_W = 400
TGT_H = 400

# --- Geometrie-Features (cnt_hier, geom_feat, sort_run) ---
EPS_FACT = 0.04
HOLE_MIN = 100 # Dein Wert war 10, der Profi-Wert 100. Ich nehme den Profi-Wert.
WIND_MIN = 500
CTR_MAXA = 3000
FRAG_MIN = 6000 # Dein Wert war 3000, der Profi-Wert 6000. Ich nehme den Profi-Wert.
RWA_BASE = 4000
RWA_STRG = 3500
RWA_CMP = 3400
RWA_LRG = 4300
RHL_BASE = 1.05
RHL_STRG = 1.08
RWR_BASE = 3.0
RWR_STRG = 4.5
RMULT_SP = 120

# --- Farb-/Spotprüfung (spot_det, sort_run) ---
ERO_KN = (5, 5)
ERO_ITER = 4
BKH_KN = (15, 15)
BKH_CON = 30
NOI_KN = (2, 2)
SPT_MIN = 60
SPT_RAT = 0.0008
FERO_ITR = 1
SPT_FIN = 20
FSPT_RAT = 0.0008
TXT_STD = 15.0
INER_ITR = 2
INSP_RAT = 0.45
LAB_STD = 4.0
COL_SYM = 60
COL_SPT = 30
COL_LAB = 40
COL_STR = 80
BRK_SYM = 78
DRK_PCT = 5
DRK_DLT = 18
DRK_MED = 80
DRK_SPT = 30

# --- Kantenschaden & Symmetrie (sort_run) ---
EDGE_DMG = 1.05
EDGE_SEG = 14
SYM_SEN = 3.0

LABEL_PRIORITIES = {
    "normal": 0,
    "different colour spot": 1,
    "similar colour spot": 1,
    "burnt": 1,
    "middle breakage": 2,
    "corner or edge breakage": 2,
    "fryum stuck together": 3,
    "small scratches": 3,
    "other": 3,
}

LABEL_CLASS_MAP = {
    "normal": "Normal",
    "middle breakage": "Bruch",
    "corner or edge breakage": "Bruch",
    "fryum stuck together": "Rest",
    "different colour spot": "Farbfehler",
    "similar colour spot": "Farbfehler",
    "burnt": "Farbfehler",
    "small scratches": "Rest",
    "other": "Rest",
}

CLASS_DESCRIPTIONS = {
    "Normal": "7 Löcher, symmetrisch und ohne sichtbare Flecken.",
    "Bruch": "Lochanzahl passt nicht -> Bruch oder Fragment.",
    "Farbfehler": "Flecken/Schatten erkannt, obwohl Geometrie ok.",
    "Rest": "Kein Objekt oder unklare Form/Fall außerhalb der Regeln.",
}

# ==========================================
# TEIL 1: BILDVORVERARBEITUNG (PROFI-CODE)
# ==========================================

def prep_img(image, result):
    image_copy = image.copy()
    image_work = image.copy()
    hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask_object = cv2.bitwise_not(mask_green)
    image_work = cv2.bitwise_and(image_work, image_work, mask=mask_object)

    _, thresh = cv2.threshold(cv2.cvtColor(image_work, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    processed = False
    for ele in contours:
        if cv2.contourArea(ele) > CNT_MINA:
            rect = cv2.minAreaRect(ele)
            if rect[1][1] > rect[1][0]:
                rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)
            
            boxf = cv2.boxPoints(rect)
            boxf = np.int64(boxf)
            mask = np.zeros((image_copy.shape[0], image_copy.shape[1])).astype(np.uint8)
            cv2.drawContours(mask, [ele], -1, (255), cv2.FILLED)
            image_work[mask == 0] = (0, 0, 0)

            size = WARP_SZ
            dst_pts = np.array([[0, size[1]-1], [0, 0], [size[0]-1, 0], [size[0]-1, size[1]-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(boxf.astype("float32"), dst_pts)
            warped = cv2.warpPerspective(image_work, M, size, cv2.INTER_CUBIC)
            warped = cv2.resize(warped, (TGT_W, TGT_H), interpolation=cv2.INTER_CUBIC)

            result.append({"name": "Result", "data": warped})
            processed = True
    return processed

def prep_set(source_dir, target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
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
        
        if image is not None:
            res = []
            has_result = prep_img(image, res)
            
            if has_result:
                class_name = os.path.basename(root)
                if class_name not in class_dirs:
                    save_path = os.path.join(target_dir, class_name)
                    os.makedirs(save_path, exist_ok=True)
                    class_dirs[class_name] = save_path
                save_path = class_dirs[class_name]

                for item in res:
                    if item["name"] == "Result":
                        cv2.imwrite(os.path.join(save_path, name), item["data"])

        prog_bar("  Segmentierung", idx, total_files)

    print("\nSegmentierung abgeschlossen.")

# ==========================================
# TEIL 2: QUALITÄTSKONTROLLE (PROFI-CODE)
# ==========================================

def cnt_hier(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def geom_feat(contours, hierarchy):
    stats = {
        "has_object": False, "area": 0, "solidity": 0,
        "num_windows": 0, "has_center_hole": False, "main_contour": None,
        "fragment_count": 0, "convex_area": 0,
        "window_areas": [], "outer_count": 0, "edge_damage": 0.0, "edge_segments": 0
    }
    
    if not contours: return stats

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
    approx = cv2.approxPolyDP(main_cnt, EPS_FACT * main_peri, True)
    stats["edge_segments"] = len(approx)

    epsilon_factor = EPS_FACT
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            parent_idx = hierarchy[0][i][3]
            cnt_area = cv2.contourArea(cnt)

            if parent_idx == -1:
                stats["outer_count"] += 1
                if i != main_cnt_idx and cnt_area > FRAG_MIN:
                    stats["fragment_count"] += 1
                    continue

            if parent_idx == main_cnt_idx:
                hole_area = cv2.contourArea(cnt)
                if hole_area < HOLE_MIN: continue 

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
                corners = len(approx)

                if 3 <= corners <= 5 and hole_area > WIND_MIN:
                    stats["num_windows"] += 1
                    stats["window_areas"].append(hole_area)
                elif corners > 5 and hole_area < CTR_MAXA:
                    stats["has_center_hole"] = True
    
    if (
        not stats["has_center_hole"]
        and len(stats["window_areas"]) >= 6
    ):
        min_idx = int(np.argmin(stats["window_areas"]))
        min_area = stats["window_areas"][min_idx]
        if min_area <= CTR_MAXA:
            stats["has_center_hole"] = True
            stats["window_areas"].pop(min_idx)
            stats["num_windows"] = len(stats["window_areas"])
            
    return stats

def spot_det(image, spot_threshold=SPT_MIN, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask_obj = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel_erode = np.ones(ERO_KN, np.uint8)
    mask_analysis = cv2.erode(mask_obj, kernel_erode, iterations=ERO_ITER) 
    object_area = cv2.countNonZero(mask_analysis)
    
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, BKH_KN)
    blackhat_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_morph)

    contrast_threshold = BKH_CON 
    _, mask_defects = cv2.threshold(blackhat_img, contrast_threshold, 255, cv2.THRESH_BINARY)
    valid_defects = cv2.bitwise_and(mask_defects, mask_defects, mask=mask_analysis)
    valid_defects = cv2.morphologyEx(valid_defects, cv2.MORPH_OPEN, np.ones(NOI_KN, np.uint8))

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
        dark_percentile = float(np.percentile(object_pixels, DRK_PCT))
        dark_delta = median_intensity - dark_percentile
    
    inner_mask = mask_analysis
    inner_spot_area = spot_area
    if INER_ITR > 0:
        inner_mask = cv2.erode(mask_analysis, kernel_erode, iterations=INER_ITR)
        inner_valid = cv2.bitwise_and(valid_defects, valid_defects, mask=inner_mask)
        inner_spot_area = cv2.countNonZero(inner_valid)

    ratio = spot_area / max(1, object_area)
    meets_ratio = (ratio >= SPT_RAT) if SPT_RAT > 0 else True
    inner_ratio = inner_spot_area / max(1, spot_area)
    meets_inner = (inner_ratio >= INSP_RAT) if spot_area > 0 else False
    is_defective = spot_area > spot_threshold and meets_ratio and meets_inner

    if not is_defective and FERO_ITR > 0:
        mask_fine = cv2.erode(mask_obj, kernel_erode, iterations=FERO_ITR)
        fine_area_obj = cv2.countNonZero(mask_fine)
        valid_fine = cv2.bitwise_and(mask_defects, mask_defects, mask=mask_fine)
        valid_fine = cv2.morphologyEx(valid_fine, cv2.MORPH_OPEN, np.ones(NOI_KN, np.uint8))
        fine_spot_area = cv2.countNonZero(valid_fine)
        fine_ratio = fine_spot_area / max(1, fine_area_obj)
        meets_fine_ratio = (fine_ratio >= FSPT_RAT) if FSPT_RAT > 0 else True
        fine_inner_area = fine_spot_area
        if INER_ITR > 0:
            fine_inner_mask = cv2.erode(mask_fine, kernel_erode, iterations=INER_ITR)
            fine_inner_valid = cv2.bitwise_and(valid_fine, valid_fine, mask=fine_inner_mask)
            fine_inner_area = cv2.countNonZero(fine_inner_valid)
        fine_inner_ratio = fine_inner_area / max(1, fine_spot_area) if fine_spot_area > 0 else 0
        meets_fine_inner = (fine_inner_ratio >= INSP_RAT) if fine_spot_area > 0 else False
        if fine_spot_area > SPT_FIN and meets_fine_ratio and meets_fine_inner:
            is_defective = True
            spot_area = fine_spot_area

    if debug:
        debug_view = cv2.bitwise_and(blackhat_img, blackhat_img, mask=mask_analysis)
        return {"is_defective": is_defective, "spot_area": spot_area, "texture_std": texture_std, "lab_std": a_std, "debug_image": debug_view}

    return {
        "is_defective": is_defective,
        "spot_area": spot_area,
        "texture_std": texture_std,
        "lab_std": a_std,
        "dark_delta": dark_delta,
        "median_intensity": median_intensity
    }

def tbl_show(headers, rows, indent="  "):
    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value))) # str() für Zahlen
    header_line = indent + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    divider_line = indent + "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(divider_line)
    for row in rows:
        print(indent + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))))
    print()

def prog_bar(prefix, current, total, bar_length=30):
    if total <= 0: return
    ratio = min(max(current / total, 0), 1)
    filled = int(bar_length * ratio)
    bar = "#" * filled + "-" * (bar_length - filled)
    percent = ratio * 100
    print(f"\r{prefix} [{bar}] {percent:5.1f}% ({current}/{total})", end="", flush=True)

def img_list(source_dir):
    image_files = []
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append((root, name))
    return image_files

def prio_map():
    class_priority = {}
    for label, prio in LABEL_PRIORITIES.items():
        cls = LABEL_CLASS_MAP.get(label, label.title())
        if cls not in class_priority or prio < class_priority[cls]:
            class_priority[cls] = prio
    if not class_priority: return ""
    ordered = [name for name, _ in sorted(class_priority.items(), key=lambda item: item[1])]
    return " > ".join(ordered)

def path_rel(path):
    if not path: return ""
    normalized = path.replace("\\", "/")
    marker = "data/Images/" # Angepasst an deinen Pfad
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    return normalized.lstrip("/")

def label_map(raw_label):
    if not raw_label: return None
    candidates = [lbl.strip().lower() for lbl in raw_label.split(",") if lbl.strip()]
    if not candidates: return None
    candidates.sort(key=lambda lbl: LABEL_PRIORITIES.get(lbl, 100))
    return candidates[0]

def anno_load(annotation_file):
    annotations = {}
    if not os.path.exists(annotation_file):
        print(f"\nHinweis: '{annotation_file}' nicht gefunden, Validierung übersprungen.")
        return annotations

    with open(annotation_file, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rel_path = path_rel(row.get("image", ""))
            if not rel_path: continue
            base_label = label_map(row.get("label", ""))
            if not base_label: continue
            annotations[rel_path] = LABEL_CLASS_MAP.get(base_label, "Rest")
    return annotations

def miss_copy(pred_entry, expected_label, falsch_dir):
    os.makedirs(falsch_dir, exist_ok=True)
    rel_name = path_rel(pred_entry.get("relative_path", ""))
    if not rel_name:
        rel_name = os.path.basename(pred_entry["source_path"])
    rel_name = rel_name.replace("/", "_")
    base, ext = os.path.splitext(rel_name)
    safe_expected = expected_label.replace(" ", "_")
    safe_pred = pred_entry["predicted"].replace(" ", "_")
    new_name = f"{base}_gt-{safe_expected}_pred-{safe_pred}{ext}"
    dest_path = os.path.join(falsch_dir, new_name)
    shutil.copy(pred_entry["source_path"], dest_path)

def pred_chk(predictions, annotations, falsch_dir):
    if not annotations:
        print("\nKeine Annotationen geladen -> Validierung übersprungen.")
        return

    if os.path.exists(falsch_dir):
        shutil.rmtree(falsch_dir)

    total = 0
    correct = 0
    mismatches = []
    per_class = {}

    for pred in predictions:
        rel_path = path_rel(pred.get("relative_path", ""))
        expected = annotations.get(rel_path)
        if expected is None:
            continue
        total += 1
        cls_stats = per_class.setdefault(expected, {"total": 0, "correct": 0})
        cls_stats["total"] += 1
        if expected == pred["predicted"]:
            correct += 1
            cls_stats["correct"] += 1
        else:
            mismatches.append((pred, expected))

    if total == 0:
        print("\nKeine passenden Annotationen gefunden -> Validierung übersprungen.")
        return

    accuracy = (correct / total) * 100
    skipped = len(predictions) - total
    print("\nValidierung (image_anno.csv):")
    print("- Gesamtstatistik:")
    print()
    summary_headers = ["Statistik", "Wert"]
    summary_rows = [
        ["Bewertet", str(total)],
        ["Treffer", str(correct)],
        ["Genauigkeit %", f"{accuracy:.2f}"],
        ["Falsch zugeordnet", str(len(mismatches))],
    ]
    if skipped:
        summary_rows.append(["Ohne passende Annotation", str(skipped)])
    tbl_show(summary_headers, summary_rows)
    chain = prio_map()
    if chain:
        print(f"Priorisierung (höchste Priorität links): {chain}")
        print()

    if per_class:
        print("- Klassenübersicht:")
        headers = ["Klasse", "Erwartet", "Treffer", "Genauigkeit %"]
        rows = []
        for cls_name in sorted(per_class.keys()):
            stats = per_class[cls_name]
            acc_cls = (stats["correct"] / stats["total"]) * 100 if stats["total"] else 0.0
            rows.append([cls_name, str(stats["total"]), str(stats["correct"]), f"{acc_cls:.2f}"])
        tbl_show(headers, rows)

    if mismatches:
        for pred, expected in mismatches:
            miss_copy(pred, expected, falsch_dir)

def sort_run(source_data_dir, sorted_data_dir):
    print("\nStarte Sortierung und Symmetrie-Berechnung...")
    CLASSES = ["Normal", "Bruch", "Farbfehler", "Rest"]
    
    if os.path.exists(sorted_data_dir):
        shutil.rmtree(sorted_data_dir)
    for cls in CLASSES:
        os.makedirs(os.path.join(sorted_data_dir, cls), exist_ok=True)

    stats_counter = {k: 0 for k in CLASSES}
    reason_counter = {k: {} for k in CLASSES}
    predictions = []

    for root, dirs, files in os.walk(source_data_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            img_path = os.path.join(root, file)
            image = cv2.imread(img_path)
            if image is None: continue

            # HIER EIN FIX: rel_path war falsch für die Validierung
            # Es muss relativ zum PROC_DIR sein, nicht zum Unterordner (z.B. Normal)
            rel_path = os.path.relpath(img_path, source_data_dir)
            rel_path = rel_path.replace("\\", "/")
            
            contours, hierarchy = cnt_hier(image)
            geo = geom_feat(contours, hierarchy)
            
            target_class = "Normal"
            reason = "OK"
            file_prefix = "" 

            total_holes = geo["num_windows"] + (1 if geo["has_center_hole"] else 0)
            source_is_anomaly = "Anomaly" in img_path.replace("\\", "/") # Robusterer Check

            areas = geo["window_areas"]
            avg_window = np.mean(areas) if areas else 0
            hull_ratio = (geo.get("convex_area", 0) / max(1, geo.get("area", 0))) if geo.get("area") else 0
            edge_damage = geo.get("edge_damage", 0.0)
            edge_segments = geo.get("edge_segments", 0)
            window_ratio = (max(areas) / max(1, min(areas))) if areas and min(areas) > 0 else 1

            symmetry_score = 0.0
            if len(areas) > 0:
                mean_a = np.mean(areas)
                std_a = np.std(areas)
                cv = std_a / mean_a if mean_a > 0 else 0
                raw_score = 100 * (1 - (cv * SYM_SEN))
                symmetry_score = max(0.0, min(100.0, round(raw_score, 1)))

            rest_reason = None
            rest_strength = 0
            rest_window_hint = False
            rest_multi_hint = False
            rest_structural_hint = False
            
            # Rest- und Farb-Checks nur bei Anomaly-Bildern durchführen
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
                if hull_ratio >= RHL_STRG:
                    rest_hints.append((2, f"Hülle: {hull_ratio:.2f}"))
                    rest_structural_hint = True
                elif hull_ratio >= RHL_BASE:
                    rest_hints.append((1, f"Hülle: {hull_ratio:.2f}"))
                
                if areas and avg_window > 0:
                    if avg_window <= RWA_BASE and window_ratio >= RWR_BASE:
                        strong = (avg_window <= RWA_STRG and window_ratio >= RWR_STRG)
                        rest_hints.append((2 if strong else 1, f"Fensterverh.: {window_ratio:.1f}"))
                        rest_window_hint = True
                    if avg_window <= RWA_CMP:
                        rest_hints.append((1, f"Fenster klein: {avg_window:.0f}"))
                        rest_window_hint = True
                    if avg_window >= RWA_LRG and window_ratio <= 1.3:
                        rest_hints.append((1, f"Fenster groß: {avg_window:.0f}"))
                        rest_window_hint = True
                
                if rest_hints:
                    rest_strength, rest_reason = max(rest_hints, key=lambda item: item[0])

                if not rest_structural_hint:
                    rest_strength = min(rest_strength, 1)

                col_res = spot_det(image, spot_threshold=SPT_MIN)
                
                color_candidate = None
                color_strength = 0
                
                def assign_color(reason, strength):
                    nonlocal color_candidate, color_strength
                    if strength > color_strength:
                        color_candidate = ("Farbfehler", reason)
                        color_strength = strength

                if col_res["is_defective"]:
                    assign_color(f"Fleck: {col_res['spot_area']}px", 2)
                if col_res["spot_area"] >= COL_STR:
                    assign_color(f"Fleck: {col_res['spot_area']}px", 2)
                if (
                    col_res["spot_area"] >= COL_SPT
                    and col_res.get("texture_std", 0) > TXT_STD
                    and symmetry_score >= COL_SYM
                ):
                    assign_color(f"Textur: {col_res['texture_std']:.1f}", 1)
                if (
                    col_res["spot_area"] >= COL_LAB
                    and col_res.get("lab_std", 0) > LAB_STD
                    and symmetry_score >= COL_SYM
                ):
                    assign_color(f"Farbe: {col_res['lab_std']:.1f}", 1)
                if (
                    col_res.get("dark_delta", 0) > DRK_DLT
                    and col_res.get("median_intensity", 0) >= DRK_MED
                    and col_res.get("spot_area", 0) >= DRK_SPT
                    and symmetry_score >= COL_SYM
                ):
                    assign_color(f"Kontrast: {col_res['dark_delta']:.1f}", 1)

                if color_candidate and rest_strength > 1 and not rest_multi_hint and geo["fragment_count"] == 0:
                    rest_strength = 1

                multi_outer_spot = (
                    geo.get("outer_count", 0) > 1
                    and col_res.get("spot_area", 0) >= RMULT_SP
                )
                if multi_outer_spot:
                    rest_strength = max(rest_strength, 2)
                    rest_reason = rest_reason or f"Mehrfachobj.: {geo['outer_count']}"

                if color_candidate and rest_strength > 1 and not multi_outer_spot:
                    rest_strength = 1
            else:
                # Wenn source_is_anomaly False ist, setzen wir col_res auf Standardwerte
                col_res = {"is_defective": False, "spot_area": 0, "texture_std": 0, "lab_std": 0, "dark_delta": 0, "median_intensity": 0}
                color_candidate = None
                color_strength = 0


            # === Decision Level 1: grundlegende Guards ===
            if not geo["has_object"]:
                target_class = "Rest"
                reason = "Kein Objekt"
            elif total_holes < 7:
                # --- Level 2A: Lochanzahl < 7 ---
                if color_candidate and color_strength >= 2:
                    target_class, reason = color_candidate
                elif rest_strength >= 2:
                    target_class = "Rest"
                    reason = rest_reason or f"Unklare Form ({total_holes})"
                else:
                    target_class = "Bruch"
                    reason = f"Zu wenig Löcher: {total_holes}"
            elif total_holes > 7:
                # --- Level 2B: Lochanzahl > 7 ---
                target_class = "Rest"
                reason = f"Zu viele Fragmente: {total_holes}"
            else:
                # --- Level 3: Lochanzahl == 7 ---
                if source_is_anomaly and rest_strength >= 2:
                    target_class = "Rest"
                    reason = rest_reason or "Starker Resthinweis"
                else:
                    classified = False
                    if color_candidate and (color_strength >= 2 or rest_strength <= 1):
                        target_class, reason = color_candidate
                        classified = True
                    if not classified and (edge_damage >= EDGE_DMG or edge_segments >= EDGE_SEG) and color_strength < 2:
                        target_class = "Bruch"
                        reason = f"Kante: {edge_damage:.2f}"
                        classified = True
                    if not classified and color_candidate:
                        target_class, reason = color_candidate
                        classified = True
                    
                    if not classified:
                        if target_class == "Normal" and rest_strength >= 1 and rest_reason:
                            target_class = "Rest"
                            reason = rest_reason
                        elif source_is_anomaly and symmetry_score < BRK_SYM:
                            target_class = "Bruch"
                            reason = f"Asymmetrie: {symmetry_score:.2f}%"
                        elif target_class == "Normal":
                            reason = f"Symmetrie: {symmetry_score:.2f}%"
            
            # --- ENDE ENTSCHEIDUNGSBAUM ---

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
            reason_counter[target_class][reason] = reason_counter[target_class].get(reason, 0) + 1
            if SORT_LOG:
                rel_dest = os.path.relpath(dest_path, sorted_data_dir).replace("\\", "/")
                print(f"[{target_class}] {new_filename} | Grund: {reason} | Ziel: {rel_dest}")
            
            predictions.append({
                "relative_path": rel_path, # Hier war der Bug
                "predicted": target_class,
                "source_path": img_path,
                "destination_path": dest_path,
                "reason": reason,
                "original_name": file,
            })

    total_sorted = sum(stats_counter.values())
    print("\nSortierung abgeschlossen:\n")
    headers = ["Klasse", "Anzahl", "Anteil %", "Beschreibung", "Häufigster Grund"]
    rows = []
    for cls in CLASSES:
        amount = stats_counter[cls]
        share = (amount / total_sorted * 100) if total_sorted else 0
        desc = CLASS_DESCRIPTIONS.get(cls, "")
        reason_text = "-"
        if reason_counter[cls]:
            top_reason, top_count = max(reason_counter[cls].items(), key=lambda kv: kv[1])
            reason_text = f"{top_reason} ({top_count}x)"
        rows.append([cls, str(amount), f"{share:.1f}", desc, reason_text])
    tbl_show(headers, rows)

    return predictions

# ==========================================
# TEIL 3: SYMMETRIE-RANKING (DEINE LOGIK)
# ==========================================

def calculate_asymmetry_score(img):
    """
    Testet die 6-fache Rotationssymmetrie, indem alle 6 Positionen
    (0, 60, 120, 180, 240, 300 Grad) verglichen werden.
    Gibt die Gesamtfläche (Anzahl der Pixel) zurück, die NICHT
    in allen 6 Rotationen vorhanden ist.
    Niedriger Score = Besser.
    """
    if img is None:
        return -1 # Fehler beim Laden

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    (h, w) = img.shape[:2]

    # 1. Mittelpunkt (Centroid) finden
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return -1 # Kein Objekt gefunden
        
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    core_mask = mask.copy()
    
    # 2. Schleife durch die 5 Rotationen (60° bis 300°)
    for i in range(1, 6): # 1, 2, 3, 4, 5
        angle = i * 60
        R = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated_mask = cv2.warpAffine(mask, R, (w, h))
        core_mask = cv2.bitwise_and(core_mask, rotated_mask)

    # 3. Asymmetrie-Teile finden
    asymmetric_parts_mask = cv2.subtract(mask, core_mask)
    
    # 4. Score berechnen und zurückgeben
    score = cv2.countNonZero(asymmetric_parts_mask)
    
    return score

def run_ranking(source_folder, ranking_folder):
    """
    Nimmt den Ordner mit den "Normal"-Bildern und sortiert sie
    in einen neuen Ordner basierend auf dem Rotations-Symmetrie-Score.
    """
    print(f"\n--- PHASE 2: STARTE SYMMETRIE-RANKING (Simon's Methode) ---")
    print(f"Quelle: {source_folder}")
    print(f"Ziel:   {ranking_folder}")

    if os.path.exists(ranking_folder):
        shutil.rmtree(ranking_folder)
    os.makedirs(ranking_folder)

    image_files = []
    try:
        for filename in os.listdir(source_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(filename)
    except FileNotFoundError:
        print(f"--- FEHLER ---")
        print(f"Quellordner '{source_folder}' nicht gefunden!")
        return

    if not image_files:
        print("Keine Bilder im Quellordner gefunden.")
        return

    print(f"Analysiere {len(image_files)} 'Normal'-Bilder auf Symmetrie...")
    
    results = [] 
    
    # 1. Alle Bilder analysieren
    for idx, filename in enumerate(image_files, 1):
        full_path = os.path.join(source_folder, filename)
        img = cv2.imread(full_path) 
        
        if img is None:
            print(f"Konnte Bild nicht laden: {filename}")
            continue
            
        score = calculate_asymmetry_score(img)
        
        if score >= 0:
            results.append((score, full_path, filename))
        
        if idx % 50 == 0 or idx == len(image_files):
            print(f"   ...verarbeitet: {idx}/{len(image_files)} (Bild: {filename}, Score: {score})")

    # 2. Nach Score sortieren (niedrigster Score = beste Symmetrie)
    results.sort(key=lambda x: x[0])
    
    print("\nRanking abgeschlossen. Kopiere sortierte Dateien...")

    # 3. Sortierte Dateien in neuen Ordner kopieren
    for rank, (score, full_path, filename) in enumerate(results, 1):
        
        # Entfernt den alten Symmetrie-Prefix vom Profi-Code (z.B. "076.50_")
        original_filename = filename
        parts = filename.split("_", 1)
        if len(parts) == 2 and parts[0].replace('.', '').isdigit():
            original_filename = parts[1]

        new_filename = f"{rank:03d}_Score-{score:05d}_{original_filename}"
        dest_path = os.path.join(ranking_folder, new_filename)
        
        shutil.copy(full_path, dest_path)

    print(f"\nFertig! {len(results)} Bilder wurden nach '{ranking_folder}' kopiert.")


# ==========================================
# MAIN PROGRAMM (ALS PIPELINE)
# ==========================================

if __name__ == '__main__':
    
    # --- ANFANG PHASE 1: KLASSIFIZIERUNG (Profi-Code) ---
    print("--- PHASE 1: STARTE KLASSIFIZIERUNG (Profi-Code) ---")
    
    # Pfade robust machen
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    RAW_DIR = os.path.join(BASE_DIR, "data", "Images")
    OUT_DIR = os.path.join(BASE_DIR, "output")
    PROC_DIR = os.path.join(OUT_DIR, "processed")
    SORT_DIR = os.path.join(OUT_DIR, "sorted")
    FAIL_DIR = os.path.join(SORT_DIR, "Falsch")
    ANNO_FILE = os.path.join(BASE_DIR, "data", "image_anno.csv")
    
    # Neuer Pfad für dein Symmetrie-Ranking
    RANKING_DIR = os.path.join(SORT_DIR, "Symmetrie-Ranking-V2")


    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(RAW_DIR):
        # 1. Vorverarbeitung starten
        prep_set(RAW_DIR, PROC_DIR)
        
        # 2. Sortierung starten
        if os.path.exists(PROC_DIR):
            predictions = sort_run(PROC_DIR, SORT_DIR)
            annotations = anno_load(ANNO_FILE)
            pred_chk(predictions, annotations, FAIL_DIR)
            
            print("\n--- PHASE 1: KLASSIFIZIERUNG ABGESCHLOSSEN ---")
            
            # --- START PHASE 2: SYMMETRIE-RANKING (Simon's Code) ---
            
            # 1. Input-Pfad (der "Normal"-Ordner, den Phase 1 erstellt hat)
            normal_folder_path = os.path.join(SORT_DIR, "Normal")

            # 2. Deine modifizierte Funktion aufrufen
            if os.path.exists(normal_folder_path):
                run_ranking(normal_folder_path, RANKING_DIR) # RANKING_DIR Variable nutzen
            else:
                print(f"Fehler in Phase 2: 'Normal'-Ordner ({normal_folder_path}) nicht gefunden.")

    else:
        print(f"Fehler: Quellordner '{RAW_DIR}' nicht gefunden!")