import csv
import json
import os

import numpy as np

from settings import get_classifier_rules, get_paths, get_sort_log

PATHS = get_paths()
PIPELINE_CSV = PATHS["PIPELINE_CSV"]
CLASSIFIER_RULES = get_classifier_rules()
SORT_LOG = get_sort_log()


def progress_bar(prefix, current, total, bar_len=30):
    if total <= 0:
        return
    ratio = min(max(current / total, 0), 1)
    filled = int(bar_len * ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    label = prefix.ljust(20)
    print(f"\r{label}[{bar}] {ratio * 100:5.1f}% ({current}/{total})", end="", flush=True)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def load_feature_rows(csv_path):
    if not os.path.exists(csv_path):
        print(f"Fehler: CSV '{csv_path}' nicht gefunden.")
        return [], []

    with open(csv_path, encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [row for row in reader]
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def ensure_columns(fieldnames, required):
    updated = list(fieldnames)
    for column in required:
        if column not in updated:
            updated.append(column)
    return updated


def store_rows(csv_path, fieldnames, rows):
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def classify_from_csv(csv_path, classifier_rules, sort_log):
    rows, fieldnames = load_feature_rows(csv_path)
    if not rows:
        return []

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

    predictions = []

    total_files = len(rows)

    for idx, row in enumerate(rows, 1):
        areas = json.loads(row.get("window_areas") or "[]")
        rel_path = row.get("relative_path", "")
        filename = row.get("filename") or os.path.basename(row.get("source_path", ""))

        geo_area = parse_float(row.get("area"))
        geo_convex = parse_float(row.get("convex_area"))
        edge_damage = parse_float(row.get("edge_damage"))
        edge_segments = parse_int(row.get("edge_segments"))
        num_windows = parse_int(row.get("num_windows"))
        has_center_hole = parse_bool(row.get("has_center_hole"))
        fragment_count = parse_int(row.get("fragment_count"))
        outer_count = parse_int(row.get("outer_count"))
        has_object = parse_bool(row.get("has_object"))
        is_anomaly = parse_bool(row.get("is_anomaly"))

        color_entries = {
            "is_defective": parse_bool(row.get("spot_is_defective")),
            "spot_area": parse_int(row.get("spot_area")),
            "texture_std": parse_float(row.get("texture_std")),
            "lab_std": parse_float(row.get("lab_std")),
            "dark_delta": parse_float(row.get("dark_delta")),
            "median_intensity": parse_float(row.get("median_intensity")),
        }

        target_class = "Normal"
        reason = "OK"
        file_prefix = ""

        total_holes = num_windows + (1 if has_center_hole else 0)
        avg_window = np.mean(areas) if areas else 0
        hull_ratio = (geo_convex / max(1, geo_area)) if geo_area else 0
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
        if is_anomaly:
            rest_hints = []
            if fragment_count > 0:
                rest_hints.append((3, f"Fragmente: {fragment_count}"))
                rest_structural_hint = True
            if outer_count > 1:
                strength = 2 if outer_count > 2 else 1
                rest_hints.append((strength, f"Mehrfachobj.: {outer_count}"))
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
                    rest_hints.append((1, f"Fenster klein: {avg_window:.0f}"))
                    rest_window_hint = True
                if avg_window >= rwa_lrg and window_ratio <= 1.3:
                    rest_hints.append((1, f"Fenster groß: {avg_window:.0f}"))
                    rest_window_hint = True
            if rest_hints:
                rest_strength, rest_reason = max(rest_hints, key=lambda item: item[0])

        if not rest_structural_hint:
            rest_strength = min(rest_strength, 1)

        if is_anomaly:
            col_res = color_entries
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
        if is_anomaly:
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

        if color_candidate and rest_strength > 1 and not rest_multi_hint and fragment_count == 0:
            rest_strength = 1

        multi_outer_spot = (
            is_anomaly
            and outer_count > 1
            and col_res.get("spot_area", 0) >= rmult_sp
        )
        if multi_outer_spot:
            rest_strength = max(rest_strength, 2)
            rest_reason = rest_reason or f"Mehrfachobj.: {outer_count}"

        if color_candidate and rest_strength > 1 and not multi_outer_spot:
            rest_strength = 1

        if not has_object:
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
            if is_anomaly and rest_strength >= 2:
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
                    elif is_anomaly and symmetry_score < brk_sym:
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

        new_filename = f"{file_prefix}{filename}"

        row["target_class"] = target_class
        row["reason"] = reason
        row["destination_filename"] = new_filename

        predictions.append(
            {
                "relative_path": rel_path,
                "predicted": target_class,
                "source_path": row.get("source_path", ""),
                "reason": reason,
                "original_name": filename,
            }
        )

        if sort_log and total_files > 0:
            progress_bar("  Klassifizierung", idx, total_files)

    if sort_log and total_files > 0:
        print()

    required_columns = ["target_class", "reason", "destination_filename"]
    final_fields = ensure_columns(fieldnames, required_columns)
    store_rows(csv_path, final_fields, rows)

    return predictions


def main():
    classify_from_csv(PIPELINE_CSV, CLASSIFIER_RULES, SORT_LOG)


if __name__ == "__main__":
    main()
