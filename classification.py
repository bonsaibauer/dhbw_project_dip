import csv
import json
import os

import numpy as np

from settings import (
    get_classifier_rules,
    get_label_class_map,
    get_label_priorities,
    get_label_rules,
    get_paths,
    get_sort_log,
)

PATHS = get_paths()
PIPELINE_CSV = PATHS["PIPELINE_CSV"]
CLASSIFIER_RULES = get_classifier_rules()
LABEL_CLASS_MAP = get_label_class_map()
LABEL_PRIORITIES = get_label_priorities()
SORT_LOG = get_sort_log()


LABEL_RULES = get_label_rules()

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


def compute_symmetry_score(areas, sensitivity):
    if not areas:
        return 0.0
    mean_val = float(np.mean(areas))
    if mean_val <= 0:
        return 0.0
    std_val = float(np.std(areas))
    cv_value = std_val / mean_val if mean_val else 0.0
    raw_score = 100 * (1 - (cv_value * sensitivity))
    return max(0.0, min(100.0, round(raw_score, 1)))


def compute_features(row, sym_sen):
    areas = json.loads(row.get("window_areas") or "[]")
    num_windows = parse_int(row.get("num_windows"))
    has_center_hole = parse_bool(row.get("has_center_hole"))
    total_holes = num_windows + (1 if has_center_hole else 0)
    avg_window = float(np.mean(areas)) if areas else 0.0
    min_window = float(np.min(areas)) if areas else 0.0
    max_window = float(np.max(areas)) if areas else 0.0
    if min_window > 0:
        window_ratio = max_window / min_window
    else:
        window_ratio = 1.0
    symmetry_score = compute_symmetry_score(areas, sym_sen)

    geo_area = parse_float(row.get("area"))
    geo_convex = parse_float(row.get("convex_area"))
    hull_ratio = (geo_convex / geo_area) if geo_area else 0.0

    spot_area = parse_int(row.get("spot_area"))
    texture_std = parse_float(row.get("texture_std"))
    lab_std = parse_float(row.get("lab_std"))
    dark_delta = parse_float(row.get("dark_delta"))
    color_flag = parse_bool(row.get("spot_is_defective"))

    features = {
        "areas": areas,
        "num_windows": num_windows,
        "has_center_hole": has_center_hole,
        "total_holes": total_holes,
        "avg_window": avg_window,
        "window_ratio": window_ratio,
        "symmetry": symmetry_score,
        "hull_ratio": hull_ratio,
        "edge_damage": parse_float(row.get("edge_damage")),
        "edge_segments": parse_int(row.get("edge_segments")),
        "fragment_count": parse_int(row.get("fragment_count")),
        "outer_count": parse_int(row.get("outer_count")),
        "is_anomaly": parse_bool(row.get("is_anomaly")),
        "has_object": parse_bool(row.get("has_object")),
        "spot_area": spot_area,
        "texture_std": texture_std,
        "lab_std": lab_std,
        "dark_delta": dark_delta,
        "color_flag": color_flag,
    }
    color_threshold = 40
    features["has_color"] = (
        color_flag
        or spot_area >= color_threshold
        or texture_std >= 12
        or lab_std >= 5
        or dark_delta >= 18
    )
    return features


def check_condition(value, condition):
    op = condition.get("op", ">=")
    if op == "between":
        min_val = condition.get("min", float("-inf"))
        max_val = condition.get("max", float("inf"))
        return min_val <= value <= max_val
    target = condition.get("value")
    if op == ">=":
        return value >= target
    if op == "<=":
        return value <= target
    if op == ">":
        return value > target
    if op == "<":
        return value < target
    if op == "==":
        return value == target
    if op == "!=":
        return value != target
    if op == "in":
        return value in condition.get("values", [])
    return False


def evaluate_label_rule(rule, features):
    score = rule.get("base_score", 0.0)
    matched_reasons = []
    for condition in rule.get("conditions", []):
        metric_value = features.get(condition.get("metric"), 0)
        if check_condition(metric_value, condition):
            score += condition.get("weight", 1.0)
            template = condition.get("reason")
            if template:
                try:
                    matched_reasons.append(template.format(**features))
                except (KeyError, ValueError):
                    matched_reasons.append(template)
    if score >= rule.get("min_score", 1.0):
        reason = "; ".join(matched_reasons[:2]) or rule.get("fallback_reason", "")
        return {
            "label": rule["label"],
            "score": round(score, 3),
            "reason": reason,
        }
    return None


def evaluate_label_ruleset(features):
    decisions = []
    for rule in LABEL_RULES:
        result = evaluate_label_rule(rule, features)
        if not result:
            continue
        target_class = LABEL_CLASS_MAP.get(rule["label"], rule["label"].title())
        result["class"] = target_class
        decisions.append(result)
    return decisions


def select_decision(decisions):
    if not decisions:
        return None
    decisions.sort(
        key=lambda item: (
            -item["score"],
            LABEL_PRIORITIES.get(item["class"].lower(), 100),
            item["label"],
        )
    )
    return decisions[0]


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

    sym_sen = classifier_rules.get('SYM_SEN', 1.0)

    predictions = []
    total_files = len(rows)

    for idx, row in enumerate(rows, 1):
        rel_path = row.get('relative_path', '')
        filename = row.get('filename') or os.path.basename(row.get('source_path', ''))

        features = compute_features(row, sym_sen)
        decisions = evaluate_label_ruleset(features)
        decision = select_decision(decisions)
        if decision is None:
            fallback_label = 'rest' if features.get('is_anomaly') else 'normal'
            decision = {
                'label': fallback_label,
                'class': LABEL_CLASS_MAP.get(fallback_label, fallback_label.title()),
                'score': 0.0,
                'reason': 'Fallback',
            }

        label_title = decision['label'].title()
        class_name = decision['class']
        decision_reason = decision.get('reason') or 'Regel erfüllt'
        reason = f"{label_title}: {decision_reason}"

        if class_name == 'Normal':
            file_prefix = f"{features['symmetry']:06.2f}_"
        else:
            file_prefix = ''

        new_filename = f"{file_prefix}{filename}"

        row['target_label'] = decision['label']
        row['target_class'] = class_name
        row['reason'] = reason
        row['destination_filename'] = new_filename

        predictions.append(
            {
                'relative_path': rel_path,
                'predicted': class_name,
                'label': decision['label'],
                'source_path': row.get('source_path', ''),
                'reason': reason,
                'original_name': filename,
            }
        )

        if sort_log and total_files > 0:
            progress_bar('  Klassifizierung', idx, total_files)

    if sort_log and total_files > 0:
        print()

    required_columns = ['target_label', 'target_class', 'reason', 'destination_filename']
    final_fields = ensure_columns(fieldnames, required_columns)
    store_rows(csv_path, final_fields, rows)

    return predictions


def main():
    classify_from_csv(PIPELINE_CSV, CLASSIFIER_RULES, SORT_LOG)


if __name__ == "__main__":
    main()
