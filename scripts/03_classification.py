
import csv
import json
import os
from functools import lru_cache

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_dir = os.path.join(project_root, "config")
path_path = os.path.join(config_dir, "path.json")
class_path = os.path.join(config_dir, "classification.json")


def load_json(path, error_msg):
    if not os.path.exists(path):
        raise FileNotFoundError(error_msg)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_path_config():
    return load_json(
        path_path,
        f"Pfaddatei '{path_path}' nicht gefunden.",
    )


@lru_cache(maxsize=1)
def class_config():
    return load_json(
        class_path,
        f"Klassifikationsdatei '{class_path}' nicht gefunden.",
    )


def norm_path(path_value):
    return os.path.normpath(path_value) if path_value else ""


def read_path_section(cfg_name):
    return load_path_config().get(cfg_name, {})


def load_paths():
    return {
        key: norm_path(value)
        for key, value in read_path_section("paths").items()
    }


def map_labels():
    return dict(class_config().get("label_class_map", {}))


def rank_labels():
    return dict(class_config().get("label_priorities", {}))


def rule_list():
    return list(class_config().get("label_rules", []))


SORT_LOG_ENABLED = True

path_map = load_paths()
pipe_csv = path_map["pipeline_csv_path"]
label_map = map_labels()
label_rank = rank_labels()
sort_flag = SORT_LOG_ENABLED
rule_defs = rule_list()
WINDOW_SIZE_VARIANCE_SENSITIVITY = 1.1
OUTER_BREAK_LOW_FRACTION_THRESHOLD = 0.02
OUTER_BREAK_MIN_POINTS = 10
OUTER_BREAK_MIN_RATIO_THRESHOLD = 0.5
INNER_BREAK_MIN_WINDOWS = 6
INNER_BREAK_DEVIATION_TOLERANCE = 0.12
ANOMALY_INNER_BREAK_TOLERANCE = 0.04
ANOMALY_COLOR_SPOT_LIMIT = 60
INNER_BREAK_DEVIATION_COUNT_THRESHOLD = 1


# ---------------------------------------------------------------------------
# CSV Helpers
# ---------------------------------------------------------------------------

def read_rows(csv_path):
    if not os.path.exists(csv_path):
        print(f"Fehler: CSV '{csv_path}' nicht gefunden.")
        return [], []

    with open(csv_path, encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [row for row in reader]
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def ensure_cols(fieldnames, required):
    updated = list(fieldnames)
    for column in required:
        if column not in updated:
            updated.append(column)
    return updated


def write_rows(csv_path, fieldnames, rows):
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Progress Feedback
# ---------------------------------------------------------------------------

def show_progress(prefix, current, total, bar_len=30):
    if total <= 0:
        return
    ratio = min(max(current / total, 0), 1)
    filled = int(bar_len * ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    label = prefix.ljust(20)
    print(
        f"\r{label}[{bar}] {ratio * 100:5.1f}% ({current}/{total})",
        end="",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Parsing & Feature Extraction
# ---------------------------------------------------------------------------

def parse_flag(value):
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


def window_size_variance_score(areas, sensitivity):
    if not areas:
        return 0.0
    mean_val = float(np.mean(areas))
    if mean_val <= 0:
        return 0.0
    std_val = float(np.std(areas))
    cv_value = std_val / mean_val if mean_val else 0.0
    raw_score = 100 * (1 - (cv_value * sensitivity))
    return max(0.0, min(100.0, round(raw_score, 1)))


def extract_metrics(row):
    window_areas = json.loads(row.get("geometry_window_area_list") or "[]")
    window_count = parse_int(row.get("geometry_window_count"))
    has_center_hole = parse_flag(row.get("geometry_has_center_hole"))
    total_holes = window_count + (1 if has_center_hole else 0)

    avg_window = float(np.mean(window_areas)) if window_areas else 0.0
    min_window = float(np.min(window_areas)) if window_areas else 0.0
    max_window = float(np.max(window_areas)) if window_areas else 0.0
    window_ratio = (max_window / min_window) if min_window > 0 else 1.0
    variance_score = window_size_variance_score(
        window_areas, WINDOW_SIZE_VARIANCE_SENSITIVITY
    )

    geo_area = parse_float(row.get("geometry_area"))
    geo_convex = parse_float(row.get("geometry_convex_area"))
    hull_ratio = (geo_convex / geo_area) if geo_area else 0.0

    spot_area = parse_int(row.get("color_spot_area"))
    texture_std = parse_float(row.get("color_texture_stddev"))
    lab_std = parse_float(row.get("color_lab_stddev"))
    dark_delta = parse_float(row.get("color_dark_delta"))
    color_flag = parse_flag(row.get("color_detection_flag"))
    symmetry_score = parse_float(row.get("symmetry_score"))

    features = {
        "geometry_window_area_list": window_areas,
        "geometry_window_count": window_count,
        "geometry_has_center_hole": has_center_hole,
        "geometry_total_hole_count": total_holes,
        "geometry_window_area_avg": avg_window,
        "geometry_window_area_ratio": window_ratio,
        "geometry_window_size_variance_score": variance_score,
        "geometry_hull_ratio": hull_ratio,
        "geometry_edge_damage_ratio": parse_float(
            row.get("geometry_edge_damage_ratio")
        ),
        "geometry_edge_segment_count": parse_int(
            row.get("geometry_edge_segment_count")
        ),
        "geometry_fragment_count": parse_int(
            row.get("geometry_fragment_count")
        ),
        "geometry_outer_contour_count": parse_int(
            row.get("geometry_outer_contour_count")
        ),
        "geometry_outer_radius_min_ratio": parse_float(
            row.get("geometry_outer_radius_min_ratio"), 1.0
        ),
        "geometry_outer_radius_low_fraction": parse_float(
            row.get("geometry_outer_radius_low_fraction")
        ),
        "geometry_window_area_deviation_max": parse_float(
            row.get("geometry_window_area_deviation_max")
        ),
        "geometry_window_area_deviation_count": parse_int(
            row.get("geometry_window_area_deviation_count")
        ),
        "geometry_outer_radius_low_count": parse_int(
            row.get("geometry_outer_radius_low_count")
        ),
        "geometry_outer_radius_point_count": parse_int(
            row.get("geometry_outer_radius_point_count")
        ),
        "pipeline_has_anomaly_flag": parse_flag(
            row.get("pipeline_has_anomaly_flag")
        ),
        "geometry_has_primary_object": parse_flag(
            row.get("geometry_has_primary_object")
        ),
        "symmetry_score": symmetry_score,
        "color_spot_area": spot_area,
        "color_texture_stddev": texture_std,
        "color_lab_stddev": lab_std,
        "color_dark_delta": dark_delta,
        "color_detection_flag": color_flag,
        "relative_path": row.get("relative_path", ""),
    }

    color_threshold = 40
    features["color_issue_detected"] = (
        color_flag
        or spot_area >= color_threshold
        or texture_std >= 12
        or lab_std >= 5
        or dark_delta >= 18
    )
    return features


# ---------------------------------------------------------------------------
# Rule Evaluation
# ---------------------------------------------------------------------------

def match_metric(value, condition):
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


def score_rule(rule, features):
    score = rule.get("base_score", 0.0)
    matched_reasons = []
    for condition in rule.get("conditions", []):
        metric_value = features.get(condition.get("metric"), 0)
        if match_metric(metric_value, condition):
            score += condition.get("weight", 1.0)
            template = condition.get("reason")
            if template:
                try:
                    matched_reasons.append(template.format(**features))
                except (KeyError, ValueError):
                    matched_reasons.append(template)
    if score >= rule.get("min_score", 1.0):
        reason = "; ".join(matched_reasons[:2]) or rule.get("fallback_reason", "")
        return {"label": rule["label"], "score": round(score, 3), "reason": reason}
    return None


def eval_rules(features):
    decisions = []
    for rule in rule_defs:
        result = score_rule(rule, features)
        if not result:
            continue
        result["class"] = label_map.get(rule["label"], rule["label"].title())
        decisions.append(result)
    return decisions


def pick_decision(decisions):
    if not decisions:
        return None
    decisions.sort(
        key=lambda item: (
            -item["score"],
            label_rank.get(item["class"].lower(), 100),
            item["label"],
        )
    )
    return decisions[0]


def fallback_pick(features):
    fallback_label = (
        "rest" if features.get("pipeline_has_anomaly_flag") else "normal"
    )
    fallback_class = label_map.get(fallback_label, fallback_label.title())
    return {
        "label": fallback_label,
        "class": fallback_class,
        "score": 0.0,
        "reason": "Fallback",
    }


def format_reason(decision):
    label_title = decision["label"].title()
    detail = decision.get("reason") or "Regel erfüllt"
    return f"{label_title}: {detail}"


def classify_row(row):
    features = extract_metrics(row)
    decisions = eval_rules(features)
    decisions.extend(bruch_decisions(features))
    decision = pick_decision(decisions) or fallback_pick(features)
    return decision, features


def bruch_decisions(features):
    """Mirror bruch.py logic using extracted metrics for inner/outer breaks."""
    if not features.get("pipeline_has_anomaly_flag"):
        return []

    color_spot_area = features.get("color_spot_area", 0)
    if color_spot_area > ANOMALY_COLOR_SPOT_LIMIT:
        return []

    decisions = []

    window_areas = features.get("geometry_window_area_list") or []
    valid_areas = [area for area in window_areas if area > 0]
    low_fraction = features.get("geometry_outer_radius_low_fraction", 0.0)
    low_count = features.get("geometry_outer_radius_low_count", 0)
    point_count = features.get("geometry_outer_radius_point_count", 0)
    min_ratio = features.get("geometry_outer_radius_min_ratio", 1.0)

    top_windows = sorted(valid_areas, reverse=True)[:INNER_BREAK_MIN_WINDOWS]
    inner_reasons = []
    if len(top_windows) < INNER_BREAK_MIN_WINDOWS:
        inner_reasons.append(
            f"Weniger Fenster: {len(top_windows)}/{INNER_BREAK_MIN_WINDOWS}"
        )
    elif top_windows:
        mean_area = float(np.mean(top_windows))
        if mean_area > 0:
            anomaly_flag = features.get("pipeline_has_anomaly_flag", False)
            inner_threshold = INNER_BREAK_DEVIATION_TOLERANCE
            if anomaly_flag:
                inner_threshold = min(
                    inner_threshold, ANOMALY_INNER_BREAK_TOLERANCE
                )
            deviations = [
                abs(area - mean_area) / mean_area for area in top_windows
            ]
            max_dev = max(deviations) if deviations else 0.0
            if max_dev >= inner_threshold:
                inner_reasons.append(
                    f"Fensterabweichung: {max_dev:.2f}"
                )
            if sum(
                dev >= inner_threshold for dev in deviations
            ) >= INNER_BREAK_DEVIATION_COUNT_THRESHOLD:
                inner_reasons.append("Mehrere abweichende Fenster")

    outer_reasons = []
    if low_count >= OUTER_BREAK_MIN_POINTS:
        outer_reasons.append(
            f"Konturpunkte <75%: {low_count}/{point_count or '?'}"
        )
    elif low_fraction >= OUTER_BREAK_LOW_FRACTION_THRESHOLD:
        outer_reasons.append(
            f"Kontur unter 75% Anteil: {low_fraction:.2%}"
        )
    if min_ratio <= OUTER_BREAK_MIN_RATIO_THRESHOLD:
        outer_reasons.append(f"Minimaler Radiusanteil: {min_ratio:.2f}")

    def add_decision(label, reason, score):
        decisions.append(
            {
                "label": label,
                "class": label_map.get(label, label.title()),
                "score": score,
                "reason": reason,
            }
        )

    if outer_reasons:
        add_decision(
            "corner or edge breakage",
            "; ".join(outer_reasons),
            9.0,
        )

    if inner_reasons:
        add_decision(
            "middle breakage",
            "; ".join(inner_reasons),
            9.0,
        )

    if outer_reasons or inner_reasons:
        combined = outer_reasons + inner_reasons
        add_decision(
            "bruch",
            "; ".join(combined),
            9.5 if len(combined) > 1 else 9.2,
        )

    return decisions


# ---------------------------------------------------------------------------
# Classification Workflow
# ---------------------------------------------------------------------------

def classify_csv(csv_path, sort_log):
    rows, fieldnames = read_rows(csv_path)
    if not rows:
        return []

    predictions = []
    total_files = len(rows)

    for idx, row in enumerate(rows, 1):
        rel_path = row.get("relative_path", "")
        filename = row.get("filename") or os.path.basename(row.get("source_path", ""))

        decision, features = classify_row(row)
        class_name = decision["class"]
        reason = format_reason(decision)
        variance_score = features.get(
            "geometry_window_size_variance_score", 0.0
        )

        row["target_label"] = decision["label"]
        row["target_class"] = class_name
        row["reason"] = reason
        row["geometry_window_size_variance_score"] = f"{variance_score:.4f}"

        predictions.append(
            {
                "relative_path": rel_path,
                "predicted": class_name,
                "label": decision["label"],
                "source_path": row.get("source_path", ""),
                "reason": reason,
                "original_name": filename,
            }
        )

        if sort_log and total_files > 0:
            show_progress("  Klassifizierung", idx, total_files)

    if sort_log and total_files > 0:
        print()

    required_columns = [
        "target_label",
        "target_class",
        "reason",
        "geometry_window_size_variance_score",
        "symmetry_score",
    ]
    final_fields = ensure_cols(fieldnames, required_columns)
    write_rows(csv_path, final_fields, rows)

    return predictions


def classify_cli():
    classify_csv(
        pipe_csv,
        sort_flag,
    )


if __name__ == "__main__":
    classify_cli()
