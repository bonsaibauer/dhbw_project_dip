import json
import os
from functools import lru_cache

import numpy as np

BASE_DIR = os.path.dirname(__file__)
PARAMETER_FILE = os.path.join(BASE_DIR, "parameter.json")
CLASSIFICATION_FILE = os.path.join(BASE_DIR, "classification.json")


def _normalize_path_value(path_value):
    return os.path.normpath(path_value) if path_value else ""


def _load_json_file(path, error_msg):
    if not os.path.exists(path):
        raise FileNotFoundError(error_msg)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _get_parameter_section(cfg_name):
    return load_parameter_config().get(cfg_name, {})


def _coerce_config_value(cfg, key, default, transform=None):
    value = cfg.get(key, default)
    return transform(value) if transform else value


@lru_cache(maxsize=1)
def load_parameter_config():
    return _load_json_file(
        PARAMETER_FILE,
        f"Parameterdatei '{PARAMETER_FILE}' nicht gefunden.",
    )


@lru_cache(maxsize=1)
def load_classification_config():
    return _load_json_file(
        CLASSIFICATION_FILE,
        f"Klassifikationsdatei '{CLASSIFICATION_FILE}' nicht gefunden.",
    )


def fetch_pipeline_paths():
    return {
        key: _normalize_path_value(value)
        for key, value in _get_parameter_section("paths").items()
    }


def is_sort_logging_enabled():
    return bool(load_parameter_config().get("sort_log", True))


def fetch_preprocessing_settings():
    cfg = _get_parameter_section("preprocessing")
    return {
        "preprocess_hsv_lower": np.array(
            _coerce_config_value(cfg, "preprocess_hsv_lower", [0, 0, 0])
        ),
        "preprocess_hsv_upper": np.array(
            _coerce_config_value(cfg, "preprocess_hsv_upper", [0, 0, 0])
        ),
        "minimum_contour_area": _coerce_config_value(cfg, "minimum_contour_area", 0),
        "warp_frame_size": tuple(
            _coerce_config_value(cfg, "warp_frame_size", [0, 0])
        ),
        "target_width": _coerce_config_value(cfg, "target_width", 0),
        "target_height": _coerce_config_value(cfg, "target_height", 0),
    }


def fetch_geometry_settings():
    cfg = _get_parameter_section("geometry")
    return {
        "polygon_epsilon_factor": _coerce_config_value(
            cfg,
            "polygon_epsilon_factor",
            0.0,
        ),
        "minimum_hole_area": _coerce_config_value(cfg, "minimum_hole_area", 0),
        "minimum_window_area": _coerce_config_value(cfg, "minimum_window_area", 0),
        "maximum_center_area": _coerce_config_value(cfg, "maximum_center_area", 0),
        "minimum_fragment_area": _coerce_config_value(cfg, "minimum_fragment_area", 0),
    }


def fetch_spot_detection_settings():
    cfg = _get_parameter_section("spot")
    return {
        "erosion_kernel_size": tuple(
            _coerce_config_value(cfg, "erosion_kernel_size", [0, 0])
        ),
        "erosion_iterations": _coerce_config_value(cfg, "erosion_iterations", 0),
        "blackhat_kernel_size": tuple(
            _coerce_config_value(cfg, "blackhat_kernel_size", [0, 0])
        ),
        "blackhat_contrast_threshold": _coerce_config_value(
            cfg,
            "blackhat_contrast_threshold",
            0,
        ),
        "noise_kernel_size": tuple(
            _coerce_config_value(cfg, "noise_kernel_size", [0, 0])
        ),
        "minimum_spot_area": _coerce_config_value(cfg, "minimum_spot_area", 0),
        "spot_area_ratio": _coerce_config_value(cfg, "spot_area_ratio", 0.0),
        "fine_erosion_iterations": _coerce_config_value(
            cfg,
            "fine_erosion_iterations",
            0,
        ),
        "inner_erosion_iterations": _coerce_config_value(
            cfg,
            "inner_erosion_iterations",
            0,
        ),
        "inner_spot_ratio": _coerce_config_value(cfg, "inner_spot_ratio", 0.0),
        "fine_spot_ratio": _coerce_config_value(cfg, "fine_spot_ratio", 0.0),
        "fine_spot_area": _coerce_config_value(cfg, "fine_spot_area", 0),
        "dark_percentile": _coerce_config_value(cfg, "dark_percentile", 0),
    }


def fetch_classifier_rules():
    return dict(load_parameter_config().get("classifier_rules", {}))


def fetch_label_priorities():
    return dict(load_classification_config().get("label_priorities", {}))


def fetch_label_class_mapping():
    return dict(load_classification_config().get("label_class_map", {}))


def fetch_label_rules():
    return list(load_classification_config().get("label_rules", []))


PATHS = fetch_pipeline_paths()
RAW_IMAGE_DIR = PATHS["raw_image_directory"]
PIPELINE_OUTPUT_DIR = PATHS["pipeline_output_directory"]
PROCESSED_IMAGE_DIR = PATHS["processed_image_directory"]
PIPELINE_CSV_PATH = PATHS["pipeline_csv_path"]
SORTED_OUTPUT_DIR = PATHS["sorted_output_directory"]
FAILED_VALIDATION_DIR = PATHS["failed_validation_directory"]
ANNOTATION_FILE_PATH = PATHS["annotation_file_path"]

PREPROCESSING_SETTINGS = fetch_preprocessing_settings()
GEOMETRY_SETTINGS = fetch_geometry_settings()
SPOT_SETTINGS = fetch_spot_detection_settings()
CLASSIFIER_RULES = fetch_classifier_rules()
SORT_LOG_ENABLED = is_sort_logging_enabled()
LABEL_PRIORITIES = fetch_label_priorities()
LABEL_CLASS_MAP = fetch_label_class_mapping()


def run_complete_inspection_pipeline():
    from classification import classify_pipeline_from_csv
    from image_processing import process_directory_to_csv
    from segmentation import segment_directory_images
    from sorting import sort_images_from_pipeline_csv
    from validation import (
        load_annotation_classes,
        validate_predictions_against_annotations,
    )

    os.makedirs(PIPELINE_OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(RAW_IMAGE_DIR):
        print(f"Fehler: Quellordner '{RAW_IMAGE_DIR}' nicht gefunden.")
        return

    segment_directory_images(
        RAW_IMAGE_DIR,
        PROCESSED_IMAGE_DIR,
        PREPROCESSING_SETTINGS,
    )
    process_directory_to_csv(
        PROCESSED_IMAGE_DIR,
        PIPELINE_CSV_PATH,
        GEOMETRY_SETTINGS,
        SPOT_SETTINGS,
    )
    predictions = classify_pipeline_from_csv(
        PIPELINE_CSV_PATH,
        CLASSIFIER_RULES,
        SORT_LOG_ENABLED,
    )
    sort_images_from_pipeline_csv(
        PIPELINE_CSV_PATH,
        SORTED_OUTPUT_DIR,
        SORT_LOG_ENABLED,
    )

    annotations = load_annotation_classes(
        ANNOTATION_FILE_PATH,
        LABEL_PRIORITIES,
        LABEL_CLASS_MAP,
    )
    validate_predictions_against_annotations(
        predictions,
        annotations,
        FAILED_VALIDATION_DIR,
        LABEL_PRIORITIES,
        LABEL_CLASS_MAP,
    )


if __name__ == "__main__":
    run_complete_inspection_pipeline()
