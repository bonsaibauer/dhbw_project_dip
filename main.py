import json
import os
from functools import lru_cache

import numpy as np

BASE_DIR = os.path.dirname(__file__)
PARAMETER_FILE = os.path.join(BASE_DIR, "parameter.json")
CLASSIFICATION_FILE = os.path.join(BASE_DIR, "classification.json")


def _norm_path(path_value):
    return os.path.normpath(path_value) if path_value else ""


def _load_json(path, error_msg):
    if not os.path.exists(path):
        raise FileNotFoundError(error_msg)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _section(cfg_name):
    return load_parameters().get(cfg_name, {})


def _cget(cfg, key, default, transform=None):
    value = cfg.get(key, default)
    return transform(value) if transform else value


@lru_cache(maxsize=1)
def load_parameters():
    return _load_json(PARAMETER_FILE, f"Parameterdatei '{PARAMETER_FILE}' nicht gefunden.")


@lru_cache(maxsize=1)
def load_classification_settings():
    return _load_json(
        CLASSIFICATION_FILE,
        f"Klassifikationsdatei '{CLASSIFICATION_FILE}' nicht gefunden.",
    )


def get_paths():
    return {key: _norm_path(value) for key, value in _section("paths").items()}


def get_sort_log():
    return bool(load_parameters().get("sort_log", True))


def get_preprocessing_params():
    cfg = _section("preprocessing")
    return {
        "HSV_LO": np.array(_cget(cfg, "HSV_LO", [0, 0, 0])),
        "HSV_HI": np.array(_cget(cfg, "HSV_HI", [0, 0, 0])),
        "CNT_MINA": _cget(cfg, "CNT_MINA", 0),
        "WARP_SZ": tuple(_cget(cfg, "WARP_SZ", [0, 0])),
        "TGT_W": _cget(cfg, "TGT_W", 0),
        "TGT_H": _cget(cfg, "TGT_H", 0),
    }


def get_geometry_params():
    cfg = _section("geometry")
    return {
        "EPS_FACT": _cget(cfg, "EPS_FACT", 0.0),
        "HOLE_MIN": _cget(cfg, "HOLE_MIN", 0),
        "WIND_MIN": _cget(cfg, "WIND_MIN", 0),
        "CTR_MAXA": _cget(cfg, "CTR_MAXA", 0),
        "FRAG_MIN": _cget(cfg, "FRAG_MIN", 0),
    }


def get_spot_params():
    cfg = _section("spot")
    return {
        "ERO_KN": tuple(_cget(cfg, "ERO_KN", [0, 0])),
        "ERO_ITER": _cget(cfg, "ERO_ITER", 0),
        "BKH_KN": tuple(_cget(cfg, "BKH_KN", [0, 0])),
        "BKH_CON": _cget(cfg, "BKH_CON", 0),
        "NOI_KN": tuple(_cget(cfg, "NOI_KN", [0, 0])),
        "SPT_MIN": _cget(cfg, "SPT_MIN", 0),
        "SPT_RAT": _cget(cfg, "SPT_RAT", 0.0),
        "FERO_ITR": _cget(cfg, "FERO_ITR", 0),
        "INER_ITR": _cget(cfg, "INER_ITR", 0),
        "INSP_RAT": _cget(cfg, "INSP_RAT", 0.0),
        "FSPT_RAT": _cget(cfg, "FSPT_RAT", 0.0),
        "SPT_FIN": _cget(cfg, "SPT_FIN", 0),
        "DRK_PCT": _cget(cfg, "DRK_PCT", 0),
    }


def get_classifier_rules():
    return dict(load_parameters().get("classifier_rules", {}))


def get_label_priorities():
    return dict(load_classification_settings().get("label_priorities", {}))


def get_label_class_map():
    return dict(load_classification_settings().get("label_class_map", {}))


def get_label_rules():
    return list(load_classification_settings().get("label_rules", []))

PATHS = get_paths()
RAW_DIR = PATHS["RAW_DIR"]
OUT_DIR = PATHS["OUT_DIR"]
PROC_DIR = PATHS["PROC_DIR"]
PIPELINE_CSV = PATHS["PIPELINE_CSV"]
SORT_DIR = PATHS["SORT_DIR"]
FAIL_DIR = PATHS["FAIL_DIR"]
ANNO_FILE = PATHS["ANNO_FILE"]

PREPROCESSING_PARAMS = get_preprocessing_params()
GEOMETRY_PARAMS = get_geometry_params()
SPOT_PARAMS = get_spot_params()
CLASSIFIER_RULES = get_classifier_rules()
SORT_LOG = get_sort_log()
LABEL_PRIORITIES = get_label_priorities()
LABEL_CLASS_MAP = get_label_class_map()


def run_full_pipeline():
    from classification import classify_from_csv
    from image_processing import process_images_to_csv
    from segmentation import process_directory
    from sorting import sort_images_from_csv
    from validation import load_annotations, validate_predictions

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(RAW_DIR):
        print(f"Fehler: Quellordner '{RAW_DIR}' nicht gefunden.")
        return

    process_directory(RAW_DIR, PROC_DIR, PREPROCESSING_PARAMS)
    process_images_to_csv(PROC_DIR, PIPELINE_CSV, GEOMETRY_PARAMS, SPOT_PARAMS)
    predictions = classify_from_csv(PIPELINE_CSV, CLASSIFIER_RULES, SORT_LOG)
    sort_images_from_csv(PIPELINE_CSV, SORT_DIR, SORT_LOG)

    annotations = load_annotations(ANNO_FILE, LABEL_PRIORITIES, LABEL_CLASS_MAP)
    validate_predictions(
        predictions,
        annotations,
        FAIL_DIR,
        LABEL_PRIORITIES,
        LABEL_CLASS_MAP,
    )


if __name__ == "__main__":
    run_full_pipeline()
