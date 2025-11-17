import json
import os
from functools import lru_cache

import numpy as np

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")


def _norm_path(path_value):
    if not path_value:
        return ""
    return os.path.normpath(path_value)


@lru_cache(maxsize=1)
def load_settings():
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Konfigurationsdatei '{CONFIG_FILE}' nicht gefunden.")
    with open(CONFIG_FILE, encoding="utf-8") as cfg_file:
        return json.load(cfg_file)


def get_paths():
    data = load_settings().get("paths", {})
    return {key: _norm_path(value) for key, value in data.items()}


def get_sort_log():
    return bool(load_settings().get("sort_log", True))


def get_preprocessing_params():
    cfg = load_settings().get("preprocessing", {})
    return {
        "HSV_LO": np.array(cfg.get("HSV_LO", [0, 0, 0])),
        "HSV_HI": np.array(cfg.get("HSV_HI", [0, 0, 0])),
        "CNT_MINA": cfg.get("CNT_MINA", 0),
        "WARP_SZ": tuple(cfg.get("WARP_SZ", [0, 0])),
        "TGT_W": cfg.get("TGT_W", 0),
        "TGT_H": cfg.get("TGT_H", 0),
    }


def get_geometry_params():
    cfg = load_settings().get("geometry", {})
    return {
        "EPS_FACT": cfg.get("EPS_FACT", 0.0),
        "HOLE_MIN": cfg.get("HOLE_MIN", 0),
        "WIND_MIN": cfg.get("WIND_MIN", 0),
        "CTR_MAXA": cfg.get("CTR_MAXA", 0),
        "FRAG_MIN": cfg.get("FRAG_MIN", 0),
    }


def get_spot_params():
    cfg = load_settings().get("spot", {})
    return {
        "ERO_KN": tuple(cfg.get("ERO_KN", [0, 0])),
        "ERO_ITER": cfg.get("ERO_ITER", 0),
        "BKH_KN": tuple(cfg.get("BKH_KN", [0, 0])),
        "BKH_CON": cfg.get("BKH_CON", 0),
        "NOI_KN": tuple(cfg.get("NOI_KN", [0, 0])),
        "SPT_MIN": cfg.get("SPT_MIN", 0),
        "SPT_RAT": cfg.get("SPT_RAT", 0.0),
        "FERO_ITR": cfg.get("FERO_ITR", 0),
        "INER_ITR": cfg.get("INER_ITR", 0),
        "INSP_RAT": cfg.get("INSP_RAT", 0.0),
        "FSPT_RAT": cfg.get("FSPT_RAT", 0.0),
        "SPT_FIN": cfg.get("SPT_FIN", 0),
        "DRK_PCT": cfg.get("DRK_PCT", 0),
    }


def get_classifier_rules():
    return dict(load_settings().get("classifier_rules", {}))


def get_label_priorities():
    return dict(load_settings().get("label_priorities", {}))


def get_label_class_map():
    return dict(load_settings().get("label_class_map", {}))
