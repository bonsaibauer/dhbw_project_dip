"""Central configuration for the fryum quality inspection pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
_PATHS_FILE = BASE_DIR / "paths.json"
_PATH_FIELD_DEFS: Tuple[Tuple[str, str, str], ...] = (
    ("data_dir", "Datenordner", "dir"),
    ("images_dir", "Bilder (Images)", "dir"),
    ("normal_dir", "Normal-Bilder", "dir"),
    ("anomaly_dir", "Anomaly-Bilder", "dir"),
    ("mask_dir", "Maskenordner", "dir"),
    ("annotation_file", "Annotation CSV", "file"),
)


def _load_path_settings() -> Dict[str, str]:
    if not _PATHS_FILE.exists():
        return {}
    try:
        data = json.loads(_PATHS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    cleaned = {}
    for key, value in data.items():
        if key in {field[0] for field in _PATH_FIELD_DEFS} and value:
            cleaned[key] = str(value)
    return cleaned


def _resolve_paths(settings: Dict[str, str]) -> Dict[str, Path]:
    data_dir = Path(settings.get("data_dir", BASE_DIR / "data"))
    defaults = {
        "data_dir": data_dir,
        "images_dir": data_dir / "Images",
        "normal_dir": data_dir / "Images" / "Normal",
        "anomaly_dir": data_dir / "Images" / "Anomaly",
        "mask_dir": data_dir / "Masks",
        "annotation_file": data_dir / "image_anno.csv",
    }
    resolved: Dict[str, Path] = {}
    for key, default in defaults.items():
        override = settings.get(key)
        resolved[key] = Path(override) if override else Path(default)
    return resolved


_PATH_SETTINGS = _load_path_settings()
_RESOLVED_PATHS = _resolve_paths(_PATH_SETTINGS)

DATA_DIR = _RESOLVED_PATHS["data_dir"]
IMAGES_DIR = _RESOLVED_PATHS["images_dir"]
NORMAL_DIR = _RESOLVED_PATHS["normal_dir"]
ANOMALY_DIR = _RESOLVED_PATHS["anomaly_dir"]
ANNOTATION_FILE = _RESOLVED_PATHS["annotation_file"]
MASK_DIR = _RESOLVED_PATHS["mask_dir"]

OUTPUT_ROOT = BASE_DIR / "output"
RUN_IMAGE_DIR = OUTPUT_ROOT / "classified"
REPORT_DIR = OUTPUT_ROOT / "reports"
INSPECTION_DIR = OUTPUT_ROOT / "inspection"

AGGREGATED_CLASS_MAPPING = {
    "normal": "Normal",
    "burnt": "Farbfehler",
    "different colour spot": "Farbfehler",
    "different colour spot,similar colour spot": "Farbfehler",
    "similar colour spot": "Farbfehler",
    "similar colour spot,other": "Farbfehler",
    "similar colour spot,small scratches": "Farbfehler",
    "small scratches": "Farbfehler",
    "corner or edge breakage": "Bruch",
    "corner or edge breakage,small scratches": "Bruch",
    "middle breakage": "Bruch",
    "middle breakage,small scratches": "Bruch",
    "middle breakage,similar colour spot": "Bruch",
    "fryum stuck together": "Rest",
}

TARGET_CLASSES = ("Normal", "Farbfehler", "Bruch", "Rest")

SUBCATEGORY_TAGS = {
    "Normal": ("normal",),
    "Farbfehler": ("burnt", "different colour spot", "similar colour spot"),
    "Bruch": ("middle breakage", "corner or edge breakage"),
    "Rest": ("fryum stuck together", "small scratches", "other"),
}

TAG_CATEGORY_LOOKUP = {
    tag: category
    for category, tags in SUBCATEGORY_TAGS.items()
    for tag in tags
}

_ALL_TAGS = tuple(dict.fromkeys(tag for tags in SUBCATEGORY_TAGS.values() for tag in tags))
CATEGORY_TAGS = {**SUBCATEGORY_TAGS, "Alle": _ALL_TAGS}

TAG_DISPLAY_NAMES = {
    "normal": "Normal",
    "burnt": "Verbrannt",
    "different colour spot": "Farbiger Fleck",
    "similar colour spot": "Ähnlicher Farbton",
    "middle breakage": "Bruch (Mitte)",
    "corner or edge breakage": "Bruch (Rand)",
    "fryum stuck together": "Verklebt",
    "small scratches": "Kratzer",
    "other": "Sonstiges",
}


def _check_path(path: Path, kind: str) -> bool:
    if kind == "file":
        return path.is_file()
    return path.is_dir()


def reload_paths() -> None:
    """Reload path overrides from disk and update module globals."""

    global _PATH_SETTINGS, _RESOLVED_PATHS
    global DATA_DIR, IMAGES_DIR, NORMAL_DIR, ANOMALY_DIR, ANNOTATION_FILE, MASK_DIR

    _PATH_SETTINGS = _load_path_settings()
    _RESOLVED_PATHS = _resolve_paths(_PATH_SETTINGS)
    DATA_DIR = _RESOLVED_PATHS["data_dir"]
    IMAGES_DIR = _RESOLVED_PATHS["images_dir"]
    NORMAL_DIR = _RESOLVED_PATHS["normal_dir"]
    ANOMALY_DIR = _RESOLVED_PATHS["anomaly_dir"]
    ANNOTATION_FILE = _RESOLVED_PATHS["annotation_file"]
    MASK_DIR = _RESOLVED_PATHS["mask_dir"]


def get_path_fields() -> List[Dict[str, object]]:
    """Return metadata about configurable paths for UI setup."""

    fields = []
    for key, label, kind in _PATH_FIELD_DEFS:
        path = _RESOLVED_PATHS[key]
        fields.append(
            {
                "key": key,
                "label": label,
                "type": kind,
                "path": str(path),
                "exists": _check_path(path, kind),
            }
        )
    return fields


def missing_paths() -> List[Dict[str, object]]:
    """Return a list of path fields that currently do not exist."""

    return [field for field in get_path_fields() if not field["exists"]]


def save_path_settings(new_paths: Dict[str, str]) -> None:
    """Persist user-defined paths and reload the configuration."""

    valid_keys = {field[0] for field in _PATH_FIELD_DEFS}
    payload = {}
    for key, value in new_paths.items():
        if key not in valid_keys:
            continue
        value = (value or "").strip()
        if value:
            payload[key] = value
    if payload:
        _PATHS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif _PATHS_FILE.exists():
        _PATHS_FILE.unlink()
    reload_paths()


def get_path_settings() -> Dict[str, str]:
    """Return the raw override dictionary (string paths)."""

    return dict(_PATH_SETTINGS)
