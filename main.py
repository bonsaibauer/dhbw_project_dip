import os

from classification import classify_from_csv
from image_processing import process_images_to_csv
from segmentation import process_directory
from settings import (
    get_classifier_rules,
    get_geometry_params,
    get_label_class_map,
    get_label_priorities,
    get_paths,
    get_preprocessing_params,
    get_sort_log,
    get_spot_params,
)
from sorting import sort_images_from_csv
from validation import load_annotations, validate_predictions

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
