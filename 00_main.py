import os
import runpy

BASE_DIR = os.path.dirname(__file__)
FEATURES = [
    ("01_segmentation.py", "run_segmentation_cli"),
    ("02_image_processing.py", "run_image_processing_cli"),
    ("03_classification.py", "run_classification_cli"),
    ("04_sorting.py", "run_sorting_cli"),
    ("05_validation.py", "run_validation_cli"),
]


def _run_feature(script_name, entry_point):
    """Lädt das Feature-Skript und führt die gewünschte CLI-Funktion aus."""
    module = runpy.run_path(os.path.join(BASE_DIR, script_name))
    func = module.get(entry_point)
    if not callable(func):
        raise RuntimeError(f"Funktion '{entry_point}' nicht in {script_name} gefunden.")
    func()


def run_complete_inspection_pipeline():
    for script_name, entry_point in FEATURES:
        _run_feature(script_name, entry_point)


if __name__ == "__main__":
    run_complete_inspection_pipeline()
