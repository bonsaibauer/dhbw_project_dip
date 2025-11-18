import os
import runpy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.join(BASE_DIR, "scripts")
STAGE_LIST = [
    ("01_segmentation.py", "segment_cli"),
    ("02_image_processing.py", "process_cli"),
    ("03_classification.py", "classify_cli"),
    ("04_sorting.py", "sort_cli"),
    ("05_validation.py", "validate_cli"),
]


def run_stage(script_name, entry_point):
    """Laedt das Feature-Skript und fuehrt die gewuenschte CLI-Funktion aus."""
    module = runpy.run_path(os.path.join(SCRIPT_DIR, script_name))
    func = module.get(entry_point)
    if not callable(func):
        raise RuntimeError(
            f"Funktion '{entry_point}' nicht in {script_name} gefunden."
        )
    func()


def run_pipeline():
    for script_name, entry_point in STAGE_LIST:
        run_stage(script_name, entry_point)


if __name__ == "__main__":
    run_pipeline()
