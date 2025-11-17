import os
import runpy

base_dir = os.path.dirname(__file__)
stage_list = [
    ("01_segmentation.py", "segment_cli"),
    ("02_image_processing.py", "process_cli"),
    ("03_classification.py", "classify_cli"),
    ("04_sorting.py", "sort_cli"),
    ("05_validation.py", "validate_cli"),
]

def run_stage(script_name, entry_point):
    """L�dt das Feature-Skript und f�hrt die gew�nschte CLI-Funktion aus."""
    module = runpy.run_path(os.path.join(base_dir, script_name))
    func = module.get(entry_point)
    if not callable(func):
        raise RuntimeError(f"Funktion '{entry_point}' nicht in {script_name} gefunden.")
    func()

def run_pipeline():
    for script_name, entry_point in stage_list:
        run_stage(script_name, entry_point)

if __name__ == "__main__":
    run_pipeline()
