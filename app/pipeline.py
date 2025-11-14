"""End-to-end pipeline orchestrating the quality inspection workflow."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Dict, List, Any

import cv2
import pandas as pd

from . import config
from .models.records import ProcessRecord
from .steps import dataset_loader
from .steps.background_segmentation import BackgroundSegmenter, SegmentationResult
from .steps.classifier import RuleBasedClassifier
from .steps.evaluator import evaluate
from .steps.feature_extraction import FEATURE_NAMES, extract_features
from .steps.result_writer import ResultWriter
from .utils import log_info, log_progress


def run_pipeline() -> Dict[str, object]:
    """Executes the full workflow and returns a dictionary with artefacts."""

    _prepare_output_dirs()

    log_info("Starting fryum quality pipeline.", progress=0.0)
    records = dataset_loader.load_dataset()
    record_tags = {
        record.relative_path.as_posix(): _extract_tags(record.original_label)
        for record in records
    }
    log_info(f"Loaded {len(records)} annotated records.", progress=5.0)
    segmenter = BackgroundSegmenter()
    writer = ResultWriter()
    classifier = RuleBasedClassifier()

    feature_rows: List[Dict[str, Any]] = []
    segmentation_cache: Dict[str, SegmentationResult] = {}
    record_ids: List[str] = []
    filenames: List[str] = []
    ground_truth: List[str] = []
    original_paths: Dict[str, Path] = {}

    log_info("Running segmentation and feature extraction.", progress=10.0)
    total_records = max(1, len(records))
    for idx, record in enumerate(records, start=1):
        record_id = record.relative_path.as_posix()
        image = cv2.imread(str(record.image_path))
        segmentation = segmenter.segment(image)
        features = extract_features(image, segmentation)
        features["target"] = record.target_label
        features["filename"] = record.image_path.name
        features["record_id"] = record_id
        feature_rows.append(features)
        segmentation_cache[record_id] = segmentation
        record_ids.append(record_id)
        filenames.append(record.image_path.name)
        ground_truth.append(record.target_label)
        original_paths[record_id] = record.image_path
        log_progress("Segmentation+Features", idx, total_records)

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(config.REPORT_DIR / "feature_table.csv", index=False)
    feature_df.set_index("record_id", inplace=True)
    log_info("Segmentation and feature extraction completed.", progress=60.0)

    log_info("Classifying samples via rule-based model.", progress=70.0)
    predictions = classifier.predict(feature_df[list(FEATURE_NAMES)])
    log_info("Classification completed.", progress=78.0)

    log_info("Evaluating predictions.", progress=82.0)
    evaluation = evaluate(feature_df["target"].values, predictions, config.TARGET_CLASSES)
    (config.REPORT_DIR / "confusion_matrix.csv").write_text(evaluation.confusion.to_csv())
    (config.REPORT_DIR / "classification_report.txt").write_text(evaluation.report)
    (config.REPORT_DIR / "summary.txt").write_text(f"Accuracy: {evaluation.accuracy:.4f}\n")
    accuracy_lines = [f"Overall accuracy: {evaluation.accuracy:.4f}", "Per-class precision/recall/F1 support report:"]
    accuracy_lines.extend(evaluation.report.strip().splitlines())
    accuracy_lines.append("Confusion matrix (rows=true, cols=pred):")
    accuracy_lines.extend(evaluation.confusion.to_string().splitlines())
    log_info("Evaluation complete.", progress=90.0)

    log_info("Saving classified crops.", progress=92.0)
    total_files = max(1, len(filenames))
    saved_items = []
    process_records: List[ProcessRecord] = []
    metric_keys = ["area_ratio", "symmetry", "lightness_mean", "laplacian_std", "dark_fraction", "bright_fraction"]
    for idx, (record_id, name, target, prediction) in enumerate(zip(record_ids, filenames, ground_truth, predictions), start=1):
        segmentation = segmentation_cache[record_id]
        symmetry = float(_scalar_df(feature_df, record_id, "symmetry"))
        inspection_paths = _save_inspection_assets(name, prediction, segmentation, original_paths[record_id])
        saved = writer.save(name, prediction, target, segmentation, symmetry)
        inspection_paths["classified"] = str(saved.image_path)
        metrics = {key: float(_scalar_df(feature_df, record_id, key)) for key in metric_keys}
        process_steps = _build_process_steps(segmentation, prediction, target, str(saved.image_path))
        saved_items.append(saved)
        process_records.append(
            ProcessRecord(
                filename=name,
                record_id=record_id,
                prediction=prediction,
                target=target,
                original_path=str(original_paths[record_id]),
                classified_path=str(saved.image_path),
                inspection_paths=inspection_paths,
                process_steps=process_steps,
                tags=record_tags.get(record_id, []),
                metrics=metrics,
            )
        )
        log_progress("Saving outputs", idx, total_files)

    log_info("Accuracy summary:", progress=99.5)
    for line in accuracy_lines:
        log_info(f"    {line}")
    prediction_validation = _validate_predictions(process_records)
    log_info(prediction_validation["message"], progress=99.7)
    for miss in prediction_validation["mismatches"]:
        log_info(f"    {miss}")
    log_info("Pipeline finished successfully.", progress=100.0)
    return {
        "features": feature_df.reset_index(),
        "predictions": predictions,
        "evaluation": evaluation,
        "saved_items": saved_items,
        "records": process_records,
    }


def _prepare_output_dirs() -> None:
    config.OUTPUT_ROOT.mkdir(exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if config.INSPECTION_DIR.exists():
        _safe_rmtree(config.INSPECTION_DIR)
    config.INSPECTION_DIR.mkdir(parents=True, exist_ok=True)


def _safe_rmtree(path: Path) -> None:
    """Remove directory trees on Windows even if files are read-only."""

    def _on_rm_error(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            raise exc_info[1]

    shutil.rmtree(path, onerror=_on_rm_error)


def _save_inspection_assets(
    filename: str,
    prediction: str,
    segmentation: SegmentationResult,
    original_path: Path,
) -> Dict[str, str]:
    dest = config.INSPECTION_DIR / prediction / Path(filename).stem
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    original_copy = dest / "original.jpg"
    blurred_path = dest / "blurred.jpg"
    raw_mask_path = dest / "raw_mask.png"
    mask_path = dest / "mask.png"
    cropped_path = dest / "cropped.png"
    overlay_path = dest / "mask_overlay.png"

    cv2.imwrite(str(raw_mask_path), segmentation.raw_mask)
    cv2.imwrite(str(mask_path), segmentation.mask)
    cv2.imwrite(str(cropped_path), segmentation.cropped_image)

    original_img = cv2.imread(str(original_path))
    masked_original = None
    if original_img is not None and original_img.shape[:2] == segmentation.mask.shape:
        masked_original = cv2.bitwise_and(original_img, original_img, mask=segmentation.mask)
        cv2.imwrite(str(original_copy), masked_original)
        masked_blurred = cv2.bitwise_and(segmentation.blurred, segmentation.blurred, mask=segmentation.mask)
        cv2.imwrite(str(blurred_path), masked_blurred)
        mask_color = cv2.cvtColor(segmentation.mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(masked_original, 0.7, mask_color, 0.3, 0)
        cv2.imwrite(str(overlay_path), overlay)
    else:
        shutil.copy2(original_path, original_copy)
        cv2.imwrite(str(blurred_path), segmentation.blurred)
        overlay_path = str(mask_path)
    return {
        "folder": str(dest),
        "original": str(original_copy),
        "blurred": str(blurred_path),
        "raw_mask": str(raw_mask_path),
        "mask": str(mask_path),
        "cropped": str(cropped_path),
        "mask_overlay": str(overlay_path),
    }


def _build_process_steps(
    segmentation: SegmentationResult,
    prediction: str,
    target: str,
    output_path: str,
) -> List[Dict[str, str]]:
    return [
        {
            "title": "Hintergrundsegmentierung",
            "status": "abgeschlossen",
            "details": f"Flaechenanteil {segmentation.area_ratio * 100:.2f} %",
        },
        {
            "title": "Feature-Extraktion",
            "status": "abgeschlossen",
            "details": f"{len(FEATURE_NAMES)} Merkmale berechnet",
        },
        {
            "title": "Klassifikation",
            "status": "abgeschlossen",
            "details": f"Ergebnis: {prediction} (Soll: {target})",
        },
        {
            "title": "Export",
            "status": "abgeschlossen",
            "details": output_path,
        },
    ]


def _extract_tags(label: str) -> List[str]:
    """Split the raw CSV label into normalized tag strings."""

    raw = (label or "").strip()
    if not raw or raw.lower() == "nan":
        return []
    return [token.strip().lower() for token in raw.split(",") if token.strip()]


def _validate_predictions(records: List[ProcessRecord]) -> Dict[str, object]:
    """Checks model predictions against CSV targets."""

    total = len(records)
    correct = 0
    mismatches: List[str] = []
    for rec in records:
        if rec.prediction == rec.target:
            correct += 1
        else:
            mismatches.append(
                f"{rec.filename}: erwartet {rec.target}, erhalten {rec.prediction}"
            )
    accuracy = (correct / total) if total else 0.0
    message = (
        f"Vorhersage-Validierung: {correct}/{total} korrekt "
        f"({accuracy:.2%})."
    )
    return {"message": message, "mismatches": mismatches}


def _scalar_df(df: pd.DataFrame, index: str, column: str) -> Any:
    """Return a scalar cell value even if duplicated index produces a Series."""

    value = df.loc[index, column]
    if isinstance(value, pd.Series):
        return value.iloc[0]
    return value


if __name__ == "__main__":
    results = run_pipeline()
    print(f"Accuracy: {results['evaluation'].accuracy:.4f}")
