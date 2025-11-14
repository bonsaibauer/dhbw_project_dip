"""End-to-end pipeline orchestrating the quality inspection workflow."""

from __future__ import annotations

from typing import Dict, List

import cv2
import pandas as pd

from . import config
from .steps import dataset_loader
from .steps.background_segmentation import BackgroundSegmenter, SegmentationResult
from .steps.classifier import RuleBasedClassifier
from .steps.evaluator import evaluate
from .steps.feature_extraction import FEATURE_NAMES, extract_features
from .steps.result_writer import ResultWriter
from .utils import log_info, log_progress


def run_pipeline() -> Dict[str, object]:
    """Executes the full workflow and returns a dictionary with artefacts."""

    config.OUTPUT_ROOT.mkdir(exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log_info("Starting fryum quality pipeline.", progress=0.0)
    records = dataset_loader.load_dataset()
    log_info(f"Loaded {len(records)} annotated records.", progress=5.0)
    segmenter = BackgroundSegmenter()
    writer = ResultWriter()
    classifier = RuleBasedClassifier()

    feature_rows: List[Dict[str, float]] = []
    segmentation_cache: Dict[str, SegmentationResult] = {}
    filenames: List[str] = []
    ground_truth: List[str] = []

    log_info("Running segmentation and feature extraction.", progress=10.0)
    total_records = max(1, len(records))
    for idx, record in enumerate(records, start=1):
        image = cv2.imread(str(record.image_path))
        segmentation = segmenter.segment(image)
        features = extract_features(image, segmentation)
        features["target"] = record.target_label
        features["filename"] = record.image_path.name
        feature_rows.append(features)
        segmentation_cache[record.image_path.name] = segmentation
        filenames.append(record.image_path.name)
        ground_truth.append(record.target_label)
        log_progress("Segmentation+Features", idx, total_records)

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(config.REPORT_DIR / "feature_table.csv", index=False)
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
    for idx, (name, target, prediction) in enumerate(zip(filenames, ground_truth, predictions), start=1):
        segmentation = segmentation_cache[name]
        symmetry = feature_df.loc[feature_df["filename"] == name, "symmetry"].iloc[0]
        saved = writer.save(name, prediction, target, segmentation, symmetry)
        saved_items.append(saved)
        log_progress("Saving outputs", idx, total_files)

    log_info("Accuracy summary:", progress=99.5)
    for line in accuracy_lines:
        log_info(f"    {line}")
    log_info("Pipeline finished successfully.", progress=100.0)
    return {
        "features": feature_df,
        "predictions": predictions,
        "evaluation": evaluation,
        "saved_items": saved_items,
    }


if __name__ == "__main__":
    results = run_pipeline()
    print(f"Accuracy: {results['evaluation'].accuracy:.4f}")
