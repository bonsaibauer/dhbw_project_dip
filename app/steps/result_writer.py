"""Handles exporting classified and cropped images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from .. import config
from ..utils import log_info
from .background_segmentation import SegmentationResult


@dataclass
class SavedItem:
    image_path: Path
    is_correct: bool
    predicted_label: str
    target_label: str
    misclassified_path: Path | None = None


class ResultWriter:
    """Writes classification artefacts to disk."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or config.RUN_IMAGE_DIR
        self.misclassified_dir = self.base_dir / "Falsch"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for target in config.TARGET_CLASSES:
            (self.base_dir / target).mkdir(parents=True, exist_ok=True)
        self.misclassified_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        filename: str,
        prediction: str,
        target: str,
        segmentation: SegmentationResult,
        symmetry: float,
    ) -> SavedItem:
        """Stores the cropped foreground in the predicted class directory."""

        predicted_dir = self.base_dir / prediction
        prefixed_name = self._prefixed_name(filename, prediction, symmetry)
        out_path = predicted_dir / prefixed_name
        cv2.imwrite(str(out_path), segmentation.cropped_image)
        log_info(
            f"Saved {filename} -> {out_path.name} (pred={prediction}, target={target})",
        )
        is_correct = prediction == target
        mismatch_path: Path | None = None
        if not is_correct:
            mismatch_name = f"{Path(filename).stem}__gt-{target}_pred-{prediction}{Path(filename).suffix}"
            mismatch_path = self.misclassified_dir / mismatch_name
            cv2.imwrite(str(mismatch_path), segmentation.cropped_image)
            log_info(f"Misclassification stored as {mismatch_name} in Falsch.")
        return SavedItem(
            image_path=out_path,
            is_correct=is_correct,
            predicted_label=prediction,
            target_label=target,
            misclassified_path=mismatch_path,
        )

    @staticmethod
    def _prefixed_name(filename: str, prediction: str, symmetry: float) -> str:
        path = Path(filename)
        if prediction != "Normal":
            return path.name
        scaled = max(0, min(999, int(round(symmetry * 1000))))
        prefix = f"SYM{scaled:03d}"
        return f"{prefix}_{path.name}"
