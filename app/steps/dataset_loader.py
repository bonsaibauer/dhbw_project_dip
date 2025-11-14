"""Dataset loading step."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .. import config


@dataclass(frozen=True)
class ImageRecord:
    """Holds metadata for a single annotated image."""

    relative_path: Path
    image_path: Path
    original_label: str
    target_label: str


def load_dataset() -> List[ImageRecord]:
    """Loads the annotation CSV and resolves the file paths."""

    annotations = pd.read_csv(config.ANNOTATION_FILE)
    records: List[ImageRecord] = []
    for _, row in annotations.iterrows():
        rel_path = Path(row["image"])
        local_name = rel_path.name
        sub_dir = config.NORMAL_DIR if "Normal" in row["image"] else config.ANOMALY_DIR
        image_path = sub_dir / local_name
        original_label = str(row["label"]).strip()
        try:
            target_label = config.AGGREGATED_CLASS_MAPPING[original_label]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Label '{original_label}' is not mapped to a target class.") from exc
        records.append(
            ImageRecord(
                relative_path=rel_path,
                image_path=image_path,
                original_label=original_label,
                target_label=target_label,
            )
        )
    return records


def iter_image_paths(records: Iterable[ImageRecord]) -> Iterable[Path]:
    """Convenience generator returning image paths for downstream steps."""

    for record in records:
        yield record.image_path
