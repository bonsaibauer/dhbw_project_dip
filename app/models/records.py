"""Data structures for storing processed image information."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ProcessRecord:
    filename: str
    record_id: str
    prediction: str
    target: str
    original_path: str
    classified_path: str
    inspection_paths: Dict[str, str]
    process_steps: List[Dict[str, str]]
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    misclassified_path: str | None = None
