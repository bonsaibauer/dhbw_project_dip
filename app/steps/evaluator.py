"""Evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass
class EvaluationResult:
    accuracy: float
    report: str
    confusion: pd.DataFrame


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...]) -> EvaluationResult:
    """Computes accuracy, classification report and confusion matrix."""

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels)
    conf = confusion_matrix(y_true, y_pred, labels=labels)
    confusion_df = pd.DataFrame(conf, index=labels, columns=labels)
    return EvaluationResult(accuracy=accuracy, report=report, confusion=confusion_df)
