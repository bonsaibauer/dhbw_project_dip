"""Rule-based classifier using a precomputed decision tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from ..models import tree_params


@dataclass(frozen=True)
class TreeModel:
    feature_names: List[str]
    classes: List[str]
    children_left: List[int]
    children_right: List[int]
    feature_index: List[int]
    threshold: List[float]
    value: List[List[float]]


class RuleBasedClassifier:
    """Traverses the frozen decision tree to assign classes."""

    def __init__(self) -> None:
        self.model = TreeModel(
            feature_names=list(tree_params.FEATURE_NAMES),
            classes=list(tree_params.CLASSES),
            children_left=list(tree_params.CHILDREN_LEFT),
            children_right=list(tree_params.CHILDREN_RIGHT),
            feature_index=list(tree_params.FEATURE_INDEX),
            threshold=list(tree_params.THRESHOLD),
            value=list(tree_params.VALUE),
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        data = features[self.model.feature_names].to_numpy()
        predictions: List[str] = []
        for row in data:
            predictions.append(self._predict_row(row))
        return np.array(predictions, dtype=str)

    def _predict_row(self, row: np.ndarray) -> str:
        node = 0
        while True:
            feature_idx = self.model.feature_index[node]
            if feature_idx == -2:
                votes = self.model.value[node]
                class_idx = int(np.argmax(votes))
                return self.model.classes[class_idx]
            threshold = self.model.threshold[node]
            node = (
                self.model.children_left[node]
                if row[feature_idx] <= threshold
                else self.model.children_right[node]
            )
            if node == -1:
                raise RuntimeError("Decision tree traversal reached an invalid node.")
