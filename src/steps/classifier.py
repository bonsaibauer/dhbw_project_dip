"""Classification step built on top of handcrafted features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@dataclass
class TrainingArtifacts:
    """Encapsulates the trained model and feature statistics."""

    model: RandomForestClassifier
    feature_names: Sequence[str]
    feature_importances: Sequence[float]


class QualityClassifier:
    """Wrapper around a RandomForest classifier."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=600,
            max_depth=12,
            random_state=random_state,
        )
        self.feature_names: Sequence[str] | None = None

    def fit(self, features: pd.DataFrame, labels: Sequence[str]) -> TrainingArtifacts:
        self.feature_names = list(features.columns)
        self.model.fit(features.values, labels)
        return TrainingArtifacts(
            model=self.model,
            feature_names=self.feature_names,
            feature_importances=self.model.feature_importances_,
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.feature_names is None:
            raise RuntimeError("Classifier must be fitted before calling predict.")
        return self.model.predict(features[self.feature_names].values)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.feature_names is None:
            raise RuntimeError("Classifier must be fitted before calling predict_proba.")
        return self.model.predict_proba(features[self.feature_names].values)
