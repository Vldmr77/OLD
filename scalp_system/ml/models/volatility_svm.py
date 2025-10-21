from __future__ import annotations

"""Volatility clustering surrogate for SVM-style decisions."""

import json
from pathlib import Path
from typing import Iterable, Sequence

from .base import LinearFeatureModel, ModelPrediction


class VolatilityClusterSVM(LinearFeatureModel):
    """Applies a margin to tanh-projected volatility estimates."""

    def __init__(self) -> None:
        super().__init__(activation="tanh", clip=1.0)
        self._margin: float = 0.1

    def load(self, path: Path) -> None:
        super().load(path)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self._margin = max(0.0, float(payload.get("margin", 0.1)))

    def predict(self, batch: Iterable[Sequence[float]]) -> list[ModelPrediction]:
        predictions: list[ModelPrediction] = []
        for features in batch:
            if not features:
                predictions.append(ModelPrediction(score=0.0, confidence=0.0))
                continue
            score = self._activate(self._project(features))
            if abs(score) < self._margin:
                score = 0.0
            confidence = self._confidence(score)
            predictions.append(ModelPrediction(score=score, confidence=confidence))
        return predictions


__all__ = ["VolatilityClusterSVM"]
