from __future__ import annotations

"""Approximation of a temporal transformer over aggregated windows."""

import json
from pathlib import Path
from typing import Iterable, Sequence

from .base import LinearFeatureModel, ModelPrediction


class TemporalTransformerModel(LinearFeatureModel):
    """Applies GELU activations to pooled temporal projections."""

    def __init__(self) -> None:
        super().__init__(activation="gelu", clip=1.2)
        self._window_sizes: list[int] = [10, 20, 40]

    def load(self, path: Path) -> None:
        super().load(path)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        sizes = payload.get("window_sizes")
        if isinstance(sizes, list) and sizes:
            self._window_sizes = [max(1, int(value)) for value in sizes]

    def predict(self, batch: Iterable[Sequence[float]]) -> list[ModelPrediction]:
        predictions: list[ModelPrediction] = []
        for features in batch:
            if not features:
                predictions.append(ModelPrediction(score=0.0, confidence=0.0))
                continue
            pooled = 0.0
            for window in self._window_sizes:
                pooled += self._activate(self._project(features[:window]))
            pooled /= len(self._window_sizes)
            score = max(-1.0, min(1.0, pooled))
            confidence = self._confidence(score)
            predictions.append(ModelPrediction(score=score, confidence=confidence))
        return predictions


__all__ = ["TemporalTransformerModel"]
