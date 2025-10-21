"""Temporal transformer placeholder."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from .base import FeatureModel, ModelPrediction


class TemporalTransformerModel(FeatureModel):
    def predict(self, batch: Iterable[Sequence[float]]) -> list[ModelPrediction]:
        predictions: list[ModelPrediction] = []
        for features in batch:
            if not features:
                predictions.append(ModelPrediction(score=0.0, confidence=0.0))
                continue
            mean_value = sum(features) / len(features)
            score = max(-1.0, min(1.0, math.sin(mean_value)))
            avg_abs = sum(abs(value) for value in features) / len(features)
            confidence = max(0.0, min(1.0, avg_abs / 10))
            predictions.append(ModelPrediction(score=score, confidence=confidence))
        return predictions


__all__ = ["TemporalTransformerModel"]
