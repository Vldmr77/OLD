"""LSTM model placeholder for order book sequences."""
from __future__ import annotations

from typing import Iterable, Sequence

from .base import FeatureModel, ModelPrediction


class LSTMOrderBookModel(FeatureModel):
    def __init__(self, *, hidden_size: int = 64) -> None:
        self.hidden_size = hidden_size

    def predict(self, batch: Iterable[Sequence[float]]) -> list[ModelPrediction]:
        predictions = []
        for features in batch:
            if not features:
                predictions.append(ModelPrediction(score=0.0, confidence=0.0))
                continue
            mean_value = sum(features) / len(features)
            score = max(-1.0, min(1.0, mean_value / 10))
            norm = sum(value * value for value in features) ** 0.5
            confidence = max(0.0, min(1.0, norm / 100))
            predictions.append(ModelPrediction(score=score, confidence=confidence))
        return predictions


__all__ = ["LSTMOrderBookModel"]
