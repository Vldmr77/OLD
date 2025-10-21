from __future__ import annotations

"""Stateful approximation of an LSTM-based order book model."""

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from .base import LinearFeatureModel, ModelPrediction


class LSTMOrderBookModel(LinearFeatureModel):
    """Applies an exponentially-smoothed projection over feature windows."""

    def __init__(self) -> None:
        super().__init__(activation="tanh", clip=1.0)
        self._momentum: float = 0.6
        self._hidden_state: float = 0.0

    def load(self, path: Path) -> None:
        super().load(path)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self._momentum = float(payload.get("momentum", 0.6))
        self.reset()

    def reset(self) -> None:
        self._hidden_state = 0.0

    def predict(self, batch: Iterable[Sequence[float]]) -> list[ModelPrediction]:
        predictions: list[ModelPrediction] = []
        for features in batch:
            if not features:
                predictions.append(ModelPrediction(score=0.0, confidence=0.0))
                continue
            projection = self._project(features)
            self._hidden_state = (
                self._momentum * self._hidden_state + (1.0 - self._momentum) * projection
            )
            score = math.tanh(self._hidden_state)
            confidence = self._confidence(score)
            predictions.append(ModelPrediction(score=score, confidence=confidence))
        return predictions


__all__ = ["LSTMOrderBookModel"]
