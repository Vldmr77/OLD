from __future__ import annotations

"""Gradient boosted feature ensemble."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .base import FeatureModel, ModelPrediction


@dataclass(frozen=True)
class _BoostingStage:
    weights: List[float]
    bias: float
    shrinkage: float


class GradientBoostedFeatureModel(FeatureModel):
    """Lightweight gradient boosting approximation over dense features."""

    def __init__(self) -> None:
        super().__init__()
        self._stages: list[_BoostingStage] = []
        self._scale: float = 1.0

    def load(self, path: Path) -> None:
        super().load(path)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stage_payloads = payload.get("stages")
        if not isinstance(stage_payloads, list) or not stage_payloads:
            raise ValueError("Gradient boosted model requires staged weights")
        stages: list[_BoostingStage] = []
        for stage in stage_payloads:
            weights = stage.get("weights")
            if not isinstance(weights, list) or not weights:
                raise ValueError("Boosting stage missing weights")
            bias = float(stage.get("bias", 0.0))
            shrinkage = float(stage.get("shrinkage", 1.0))
            stages.append(
                _BoostingStage(weights=[float(value) for value in weights], bias=bias, shrinkage=shrinkage)
            )
        self._stages = stages
        self._scale = max(float(payload.get("scale", 1.0)), 1e-6)

    def reset(self) -> None:
        # No state to clear for the boosting surrogate.
        return None

    def predict(self, batch: Iterable[Sequence[float]]) -> list[ModelPrediction]:
        predictions: list[ModelPrediction] = []
        for features in batch:
            if not features:
                predictions.append(ModelPrediction(score=0.0, confidence=0.0))
                continue
            score = 0.0
            for stage in self._stages:
                limit = min(len(features), len(stage.weights))
                contribution = sum(
                    stage.weights[idx] * float(features[idx]) for idx in range(limit)
                ) + stage.bias
                score += stage.shrinkage * math.tanh(contribution)
            score = max(-1.0, min(1.0, score))
            confidence = 1.0 / (1.0 + math.exp(-abs(score) * self._scale))
            predictions.append(ModelPrediction(score=score, confidence=confidence))
        return predictions


__all__ = ["GradientBoostedFeatureModel"]
