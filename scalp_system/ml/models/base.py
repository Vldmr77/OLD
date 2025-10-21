"""Common interfaces for ML models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class ModelPrediction:
    score: float
    confidence: float


class FeatureModel(ABC):
    @abstractmethod
    def predict(self, batch: Iterable[Sequence[float]]) -> list[ModelPrediction]:
        raise NotImplementedError


class WeightedEnsemble:
    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = weights

    def combine(self, predictions: dict[str, ModelPrediction]) -> ModelPrediction:
        score = 0.0
        confidence = 0.0
        weight_sum = 0.0
        for name, pred in predictions.items():
            weight = self._weights.get(name, 0.0)
            score += weight * pred.score
            confidence += weight * pred.confidence
            weight_sum += weight
        if weight_sum == 0:
            raise ValueError("No weights provided for ensemble")
        return ModelPrediction(score=score / weight_sum, confidence=confidence / weight_sum)


__all__ = ["FeatureModel", "ModelPrediction", "WeightedEnsemble"]
