"""Common interfaces for ML models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


@dataclass
class ModelPrediction:
    score: float
    confidence: float


class FeatureModel(ABC):
    def __init__(self) -> None:
        self._model_path: Optional[Path] = None
        self._device: str = "gpu"

    @abstractmethod
    def predict(self, batch: Iterable[Sequence[float]]) -> list[ModelPrediction]:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        """Load model weights from the provided path."""

        resolved = path if isinstance(path, Path) else Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Model file not found: {resolved}")
        self._model_path = resolved

    def reset(self) -> None:
        """Reset transient interpreter state after reloading weights."""

        # Default implementation is stateless.
        return None

    def set_device(self, device: str) -> None:
        self._device = device

    @property
    def device(self) -> str:
        return self._device


class WeightedEnsemble:
    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = dict(weights)

    def set_weights(self, weights: dict[str, float]) -> None:
        self._weights = dict(weights)

    def combine(
        self, predictions: dict[str, ModelPrediction], weights: Optional[dict[str, float]] = None
    ) -> ModelPrediction:
        active_weights = weights or self._weights
        score = 0.0
        confidence = 0.0
        weight_sum = 0.0
        for name, pred in predictions.items():
            weight = active_weights.get(name, 0.0)
            score += weight * pred.score
            confidence += weight * pred.confidence
            weight_sum += weight
        if weight_sum == 0:
            raise ValueError("No weights provided for ensemble")
        return ModelPrediction(score=score / weight_sum, confidence=confidence / weight_sum)


__all__ = ["FeatureModel", "ModelPrediction", "WeightedEnsemble"]
