from __future__ import annotations

"""Common base classes for ML inference models."""

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


@dataclass
class ModelPrediction:
    score: float
    confidence: float


class FeatureModel(ABC):
    """Abstract interface for a predictive model used by the ensemble."""

    def __init__(self) -> None:
        self._model_path: Optional[Path] = None
        self._device: str = "gpu"

    @abstractmethod
    def predict(self, batch: Iterable[Sequence[float]]) -> list[ModelPrediction]:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        """Remember the resolved path for subclasses that rely on local files."""

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


class LinearFeatureModel(FeatureModel):
    """Simple dense model driven by JSON encoded weight vectors."""

    def __init__(self, *, activation: str = "tanh", clip: float = 1.0) -> None:
        super().__init__()
        self._weights: list[float] = []
        self._bias: float = 0.0
        self._scale: float = 1.0
        self._activation = activation
        self._clip = max(0.0, float(clip))

    def load(self, path: Path) -> None:
        super().load(path)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = payload.get("weights")
        if not isinstance(weights, list) or not weights:
            raise ValueError("Model payload missing weights")
        self._weights = [float(value) for value in weights]
        self._bias = float(payload.get("bias", 0.0))
        self._scale = max(float(payload.get("scale", 1.0)), 1e-6)
        if "activation" in payload:
            self._activation = str(payload["activation"]).lower()
        self._clip = max(float(payload.get("clip", self._clip)), 0.0)
        self.reset()

    def reset(self) -> None:
        return None

    def predict(self, batch: Iterable[Sequence[float]]) -> list[ModelPrediction]:
        predictions: list[ModelPrediction] = []
        for features in batch:
            if not features:
                predictions.append(ModelPrediction(score=0.0, confidence=0.0))
                continue
            score = self._activate(self._project(features))
            confidence = self._confidence(score)
            predictions.append(ModelPrediction(score=score, confidence=confidence))
        return predictions

    def _project(self, features: Sequence[float]) -> float:
        limit = min(len(features), len(self._weights))
        weighted = sum(self._weights[idx] * float(features[idx]) for idx in range(limit))
        return weighted + self._bias

    def _activate(self, value: float) -> float:
        activation = self._activation
        if activation == "tanh":
            score = math.tanh(value)
        elif activation == "relu":
            score = max(0.0, value)
        elif activation == "gelu":
            score = 0.5 * value * (1.0 + math.tanh(math.sqrt(2 / math.pi) * (value + 0.044715 * value**3)))
        elif activation == "linear":
            score = value
        else:
            score = math.tanh(value)
        if self._clip:
            score = max(-self._clip, min(self._clip, score))
        return score

    def _confidence(self, score: float) -> float:
        magnitude = abs(score)
        logistic = 1.0 / (1.0 + math.exp(-magnitude * self._scale))
        return max(0.0, min(1.0, logistic))


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
