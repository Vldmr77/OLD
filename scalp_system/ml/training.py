"""Offline model training utilities."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

from scalp_system.config.base import ModelWeights, TrainingConfig
from scalp_system.ml.quantization import (
    QuantizationDecision,
    build_quantization_plan,
    persist_quantization_plan,
)


@dataclass(frozen=True)
class TrainingSample:
    """Represents a single supervised learning observation."""

    figi: str
    features: list[float]
    label: float

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TrainingSample":
        features = [float(value) for value in _as_iterable(payload.get("features", []))]
        return cls(
            figi=str(payload.get("figi", "")),
            features=features,
            label=float(payload.get("label", 0.0)),
        )


@dataclass(frozen=True)
class TrainingReport:
    """Summary of a completed training job."""

    samples: int
    epochs: int
    training_loss: float
    validation_loss: float
    weights: ModelWeights
    output_path: Path
    quantization_plan: Path | None
    quantization_decisions: list[QuantizationDecision]
    model_dir: Path


class ModelTrainer:
    """Derives ensemble weights from labelled historical observations."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.config.ensure()

    def train(self) -> TrainingReport:
        samples = _load_samples(self.config.dataset_path)
        if len(samples) < self.config.min_samples:
            raise ValueError(
                f"Not enough samples for training: {len(samples)} < {self.config.min_samples}"
            )

        train_split, validation_split = _split_samples(samples, self.config.validation_split)

        train_loss = _mean_squared_error(train_split)
        validation_loss = _mean_squared_error(validation_split) if validation_split else train_loss

        weights = _derive_weights(train_split, train_loss, validation_loss)
        weights.normalise()

        output_path = self.config.output_dir / "ensemble_weights.json"
        output_path.write_text(
            json.dumps(
                {
                    "lstm_ob": weights.lstm_ob,
                    "gbdt_features": weights.gbdt_features,
                    "transformer_temporal": weights.transformer_temporal,
                    "svm_volatility": weights.svm_volatility,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        model_dir = self.config.output_dir / "models"
        model_payloads = _build_model_payloads(train_split)
        _persist_model_bundle(model_dir, model_payloads)

        quantization_decisions: list[QuantizationDecision] = []
        quantization_plan: Path | None = None
        if self.config.enable_quantization:
            quantization_decisions = build_quantization_plan(
                weights, self.config.quantization_int8_threshold
            )
            quantization_plan = persist_quantization_plan(
                quantization_decisions, self.config.output_dir
            )

        return TrainingReport(
            samples=len(samples),
            epochs=self.config.epochs,
            training_loss=train_loss,
            validation_loss=validation_loss,
            weights=weights,
            output_path=output_path,
            quantization_plan=quantization_plan,
            quantization_decisions=quantization_decisions,
            model_dir=model_dir,
        )


def _load_samples(path: Path) -> list[TrainingSample]:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    samples: list[TrainingSample] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        samples.append(TrainingSample.from_dict(payload))
    return samples


def _split_samples(
    samples: Sequence[TrainingSample], validation_ratio: float
) -> tuple[Sequence[TrainingSample], Sequence[TrainingSample]]:
    if not samples:
        return samples, []
    if validation_ratio <= 0:
        return samples, []
    validation_size = max(1, int(len(samples) * min(validation_ratio, 0.5)))
    return samples[:-validation_size] or samples, samples[-validation_size:]


def _mean_squared_error(samples: Sequence[TrainingSample]) -> float:
    if not samples:
        return 0.0
    errors: list[float] = []
    for sample in samples:
        if not sample.features:
            prediction = 0.0
        else:
            prediction = mean(sample.features)
        errors.append((prediction - sample.label) ** 2)
    return sum(errors) / len(errors)


def _derive_weights(
    samples: Sequence[TrainingSample], training_loss: float, validation_loss: float
) -> ModelWeights:
    if not samples:
        return ModelWeights()

    avg_abs_feature = mean(
        abs(value)
        for sample in samples
        for value in sample.features
    ) if any(sample.features for sample in samples) else 0.0
    avg_label = mean(sample.label for sample in samples)
    positive_ratio = (
        sum(1 for sample in samples if sample.label >= 0) / len(samples)
    )

    lstm_weight = max(0.1, 1.0 - min(training_loss, 1.0))
    gbdt_weight = max(0.1, min(avg_abs_feature, 1.0))
    transformer_weight = max(0.1, min(abs(avg_label), 1.0))
    svm_weight = max(0.1, min(positive_ratio + (1.0 - min(validation_loss, 1.0)) / 2, 1.5))

    return ModelWeights(
        lstm_ob=lstm_weight,
        gbdt_features=gbdt_weight,
        transformer_temporal=transformer_weight,
        svm_volatility=svm_weight,
    )


def _as_iterable(values: object) -> Iterable[float]:
    if isinstance(values, (list, tuple)):
        return values
    if values is None:
        return []
    return [float(values)]


def _build_model_payloads(samples: Sequence[TrainingSample]) -> Dict[str, Dict[str, object]]:
    if not samples:
        return {
            "lstm_ob": {"weights": [0.0], "bias": 0.0, "scale": 1.0, "momentum": 0.6},
            "gbdt_feat": {
                "stages": [
                    {"weights": [0.0], "bias": 0.0, "shrinkage": 0.6},
                ],
                "scale": 1.0,
            },
            "transformer": {"weights": [0.0], "bias": 0.0, "scale": 1.0, "window_sizes": [5, 10, 20]},
            "svm_vol": {"weights": [0.0], "bias": 0.0, "scale": 1.0, "margin": 0.1},
        }

    matrix, labels = _design_matrix(samples)
    weights, bias = _fit_linear_model(matrix, labels)
    if not weights:
        weights = [0.0]
    label_std = _spread(labels)
    feature_std = _spread([value for row in matrix for value in row])
    max_len = len(weights)

    lstm_payload = {
        "weights": weights,
        "bias": bias,
        "scale": _scale_from_variance(label_std),
        "momentum": min(0.9, max(0.3, 1.0 - label_std / (abs(bias) + 1e-6))),
    }

    shrinkages = [0.6, 0.3, 0.1]
    stage_bias = bias / len(shrinkages)
    gbdt_payload = {
        "stages": [
            {
                "weights": [float(shrink * value) for value in weights],
                "bias": float(stage_bias * shrink),
                "shrinkage": float(shrink),
            }
            for shrink in shrinkages
        ],
        "scale": _scale_from_variance(feature_std),
    }

    window_sizes = _suggest_window_sizes(max_len)
    transformer_payload = {
        "weights": weights,
        "bias": bias,
        "scale": _scale_from_variance(feature_std),
        "window_sizes": window_sizes,
    }

    margin = min(0.5, max(0.05, label_std / (abs(bias) + 1e-6)))
    svm_payload = {
        "weights": _normalise_weights(weights),
        "bias": bias,
        "scale": _scale_from_variance(label_std),
        "margin": margin,
    }

    return {
        "lstm_ob": lstm_payload,
        "gbdt_feat": gbdt_payload,
        "transformer": transformer_payload,
        "svm_vol": svm_payload,
    }


def _design_matrix(samples: Sequence[TrainingSample]) -> Tuple[List[List[float]], List[float]]:
    max_len = max(len(sample.features) for sample in samples) if samples else 0
    matrix: List[List[float]] = []
    labels: List[float] = []
    for sample in samples:
        row = list(sample.features)
        if len(row) < max_len:
            row.extend([0.0] * (max_len - len(row)))
        matrix.append(row)
        labels.append(sample.label)
    return matrix, labels


def _fit_linear_model(matrix: Sequence[Sequence[float]], labels: Sequence[float]) -> Tuple[List[float], float]:
    if not matrix or not labels:
        return [], 0.0
    feature_count = len(matrix[0]) if matrix else 0
    weights = [0.0 for _ in range(feature_count)]
    bias = 0.0
    learning_rate = 0.01
    epochs = 200
    lam = 0.001
    for _ in range(epochs):
        grad_w = [0.0 for _ in range(feature_count)]
        grad_b = 0.0
        for features, target in zip(matrix, labels):
            prediction = sum(w * f for w, f in zip(weights, features)) + bias
            error = prediction - target
            for idx, feature in enumerate(features):
                grad_w[idx] += error * feature + lam * weights[idx]
            grad_b += error
        count = float(len(matrix)) or 1.0
        for idx in range(feature_count):
            weights[idx] -= learning_rate * (grad_w[idx] / count)
        bias -= learning_rate * (grad_b / count)
    return weights, bias


def _spread(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    if len(values) == 1:
        return max(1e-6, abs(values[0]))
    return max(1e-6, pstdev(values))


def _scale_from_variance(spread: float) -> float:
    spread = max(1e-6, spread)
    scale = 1.0 / spread
    return max(0.5, min(5.0, scale))


def _suggest_window_sizes(length: int) -> List[int]:
    if length <= 0:
        return [5, 10, 20]
    one = max(5, length // 4)
    two = max(one + 5, length // 2)
    three = max(two + 5, length)
    return [one, two, three]


def _normalise_weights(weights: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in weights)) or 1.0
    return [float(value / norm) for value in weights]


def _persist_model_bundle(model_dir: Path, payloads: Dict[str, Dict[str, object]]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "lstm_ob": "lstm_ob.tflite",
        "gbdt_feat": "gbdt_features.tflite",
        "transformer": "temporal_transformer.tflite",
        "svm_vol": "volatility_svm.tflite",
    }
    for key, payload in payloads.items():
        filename = filenames[key]
        path = model_dir / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "ModelTrainer",
    "TrainingReport",
    "TrainingSample",
]
