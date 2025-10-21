"""Offline model training utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

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


__all__ = [
    "ModelTrainer",
    "TrainingReport",
    "TrainingSample",
]
