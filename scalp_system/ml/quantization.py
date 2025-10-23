"""Utilities for describing model quantisation plans."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal

from scalp_system.config.base import ModelWeights


@dataclass(frozen=True)
class QuantizationDecision:
    """Decision describing how a particular model should be quantised."""

    model_name: str
    precision: Literal["int8", "float16"]
    weight: float
    converter_directive: str


def build_quantization_plan(weights: ModelWeights, threshold: float) -> List[QuantizationDecision]:
    """Derive quantisation directives for ensemble models."""

    mapping = {
        "lstm_ob": weights.lstm_ob,
        "gbdt_features": weights.gbdt_features,
        "transformer_temporal": weights.transformer_temporal,
        "svm_volatility": weights.svm_volatility,
    }
    decisions: List[QuantizationDecision] = []
    for name, weight in mapping.items():
        if weight > threshold:
            decisions.append(
                QuantizationDecision(
                    model_name=name,
                    precision="int8",
                    weight=weight,
                    converter_directive="tf.lite.OpsSet.TFLITE_BUILTINS_INT8",
                )
            )
        else:
            decisions.append(
                QuantizationDecision(
                    model_name=name,
                    precision="float16",
                    weight=weight,
                    converter_directive="tf.float16",
                )
            )
    return decisions


def persist_quantization_plan(decisions: Iterable[QuantizationDecision], output_dir: Path) -> Path:
    """Serialise the quantisation plan alongside the training artefacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "quantization_plan.json"
    payload = [
        {
            "model": decision.model_name,
            "precision": decision.precision,
            "weight": decision.weight,
            "converter": decision.converter_directive,
        }
        for decision in decisions
    ]
    plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return plan_path


__all__ = [
    "QuantizationDecision",
    "build_quantization_plan",
    "persist_quantization_plan",
]
