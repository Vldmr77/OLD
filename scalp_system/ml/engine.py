"""ML inference orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from ..config.base import MLConfig
from ..features.pipeline import FeatureVector
from .models.base import FeatureModel, ModelPrediction, WeightedEnsemble
from .models.gbdt_features import GradientBoostedFeatureModel
from .models.lstm_orderbook import LSTMOrderBookModel
from .models.temporal_transformer import TemporalTransformerModel
from .models.volatility_svm import VolatilityClusterSVM


@dataclass
class MLSignal:
    figi: str
    direction: int  # 1 for buy, -1 for sell
    confidence: float
    source: str = "ml_engine"


class MLEngine:
    def __init__(self, config: MLConfig) -> None:
        self._config = config
        self._models: Dict[str, FeatureModel] = {
            "lstm_ob": LSTMOrderBookModel(),
            "gbdt_feat": GradientBoostedFeatureModel(),
            "transformer": TemporalTransformerModel(),
            "svm_vol": VolatilityClusterSVM(),
        }
        self._ensemble = WeightedEnsemble(
            weights={
                "lstm_ob": config.weights.lstm_ob,
                "gbdt_feat": config.weights.gbdt_features,
                "transformer": config.weights.transformer_temporal,
                "svm_vol": config.weights.svm_volatility,
            }
        )

    def infer(self, batch: Iterable[FeatureVector]) -> list[MLSignal]:
        batch = list(batch)
        feature_batches = [vector.features for vector in batch]
        predictions: Dict[str, list[ModelPrediction]] = {}
        for name, model in self._models.items():
            predictions[name] = model.predict(feature_batches)
        signals: list[MLSignal] = []
        for idx, vector in enumerate(batch):
            combined = self._ensemble.combine({name: preds[idx] for name, preds in predictions.items()})
            direction = 1 if combined.score >= 0 else -1
            signals.append(
                MLSignal(figi=vector.figi, direction=direction, confidence=combined.confidence)
            )
        return signals


__all__ = ["MLSignal", "MLEngine"]
