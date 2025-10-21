"""Configuration models for the scalping system."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional


@dataclass
class LoggingConfig:
    level: str = "INFO"
    json: bool = False
    sentry_dsn: Optional[str] = None


@dataclass
class DataFeedConfig:
    sandbox_token: Optional[str] = None
    production_token: Optional[str] = None
    use_sandbox: bool = True
    instruments: list[str] = field(default_factory=list)
    depth: int = 20
    candle_interval: str = "1min"
    throttle_rate_limit: float = 0.05


@dataclass
class FeatureConfig:
    lob_levels: int = 10
    rolling_window: int = 120
    include_micro_price: bool = True
    include_order_flow_imbalance: bool = True
    include_volatility_cluster: bool = True


@dataclass
class ModelWeights:
    lstm_ob: float = 0.4
    gbdt_features: float = 0.3
    transformer_temporal: float = 0.2
    svm_volatility: float = 0.1

    def normalise(self) -> None:
        total = self.lstm_ob + self.gbdt_features + self.transformer_temporal + self.svm_volatility
        if not total:
            self.lstm_ob = 0.25
            self.gbdt_features = 0.25
            self.transformer_temporal = 0.25
            self.svm_volatility = 0.25
            return
        self.lstm_ob /= total
        self.gbdt_features /= total
        self.transformer_temporal /= total
        self.svm_volatility /= total


@dataclass
class MLConfig:
    batch_size: int = 64
    max_latency_ms: int = 20
    weights: ModelWeights = field(default_factory=ModelWeights)
    drift_threshold: float = 0.05


@dataclass
class RiskLimits:
    max_position: int = 100
    max_daily_loss: float = 10000.0
    max_order_rate_per_minute: int = 120
    var_horizon_minutes: int = 30
    max_gross_exposure: float = 500000.0


@dataclass
class ExecutionConfig:
    account_id: Optional[str] = None
    venue: Literal["MOEX", "SPB"] = "MOEX"
    fallback_latency_ms: int = 150
    max_slippage_bps: int = 5


@dataclass
class StorageConfig:
    base_path: Path = Path("./runtime")
    enable_zstd: bool = True

    def ensure(self) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)


@dataclass
class OrchestratorConfig:
    environment: str = "development"
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    datafeed: DataFeedConfig = field(default_factory=DataFeedConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def __post_init__(self) -> None:
        self.storage.ensure()
        self.ml.weights.normalise()

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "OrchestratorConfig":
        return cls(
            environment=data.get("environment", "development"),
            logging=LoggingConfig(**_ensure_dict(data.get("logging", {}))),
            datafeed=DataFeedConfig(**_ensure_dict(data.get("datafeed", {}))),
            features=FeatureConfig(**_ensure_dict(data.get("features", {}))),
            ml=MLConfig(
                batch_size=_ensure_dict(data.get("ml", {})).get("batch_size", 64),
                max_latency_ms=_ensure_dict(data.get("ml", {})).get("max_latency_ms", 20),
                weights=ModelWeights(**_ensure_dict(_ensure_dict(data.get("ml", {})).get("weights", {}))),
                drift_threshold=_ensure_dict(data.get("ml", {})).get("drift_threshold", 0.05),
            ),
            risk=RiskLimits(**_ensure_dict(data.get("risk", {}))),
            execution=ExecutionConfig(**_ensure_dict(data.get("execution", {}))),
            storage=StorageConfig(**_ensure_dict(data.get("storage", {}))),
        )

    def json(self, indent: int = 2) -> str:
        return _to_json(self, indent=indent)


def _ensure_dict(value: object) -> Dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _to_json(config: OrchestratorConfig, indent: int = 2) -> str:
    import json

    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return {k: convert(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    return json.dumps(convert(config), indent=indent)


__all__ = [
    "LoggingConfig",
    "DataFeedConfig",
    "FeatureConfig",
    "ModelWeights",
    "MLConfig",
    "RiskLimits",
    "ExecutionConfig",
    "StorageConfig",
    "OrchestratorConfig",
]
