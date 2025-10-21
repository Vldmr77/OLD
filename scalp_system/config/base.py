"""Configuration models for the scalping system."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional


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
    monitored_instruments: list[str] = field(default_factory=list)
    depth: int = 20
    candle_interval: str = "1min"
    throttle_rate_limit: float = 0.05
    current_cache_ttl: float = 60.0
    current_cache_size: int = 50
    history_length: int = 30
    soft_rss_limit_mb: int = 400
    hard_rss_limit_mb: int = 700
    reconnect_delay: float = 1.0


@dataclass
class FeatureConfig:
    lob_levels: int = 10
    rolling_window: int = 120
    include_micro_price: bool = True
    include_order_flow_imbalance: bool = True
    include_volatility_cluster: bool = True


@dataclass
class ModelWeights:
    lstm_ob: float = 0.35
    gbdt_features: float = 0.25
    transformer_temporal: float = 0.25
    svm_volatility: float = 0.15

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
    loss_cooldown_minutes: int = 15
    max_consecutive_losses: int = 3


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
    cache_flush_min_seconds: float = 1.0
    cache_flush_max_seconds: float = 5.0
    cache_target_buffer: int = 50

    def ensure(self) -> None:
        if not isinstance(self.base_path, Path):
            self.base_path = Path(self.base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)
        if self.cache_flush_min_seconds <= 0:
            self.cache_flush_min_seconds = 1.0
        if self.cache_flush_max_seconds < self.cache_flush_min_seconds:
            self.cache_flush_max_seconds = self.cache_flush_min_seconds
        if self.cache_target_buffer <= 0:
            self.cache_target_buffer = 1


@dataclass
class MonitoringConfig:
    cpu_soft_limit: float = 90.0
    memory_soft_limit: float = 90.0
    gpu_soft_limit: float = 90.0
    audit_log_filename: str = "audit.log"


@dataclass
class SecurityConfig:
    encryption_key_path: Optional[Path] = None


@dataclass
class NotificationConfig:
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    enable_sound_alerts: bool = False
    high_risk_frequency_hz: int = 2500
    low_liquidity_frequency_hz: int = 1500
    liquidity_spread_threshold_bps: float = 5.0
    cooldown_seconds: int = 30


@dataclass
class TrainingConfig:
    dataset_path: Path = Path("./data/training.jsonl")
    output_dir: Path = Path("./models")
    epochs: int = 5
    learning_rate: float = 0.01
    validation_split: float = 0.2
    min_samples: int = 10
    enable_quantization: bool = True
    quantization_int8_threshold: float = 0.5

    def ensure(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if self.quantization_int8_threshold <= 0:
            self.quantization_int8_threshold = 0.5
        elif self.quantization_int8_threshold >= 1:
            self.quantization_int8_threshold = 0.9


@dataclass
class BacktestConfig:
    dataset_path: Path = Path("./data/backtest.jsonl")
    initial_capital: float = 100000.0
    transaction_cost_bps: float = 1.0
    slippage_bps: float = 0.5
    max_orders: int = 1000

    def ensure(self) -> None:
        if not isinstance(self.dataset_path, Path):
            self.dataset_path = Path(self.dataset_path).expanduser()
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if self.initial_capital <= 0:
            self.initial_capital = 10000.0
        if self.transaction_cost_bps < 0:
            self.transaction_cost_bps = 0.0
        if self.slippage_bps < 0:
            self.slippage_bps = 0.0
        if self.max_orders <= 0:
            self.max_orders = 100


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
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def __post_init__(self) -> None:
        self.storage.ensure()
        self.ml.weights.normalise()
        self.training.ensure()
        self.backtest.ensure()

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "OrchestratorConfig":
        monitoring_data = _ensure_dict(data.get("monitoring", {}))
        security_data = _ensure_dict(data.get("security", {}))
        notifications_data = _ensure_dict(data.get("notifications", {}))
        training_data = _ensure_dict(data.get("training", {}))
        backtest_data = _ensure_dict(data.get("backtest", {}))
        encryption_value = security_data.get("encryption_key_path")
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
            monitoring=MonitoringConfig(**monitoring_data),
            security=SecurityConfig(
                encryption_key_path=_to_path(encryption_value) if encryption_value else None
            ),
            notifications=NotificationConfig(**notifications_data),
            training=TrainingConfig(
                dataset_path=_to_path(training_data.get("dataset_path", Path("./data/training.jsonl"))),
                output_dir=_to_path(training_data.get("output_dir", Path("./models"))),
                epochs=training_data.get("epochs", 5),
                learning_rate=training_data.get("learning_rate", 0.01),
                validation_split=training_data.get("validation_split", 0.2),
                min_samples=training_data.get("min_samples", 10),
                enable_quantization=training_data.get("enable_quantization", True),
                quantization_int8_threshold=training_data.get("quantization_int8_threshold", 0.5),
            ),
            backtest=BacktestConfig(
                dataset_path=_to_path(backtest_data.get("dataset_path", Path("./data/backtest.jsonl"))),
                initial_capital=backtest_data.get("initial_capital", 100000.0),
                transaction_cost_bps=backtest_data.get("transaction_cost_bps", 1.0),
                slippage_bps=backtest_data.get("slippage_bps", 0.5),
                max_orders=backtest_data.get("max_orders", 1000),
            ),
        )

    def json(self, indent: int = 2) -> str:
        return _to_json(self, indent=indent)


def _ensure_dict(value: object) -> Dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _to_path(value: object) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value).expanduser()
    raise TypeError("Expected string or Path for path value")


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
    "MonitoringConfig",
    "NotificationConfig",
    "SecurityConfig",
    "TrainingConfig",
    "BacktestConfig",
    "OrchestratorConfig",
]
