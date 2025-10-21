"""Configuration models for the scalping system."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
    heartbeat_enabled: bool = True
    heartbeat_interval_seconds: float = 5.0
    heartbeat_miss_threshold: int = 3

    def ensure(self) -> None:
        if self.heartbeat_interval_seconds <= 0:
            self.heartbeat_interval_seconds = 5.0
        self.heartbeat_miss_threshold = max(1, int(self.heartbeat_miss_threshold))


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
class FallbackConfig:
    enabled: bool = True
    max_signals_per_hour: int = 10
    imbalance_threshold: float = 0.7
    volatility_threshold: float = 0.01
    confidence: float = 0.7

    def ensure(self) -> None:
        self.max_signals_per_hour = max(1, int(self.max_signals_per_hour))
        self.imbalance_threshold = float(min(max(self.imbalance_threshold, 0.0), 1.0))
        self.volatility_threshold = float(max(self.volatility_threshold, 0.0))
        self.confidence = float(min(max(self.confidence, 0.0), 1.0))


@dataclass
class ConnectivityConfig:
    primary_label: str = "fiber"
    backup_label: str = "lte"
    failure_threshold: int = 1
    recovery_message_count: int = 20
    failover_latency_ms: int = 150

    def ensure(self) -> None:
        self.primary_label = self.primary_label or "fiber"
        self.backup_label = self.backup_label or "lte"
        if self.backup_label == self.primary_label:
            self.backup_label = f"{self.backup_label}_backup"
        self.failure_threshold = max(1, int(self.failure_threshold))
        self.recovery_message_count = max(1, int(self.recovery_message_count))
        self.failover_latency_ms = max(1, int(self.failover_latency_ms))


@dataclass
class ManualOverrideConfig:
    enabled: bool = False
    flag_path: Path = Path("./runtime/manual_override.flag")
    poll_interval_seconds: float = 1.0
    auto_resume_minutes: Optional[int] = None

    def ensure(self) -> None:
        if not isinstance(self.flag_path, Path):
            self.flag_path = Path(self.flag_path).expanduser()
        self.flag_path.parent.mkdir(parents=True, exist_ok=True)
        if self.poll_interval_seconds <= 0:
            self.poll_interval_seconds = 1.0
        if self.auto_resume_minutes is not None:
            try:
                self.auto_resume_minutes = int(self.auto_resume_minutes)
            except (TypeError, ValueError):
                self.auto_resume_minutes = None
            if self.auto_resume_minutes is not None and self.auto_resume_minutes <= 0:
                self.auto_resume_minutes = None


@dataclass
class DisasterRecoveryConfig:
    enabled: bool = False
    replication_path: Path = Path("./runtime/backups")
    include_patterns: list[str] = field(
        default_factory=lambda: ["*.db", "*.json", "*.jsonl"]
    )
    max_snapshots: int = 5
    interval_minutes: int = 60

    def ensure(self) -> None:
        if not isinstance(self.replication_path, Path):
            self.replication_path = Path(self.replication_path).expanduser()
        self.replication_path.mkdir(parents=True, exist_ok=True)
        if not self.include_patterns:
            self.include_patterns = ["*.db", "*.json", "*.jsonl"]
        self.max_snapshots = max(1, int(self.max_snapshots))
        self.interval_minutes = max(0, int(self.interval_minutes))


@dataclass
class SessionScheduleConfig:
    enabled: bool = False
    start_time: str = "09:50"
    end_time: str = "23:50"
    pre_open_minutes: int = 5
    post_close_minutes: int = 5
    allowed_weekdays: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    def ensure(self) -> None:
        self.pre_open_minutes = max(0, int(self.pre_open_minutes))
        self.post_close_minutes = max(0, int(self.post_close_minutes))
        self.start_time = _validate_time(self.start_time, default="09:50")
        self.end_time = _validate_time(self.end_time, default="23:50")
        cleaned: list[int] = []
        for value in self.allowed_weekdays:
            try:
                weekday = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= weekday <= 6:
                cleaned.append(weekday)
        if not cleaned:
            cleaned = list(range(7))
        self.allowed_weekdays = sorted(dict.fromkeys(cleaned))


@dataclass
class SystemConfig:
    mode: Literal["development", "forward-test", "production"] = "development"
    startup_time: Optional[str] = None
    checkpoint_interval_seconds: int = 300
    latency_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "features": 12.0,
            "ml": 20.0,
            "risk": 8.0,
        }
    )
    latency_violation_limit: int = 3

    def ensure(self) -> None:
        if self.checkpoint_interval_seconds < 0:
            self.checkpoint_interval_seconds = 0
        self.latency_violation_limit = max(1, self.latency_violation_limit)
        if not isinstance(self.latency_thresholds, dict):
            self.latency_thresholds = {
                "features": 12.0,
                "ml": 20.0,
                "risk": 8.0,
            }
        else:
            defaults = {"features": 12.0, "ml": 20.0, "risk": 8.0}
            merged = {**defaults, **self.latency_thresholds}
            self.latency_thresholds = merged


@dataclass
class ReportingConfig:
    enabled: bool = True
    interval_minutes: int = 60
    max_history_days: int = 14
    report_path: Path = Path("./runtime/reports/performance.jsonl")

    def ensure(self) -> None:
        if not isinstance(self.report_path, Path):
            self.report_path = Path(self.report_path).expanduser()
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_minutes = max(1, int(self.interval_minutes))
        self.max_history_days = max(1, int(self.max_history_days))


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
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    manual_override: ManualOverrideConfig = field(default_factory=ManualOverrideConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    session: SessionScheduleConfig = field(default_factory=SessionScheduleConfig)
    disaster_recovery: DisasterRecoveryConfig = field(
        default_factory=DisasterRecoveryConfig
    )

    def __post_init__(self) -> None:
        self.storage.ensure()
        self.ml.weights.normalise()
        self.training.ensure()
        self.backtest.ensure()
        self.system.ensure()
        self.reporting.ensure()
        self.manual_override.ensure()
        self.fallback.ensure()
        self.connectivity.ensure()
        self.monitoring.ensure()
        self.disaster_recovery.ensure()
        self.session.ensure()

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "OrchestratorConfig":
        monitoring_data = _ensure_dict(data.get("monitoring", {}))
        security_data = _ensure_dict(data.get("security", {}))
        notifications_data = _ensure_dict(data.get("notifications", {}))
        fallback_data = _ensure_dict(data.get("fallback", {}))
        connectivity_data = _ensure_dict(data.get("connectivity", {}))
        manual_override_data = _ensure_dict(data.get("manual_override", {}))
        training_data = _ensure_dict(data.get("training", {}))
        backtest_data = _ensure_dict(data.get("backtest", {}))
        system_data = _ensure_dict(data.get("system", {}))
        reporting_data = _ensure_dict(data.get("reporting", {}))
        disaster_data = _ensure_dict(data.get("disaster_recovery", {}))
        session_data = _ensure_dict(data.get("session", {}))
        encryption_value = security_data.get("encryption_key_path")
        include_patterns = disaster_data.get("include_patterns")
        if isinstance(include_patterns, list):
            patterns = [str(value) for value in include_patterns]
        else:
            patterns = ["*.db", "*.json", "*.jsonl"]
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
            monitoring=MonitoringConfig(
                cpu_soft_limit=monitoring_data.get("cpu_soft_limit", 90.0),
                memory_soft_limit=monitoring_data.get("memory_soft_limit", 90.0),
                gpu_soft_limit=monitoring_data.get("gpu_soft_limit", 90.0),
                audit_log_filename=monitoring_data.get("audit_log_filename", "audit.log"),
                heartbeat_enabled=monitoring_data.get("heartbeat_enabled", True),
                heartbeat_interval_seconds=monitoring_data.get(
                    "heartbeat_interval_seconds", 5.0
                ),
                heartbeat_miss_threshold=monitoring_data.get(
                    "heartbeat_miss_threshold", 3
                ),
            ),
            security=SecurityConfig(
                encryption_key_path=_to_path(encryption_value) if encryption_value else None
            ),
            notifications=NotificationConfig(**notifications_data),
            fallback=FallbackConfig(
                enabled=fallback_data.get("enabled", True),
                max_signals_per_hour=fallback_data.get("max_signals_per_hour", 10),
                imbalance_threshold=fallback_data.get("imbalance_threshold", 0.7),
                volatility_threshold=fallback_data.get("volatility_threshold", 0.01),
                confidence=fallback_data.get("confidence", 0.7),
            ),
            connectivity=ConnectivityConfig(
                primary_label=connectivity_data.get("primary_label", "fiber"),
                backup_label=connectivity_data.get("backup_label", "lte"),
                failure_threshold=connectivity_data.get("failure_threshold", 1),
                recovery_message_count=connectivity_data.get("recovery_message_count", 20),
                failover_latency_ms=connectivity_data.get("failover_latency_ms", 150),
            ),
            manual_override=ManualOverrideConfig(
                enabled=manual_override_data.get("enabled", False),
                flag_path=_to_path(
                    manual_override_data.get(
                        "flag_path", Path("./runtime/manual_override.flag")
                    )
                ),
                poll_interval_seconds=float(
                    manual_override_data.get("poll_interval_seconds", 1.0)
                ),
                auto_resume_minutes=manual_override_data.get("auto_resume_minutes"),
            ),
            reporting=ReportingConfig(
                enabled=reporting_data.get("enabled", True),
                interval_minutes=reporting_data.get("interval_minutes", 60),
                max_history_days=reporting_data.get("max_history_days", 14),
                report_path=_to_path(
                    reporting_data.get(
                        "report_path", Path("./runtime/reports/performance.jsonl")
                    )
                ),
            ),
            disaster_recovery=DisasterRecoveryConfig(
                enabled=disaster_data.get("enabled", False),
                replication_path=_to_path(
                    disaster_data.get(
                        "replication_path", Path("./runtime/backups")
                    )
                ),
                include_patterns=patterns,
                max_snapshots=disaster_data.get("max_snapshots", 5),
                interval_minutes=disaster_data.get("interval_minutes", 60),
            ),
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
            system=SystemConfig(
                mode=system_data.get("mode", "development"),
                startup_time=system_data.get("startup_time"),
                checkpoint_interval_seconds=system_data.get("checkpoint_interval_seconds", 300),
                latency_thresholds=_ensure_dict(system_data.get("latency_thresholds", {})),
                latency_violation_limit=system_data.get("latency_violation_limit", 3),
            ),
            session=SessionScheduleConfig(
                enabled=session_data.get("enabled", False),
                start_time=session_data.get("start_time", "09:50"),
                end_time=session_data.get("end_time", "23:50"),
                pre_open_minutes=session_data.get("pre_open_minutes", 5),
                post_close_minutes=session_data.get("post_close_minutes", 5),
                allowed_weekdays=list(session_data.get("allowed_weekdays", [0, 1, 2, 3, 4])),
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


def _validate_time(value: object, *, default: str) -> str:
    if isinstance(value, str):
        try:
            datetime.strptime(value, "%H:%M")
            return value
        except ValueError:
            pass
    return default


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
    "FallbackConfig",
    "SecurityConfig",
    "ManualOverrideConfig",
    "SessionScheduleConfig",
    "SystemConfig",
    "ReportingConfig",
    "TrainingConfig",
    "BacktestConfig",
    "DisasterRecoveryConfig",
    "OrchestratorConfig",
]
