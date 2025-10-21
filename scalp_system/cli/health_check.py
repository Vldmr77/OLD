"""Basic health checks for system components."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from scalp_system.config import DEFAULT_CONFIG_PATH
from scalp_system.config.base import OrchestratorConfig
from scalp_system.config.loader import load_config
from scalp_system.data.engine import DataEngine
from scalp_system.features.pipeline import FeaturePipeline
from scalp_system.ml.calibration import CalibrationCoordinator
from scalp_system.ml.engine import MLSignal, MLEngine
from scalp_system.monitoring.drift import DriftDetector
from scalp_system.monitoring.notifications import NotificationDispatcher
from scalp_system.risk.engine import RiskEngine
from scalp_system.storage.repository import SQLiteRepository


def _check_storage(config: OrchestratorConfig) -> str:
    path = config.storage.base_path / "health_check.db"
    repository = SQLiteRepository(path)
    repository.summary()
    return f"storage:{path}"


def _check_data(config: OrchestratorConfig) -> str:
    engine = DataEngine(
        ttl_seconds=config.datafeed.current_cache_ttl,
        max_instruments=config.datafeed.current_cache_size,
        history_size=config.datafeed.history_length,
        monitored_instruments=config.datafeed.monitored_instruments,
        max_active_instruments=config.datafeed.max_active_instruments,
        monitor_pool_size=config.datafeed.monitor_pool_size,
        trade_history_size=config.datafeed.trade_history_size,
        candle_history_size=config.datafeed.candle_history_size,
    )
    engine.seed_instruments(
        config.datafeed.instruments, config.datafeed.monitored_instruments
    )
    FeaturePipeline(config.features)
    return f"data:{len(engine.active_instruments())} active"


def _check_ml(config: OrchestratorConfig) -> str:
    ml_engine = MLEngine(config.ml)
    ml_engine.set_instrument_classes(config.datafeed.instrument_classes)
    return f"ml:models={len(ml_engine._models)}"  # type: ignore[attr-defined]


def _check_notifications(config: OrchestratorConfig) -> str:
    dispatcher = NotificationDispatcher(config.notifications)
    asyncio.run(
        dispatcher.notify_latency_violation("health", 1.0, 10.0, severity="warning")
    )
    return "notifications:ok"


def _check_risk(config: OrchestratorConfig) -> str:
    risk = RiskEngine(config.risk)
    plan = risk.calculate_position_size(
        signal=MLSignal(figi="FIGI", direction=1, confidence=0.5),
        price=100.0,
        spread=0.01,
        atr=0.05,
    )
    return f"risk:position_size={plan}"


def _check_calibration(config: OrchestratorConfig) -> str:
    DriftDetector(
        threshold=config.ml.drift_threshold,
        history_dir=config.storage.base_path / "drift_metrics",
    )
    CalibrationCoordinator(
        queue_path=config.storage.base_path / "calibration_queue.jsonl"
    )
    return "calibration:ok"


CHECKS: Dict[str, Callable[[OrchestratorConfig], str]] = {
    "storage": _check_storage,
    "data": _check_data,
    "ml": _check_ml,
    "notifications": _check_notifications,
    "risk": _check_risk,
    "calibration": _check_calibration,
}


def run_health_checks(
    config_path: Optional[Path] = None,
    *,
    module: str = "all",
    config: Optional[OrchestratorConfig] = None,
) -> List[str]:
    """Validate that critical runtime components can be initialised."""

    if config is None:
        config_path = config_path or DEFAULT_CONFIG_PATH
        config = load_config(config_path)
    selected: Iterable[str]
    if module == "all":
        selected = CHECKS.keys()
    else:
        if module not in CHECKS:
            raise ValueError(f"Unknown module '{module}'")
        selected = (module,)
    results: List[str] = []
    for name in selected:
        results.append(CHECKS[name](config))
    return results


def main() -> None:
    """CLI entry point for running health checks."""

    parser = argparse.ArgumentParser(description="Run system health checks")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration (defaults to packaged example)",
    )
    parser.add_argument(
        "--module",
        default="all",
        choices=["all", *CHECKS.keys()],
        help="Run checks for a specific module or all modules",
    )
    args = parser.parse_args()
    results = run_health_checks(args.config, module=args.module)
    for line in results:
        print(line)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
