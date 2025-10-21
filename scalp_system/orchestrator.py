"""Main orchestrator for the scalping system."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .config.base import OrchestratorConfig
from .config.loader import load_config
from .data.engine import DataEngine
from .data.streams import MarketDataStream, iterate_stream
from .execution.client import BrokerClient
from .execution.executor import ExecutionEngine
from .features.pipeline import FeaturePipeline
from .ml.engine import MLEngine
from .ml.calibration import CalibrationCoordinator
from .monitoring.audit import AuditLogger
from .monitoring.drift import DriftDetector
from .monitoring.metrics import MetricsRegistry
from .monitoring.notifications import NotificationDispatcher
from .monitoring.resource import ResourceMonitor
from .risk.engine import RiskEngine
from .security import KeyManager
from .storage.repository import SQLiteRepository
from .utils.integrity import check_data_integrity
from .utils.timing import timed

LOGGER = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config
        self._feature_pipeline = FeaturePipeline(config.features)
        self._ml_engine = MLEngine(config.ml)
        self._risk_engine = RiskEngine(config.risk)
        self._metrics = MetricsRegistry()
        self._repository = SQLiteRepository(
            config.storage.base_path / "signals.db",
            cache_min_interval=config.storage.cache_flush_min_seconds,
            cache_max_interval=config.storage.cache_flush_max_seconds,
            cache_target_buffer=config.storage.cache_target_buffer,
        )
        self._drift_detector = DriftDetector(
            threshold=config.ml.drift_threshold,
            history_dir=config.storage.base_path / "drift_metrics",
        )
        self._data_engine = DataEngine(
            ttl_seconds=config.datafeed.current_cache_ttl,
            max_instruments=config.datafeed.current_cache_size,
            history_size=config.datafeed.history_length,
            monitored_instruments=config.datafeed.monitored_instruments,
            soft_rss_limit_mb=config.datafeed.soft_rss_limit_mb,
            hard_rss_limit_mb=config.datafeed.hard_rss_limit_mb,
        )
        self._data_engine.update_active_instruments(config.datafeed.instruments)
        self._calibration = CalibrationCoordinator(
            queue_path=config.storage.base_path / "calibration_queue.jsonl"
        )
        self._audit_logger = AuditLogger(
            config.storage.base_path / config.monitoring.audit_log_filename
        )
        self._resource_monitor = ResourceMonitor(
            cpu_threshold=config.monitoring.cpu_soft_limit,
            memory_threshold=config.monitoring.memory_soft_limit,
            gpu_threshold=config.monitoring.gpu_soft_limit,
        )
        self._notifications = NotificationDispatcher(config.notifications)
        self._key_manager: Optional[KeyManager] = None
        key_path = config.security.encryption_key_path
        env_key = os.getenv("SCALP_ENCRYPTION_KEY")
        try:
            if key_path and key_path.exists():
                self._key_manager = KeyManager.from_file(key_path)
            elif env_key:
                self._key_manager = KeyManager.from_key_string(env_key)
        except ValueError:
            LOGGER.exception("Failed to load encryption key")
            self._key_manager = None

    def _market_stream_factory(self) -> Callable[[], MarketDataStream]:
        token = (
            self._config.datafeed.sandbox_token
            if self._config.datafeed.use_sandbox
            else self._config.datafeed.production_token
        )
        token = self._decrypt_token(token)
        if not token:
            raise RuntimeError("API token is required")

        def factory() -> MarketDataStream:
            return MarketDataStream(
                token=token,
                use_sandbox=self._config.datafeed.use_sandbox,
                instruments=self._config.datafeed.instruments,
                depth=self._config.datafeed.depth,
            )

        return factory

    def _broker_factory(self):
        token = (
            self._config.datafeed.sandbox_token
            if self._config.datafeed.use_sandbox
            else self._config.datafeed.production_token
        )
        token = self._decrypt_token(token)
        if not token:
            raise RuntimeError("Broker token is required")

        def factory() -> BrokerClient:
            return BrokerClient(token, sandbox=self._config.datafeed.use_sandbox, account_id=self._config.execution.account_id)

        return factory

    def _decrypt_token(self, token: Optional[str]) -> Optional[str]:
        if self._key_manager is None:
            return token
        return self._key_manager.maybe_decrypt(token)

    async def run(self) -> None:
        stream_factory = self._market_stream_factory()
        broker_factory = self._broker_factory()
        rotation_counter = 0
        async def integrity_probe() -> bool:
            return await check_data_integrity(self._data_engine)

        async for order_book in iterate_stream(
            stream_factory,
            delay=self._config.datafeed.reconnect_delay,
            integrity_check=integrity_probe,
        ):
            snapshot, mitigations = self._resource_monitor.check_thresholds()
            if mitigations:
                LOGGER.warning(
                    "Resource pressure detected cpu=%.1f mem=%.1f gpu=%s actions=%s",
                    snapshot.cpu_percent,
                    snapshot.memory_percent,
                    f"{snapshot.gpu_memory_percent:.1f}" if snapshot.gpu_memory_percent is not None else "n/a",
                    mitigations,
                )
                gpu_repr = (
                    f"{snapshot.gpu_memory_percent:.1f}"
                    if snapshot.gpu_memory_percent is not None
                    else "n/a"
                )
                for action in mitigations:
                    detail = (
                        f"cpu={snapshot.cpu_percent:.1f};"
                        f"mem={snapshot.memory_percent:.1f};gpu={gpu_repr}"
                    )
                    self._audit_logger.log("RESOURCE", action.upper(), detail)
            self._data_engine.ingest_order_book(order_book)
            mid_price = order_book.mid_price()
            spread = order_book.spread()
            if mid_price > 0:
                spread_bps = (spread / mid_price) * 10_000
                if (
                    spread_bps
                    >= self._config.notifications.liquidity_spread_threshold_bps
                ):
                    await self._notifications.notify_low_liquidity(
                        order_book.figi, spread_bps
                    )
            window = self._data_engine.last_window(
                order_book.figi, self._config.features.rolling_window
            )
            if len(window) < 2:
                continue
            with timed("features", self._metrics.record_latency):
                market = self._data_engine.market_indicators()
                features = self._feature_pipeline.transform(window, market=market)
            with timed("ml", self._metrics.record_latency):
                signals = self._ml_engine.infer([features])
            for signal in signals:
                self._metrics.record_signal(signal.figi, signal.confidence)
                self._repository.persist_signal(signal.figi, signal.direction, signal.confidence)
                if not self._risk_engine.evaluate_signal(signal, order_book.mid_price()):
                    LOGGER.info("Signal rejected by risk engine")
                    self._audit_logger.log(
                        "ORDER", "REJECTED", f"figi={signal.figi};confidence={signal.confidence:.3f}"
                    )
                    continue
                report = await ExecutionEngine(broker_factory, self._risk_engine).execute_signal(
                    signal, order_book.mid_price()
                )
                if report.accepted:
                    LOGGER.info("Order executed for %s", signal.figi)
                    self._audit_logger.log(
                        "ORDER", "EXECUTED", f"figi={signal.figi};confidence={signal.confidence:.3f}"
                    )
                    await self._notifications.notify_order_filled(
                        signal.figi, order_book.mid_price(), signal.confidence
                    )
                else:
                    LOGGER.warning("Order not executed: %s", report.reason)
                    self._audit_logger.log(
                        "ORDER", "FAILED", f"figi={signal.figi};reason={report.reason}"
                    )
            history = self._data_engine.history(order_book.figi)
            reference = [level.price for level in history[0].bids] if len(history) else []
            current = [level.price for level in order_book.bids]
            drift = self._drift_detector.evaluate(reference=reference, current=current)
            self._risk_engine.record_drift(drift)
            if drift.triggered:
                LOGGER.warning("Data drift detected p=%.4f mean_diff=%.4f", drift.p_value, drift.mean_diff)
                self._audit_logger.log(
                    "RISK", "DATA_DRIFT", f"figi={order_book.figi};p={drift.p_value:.4f};mean={drift.mean_diff:.4f}"
                )
            if drift.alert:
                LOGGER.error("Drift alert severity=%s halting trading", drift.severity)
                self._audit_logger.log(
                    "RISK", "CIRCUIT_BREAKER", f"figi={order_book.figi};severity={drift.severity}"
                )
                await self._notifications.notify_circuit_breaker(
                    order_book.figi, drift.severity
                )
            rotation_counter += 1
            if rotation_counter % 50 == 0:
                replacements = self._data_engine.rotate_instruments()
                if replacements:
                    LOGGER.info("Rotated instruments: %s", replacements)
            if self._risk_engine.calibration_required():
                enqueued = self._calibration.enqueue(
                    reason="drift",
                    metadata={
                        "figi": order_book.figi,
                        "p_value": drift.p_value,
                        "mean_diff": drift.mean_diff,
                        "sigma": drift.sigma,
                        "severity": drift.severity,
                    },
                    dedupe_key=f"drift:{order_book.figi}:{drift.severity}",
                )
                if enqueued:
                    LOGGER.info("Calibration request enqueued for %s", order_book.figi)
                    self._risk_engine.acknowledge_calibration()

    def reload_models(self, model_dir: Path) -> None:
        """Reload quantised models and clear cached feature state."""

        self._feature_pipeline.reset_cache()
        self._ml_engine.reload_models(model_dir)
        self._risk_engine.notify_model_reload()


def run_from_yaml(path: str | Path) -> None:
    config = load_config(path)
    logging.basicConfig(level=getattr(logging, config.logging.level))
    orchestrator = Orchestrator(config)
    asyncio.run(orchestrator.run())


__all__ = ["Orchestrator", "run_from_yaml"]
