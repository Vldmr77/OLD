"""Main orchestrator for the scalping system."""
from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from typing import Callable, Deque, Optional

from .broker.tinkoff import (
    TinkoffAPI,
    TinkoffSDKUnavailable,
    ensure_sdk_available,
)
from .config.base import OrchestratorConfig
from .config.loader import load_config
from .control.manual_override import ManualOverrideGuard
from .control.session import SessionGuard
from .data.engine import DataEngine
from .data.streams import MarketDataStream, OfflineMarketDataStream, iterate_stream
from .execution.client import BrokerClient
from .execution.executor import ExecutionEngine
from .features.pipeline import FeaturePipeline
from .ml.engine import MLSignal, MLEngine
from .ml.fallback import FallbackSignalGenerator
from .ml.calibration import CalibrationCoordinator
from .ml.training import ModelTrainer
from .monitoring.audit import AuditLogger
from .monitoring.drift import DriftDetector
from .monitoring.latency import LatencyAlert, LatencyMonitor
from .monitoring.metrics import MetricsRegistry
from .monitoring.heartbeat import HeartbeatMonitor, HeartbeatStatus
from .monitoring.notifications import NotificationDispatcher
from .monitoring.reporting import PerformanceReporter
from .monitoring.resource import ResourceMonitor
from .network import ConnectivityFailover
from .risk.engine import PositionAdjustment, RiskEngine
from .security import KeyManager
from .storage.checkpoint import CheckpointManager
from .storage.disaster_recovery import DisasterRecoveryManager
from .storage.historical import HistoricalDataStorage
from .storage.repository import SQLiteRepository
from .utils.integrity import check_data_integrity
from .utils.timing import timed
from .config.token_prompt import is_placeholder_token
from .ui import DashboardStatus, ModuleStatus, run_dashboard

LOGGER = logging.getLogger(__name__)


class RestartRequested(RuntimeError):
    """Raised when the dashboard requests an orchestrator restart."""


class Orchestrator:
    def __init__(self, config: OrchestratorConfig, *, config_path: Path | None = None) -> None:
        self._config = config
        self._feature_pipeline = FeaturePipeline(config.features)
        self._ml_engine = MLEngine(config.ml)
        self._risk_engine = RiskEngine(config.risk)
        self._risk_schedule = config.risk_schedule
        self._metrics = MetricsRegistry()
        self._repository = SQLiteRepository(
            config.dashboard.repository_path,
            cache_min_interval=config.storage.cache_flush_min_seconds,
            cache_max_interval=config.storage.cache_flush_max_seconds,
            cache_target_buffer=config.storage.cache_target_buffer,
        )
        self._historical_storage = HistoricalDataStorage(
            config.storage.base_path,
            orderbook_path=config.storage.orderbooks_path,
            trades_path=config.storage.trades_path,
            candles_path=config.storage.candles_path,
            features_path=config.storage.features_path,
            enable_zstd=config.storage.enable_zstd,
            parquet_compression=config.storage.parquet_compression,
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
            max_active_instruments=config.datafeed.max_active_instruments,
            monitor_pool_size=config.datafeed.monitor_pool_size,
            trade_history_size=config.datafeed.trade_history_size,
            candle_history_size=config.datafeed.candle_history_size,
        )
        self._data_engine.seed_instruments(
            config.datafeed.instruments, config.datafeed.monitored_instruments
        )
        self._ml_engine.set_instrument_classes(config.datafeed.instrument_classes)
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
        self._heartbeat = HeartbeatMonitor(
            enabled=config.monitoring.heartbeat_enabled,
            interval_seconds=config.monitoring.heartbeat_interval_seconds,
            miss_threshold=config.monitoring.heartbeat_miss_threshold,
        )
        self._fallback = FallbackSignalGenerator(config.fallback)
        self._connectivity = ConnectivityFailover(config.connectivity)
        self._latency_monitor = LatencyMonitor(
            config.system.latency_thresholds,
            violation_limit=config.system.latency_violation_limit,
        )
        self._config_path = Path(config_path).expanduser() if config_path else None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._restart_event: Optional[asyncio.Event] = None
        self._restart_callback_logged = False
        self._pending_restart = False
        self._latency_events: Deque[LatencyAlert] = deque()
        self._checkpoint_manager = CheckpointManager(
            config.storage.base_path / "checkpoint.json"
        )
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._disaster_recovery = DisasterRecoveryManager(
            config.storage.base_path,
            config.disaster_recovery,
            notifications=self._notifications,
        )
        self._manual_override = ManualOverrideGuard(config.manual_override)
        self._manual_override_active = False
        self._manual_override_reason: Optional[str] = None
        self._session_guard = SessionGuard(config.session)
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
        self._api_token = self._resolve_api_token()
        self._rest_api: Optional[TinkoffAPI] = None
        if self._api_token:
            self._rest_api = TinkoffAPI(
                token=self._api_token,
                use_sandbox=self._config.datafeed.use_sandbox,
                account_id=self._config.execution.account_id,
            )
        self._broker_factory_fn = self._broker_factory()
        self._execution_engine = ExecutionEngine(
            self._broker_factory_fn, self._risk_engine, mode=self._config.system.mode
        )
        self._restore_from_checkpoint()
        self._performance_reporter = PerformanceReporter(
            repository=self._repository,
            risk_engine=self._risk_engine,
            report_path=config.reporting.report_path,
            interval_minutes=config.reporting.interval_minutes,
            max_history_days=config.reporting.max_history_days,
            enabled=config.reporting.enabled,
            notifications=self._notifications,
        )
        self._training_task: Optional[asyncio.Task] = None
        self._risk_tasks: list[asyncio.Task] = []
        self._risk_task_labels: dict[asyncio.Task, str] = {}
        self._dashboard_thread: Optional[Thread] = None

    def _resolve_api_token(self) -> Optional[str]:
        token = (
            self._config.datafeed.sandbox_token
            if self._config.datafeed.use_sandbox
            else self._config.datafeed.production_token
        )
        if is_placeholder_token(token):
            return None
        decrypted = self._decrypt_token(token)
        if is_placeholder_token(decrypted):
            return None
        return decrypted

    def _market_stream_factory(self) -> Callable[[], MarketDataStream]:
        if not self._api_token:
            LOGGER.warning(
                "No API token configured; starting offline market data stream."
            )

            def factory() -> OfflineMarketDataStream:
                return OfflineMarketDataStream(
                    instruments=self._data_engine.active_instruments()
                    or self._config.datafeed.instruments
                    or self._config.datafeed.monitored_instruments,
                    depth=self._config.datafeed.depth,
                    dataset_path=self._config.backtest.dataset_path,
                )

            return factory

        def factory() -> MarketDataStream:
            return MarketDataStream(
                token=self._api_token,
                use_sandbox=self._config.datafeed.use_sandbox,
                instruments=self._data_engine.active_instruments(),
                depth=self._config.datafeed.depth,
            )

        return factory

    def _broker_factory(self) -> Callable[[], BrokerClient]:
        if not self._api_token:
            if self._config.system.mode == "production":
                raise RuntimeError("Broker token is required in production mode")

            LOGGER.warning(
                "No broker token configured; execution will remain in paper mode."
            )

            class _PaperBroker:
                async def __aenter__(self_inner):  # pragma: no cover - trivial async context
                    return self_inner

                async def __aexit__(self_inner, exc_type, exc, tb):  # pragma: no cover
                    return False

                async def place_order(self_inner, order):  # pragma: no cover - logging only
                    LOGGER.info(
                        "Skipping order placement for %s due to missing broker token", order.figi
                    )

            def factory():  # pragma: no cover - simple stub
                return _PaperBroker()

            return factory

        def factory() -> BrokerClient:
            return BrokerClient(
                self._api_token,
                sandbox=self._config.datafeed.use_sandbox,
                account_id=self._config.execution.account_id,
                orders_per_minute=self._config.risk.max_order_rate_per_minute,
            )

        return factory

    def _decrypt_token(self, token: Optional[str]) -> Optional[str]:
        if self._key_manager is None:
            return token
        return self._key_manager.maybe_decrypt(token)

    def _handle_latency(self, stage: str, latency_ms: float) -> None:
        self._metrics.record_latency(stage, latency_ms)
        alert = self._latency_monitor.observe(stage, latency_ms)
        if alert:
            self._latency_events.append(alert)

    def dashboard_status(self) -> DashboardStatus:
        data_snapshot = self._data_engine.snapshot()
        active = data_snapshot.get("active_instruments", [])
        monitored = data_snapshot.get("monitored_instruments", [])
        active_preview = ", ".join(active[:4])
        if len(active) > 4:
            active_preview += f" … (+{len(active) - 4})"
        elif not active_preview:
            active_preview = "No active instruments"
        modules: list[ModuleStatus] = [
            ModuleStatus(
                "Data engine",
                f"{len(active)} active",
                f"pool {len(monitored)}; {active_preview}",
                "warning" if not active else "normal",
                "running" if active else "inactive",
            )
        ]

        stream_state = "LIVE" if self._api_token else "OFFLINE"
        modules.append(
            ModuleStatus(
                "Market stream",
                stream_state,
                f"channel={self._connectivity.active_channel}",
                "warning" if not self._api_token else "normal",
                "running" if self._api_token else "inactive",
            )
        )

        session_state = self._session_guard.evaluate()
        session_detail = session_state.reason or "within window"
        modules.append(
            ModuleStatus(
                "Session schedule",
                "ACTIVE" if session_state.active else "PAUSED",
                session_detail,
                "warning" if not session_state.active else "normal",
                "running" if session_state.active else "inactive",
            )
        )

        manual_state = self._manual_override.status()
        override_detail = manual_state.reason or "no flag"
        if manual_state.expires_at:
            override_detail += f";expires={manual_state.expires_at.isoformat()}"
        modules.append(
            ModuleStatus(
                "Manual override",
                "HALTED" if manual_state.halted else "CLEAR",
                override_detail,
                "error" if manual_state.halted else "normal",
                "error" if manual_state.halted else "running",
            )
        )

        resource_snapshot = self._resource_monitor.snapshot()
        resource_severity = "normal"
        if (
            resource_snapshot.cpu_percent >= self._resource_monitor.cpu_threshold
            or resource_snapshot.memory_percent >= self._resource_monitor.memory_threshold
            or (
                resource_snapshot.gpu_memory_percent is not None
                and resource_snapshot.gpu_memory_percent
                >= self._resource_monitor.gpu_threshold
            )
        ):
            resource_severity = "warning"
        gpu_repr = (
            f"{resource_snapshot.gpu_memory_percent:.1f}%"
            if resource_snapshot.gpu_memory_percent is not None
            else "n/a"
        )
        modules.append(
            ModuleStatus(
                "Resources",
                "PRESSURE" if resource_severity == "warning" else "OK",
                f"cpu={resource_snapshot.cpu_percent:.1f}% mem={resource_snapshot.memory_percent:.1f}% gpu={gpu_repr}",
                resource_severity,
                "running",
            )
        )

        modules.append(
            ModuleStatus(
                "Feature pipeline",
                "READY",
                (
                    f"levels={self._config.features.lob_levels};"
                    f"window={self._config.features.rolling_window}"
                ),
                status="running",
            )
        )

        ml_device = self._ml_engine.device_mode.upper()
        ml_severity = (
            "warning"
            if self._ml_engine.device_mode != "gpu"
            or self._ml_engine.throttle_delay > 0
            else "normal"
        )
        modules.append(
            ModuleStatus(
                "ML engine",
                ml_device,
                f"throttle={self._ml_engine.throttle_delay * 1000:.0f}ms",
                ml_severity,
                "running" if self._ml_engine.device_mode else "inactive",
            )
        )

        fallback_state = "ENABLED" if self._config.fallback.enabled else "DISABLED"
        fallback_detail = f"quota_remaining={self._fallback.remaining_allowance()}"
        fallback_last = self._fallback.last_rejection
        if fallback_last:
            fallback_detail += f";last={fallback_last}"
        modules.append(
            ModuleStatus(
                "Fallback signals",
                fallback_state,
                fallback_detail,
                "warning" if not self._config.fallback.enabled else "normal",
                "running" if self._config.fallback.enabled else "inactive",
            )
        )

        risk_snapshot = self._risk_engine.snapshot()
        risk_state = "HALTED" if risk_snapshot.get("halt_trading") else "OK"
        risk_severity = "error" if risk_snapshot.get("halt_trading") else "normal"
        if risk_snapshot.get("daily_loss_triggered") and risk_severity != "error":
            risk_severity = "warning"
        risk_detail = (
            f"realized={risk_snapshot.get('realized_pnl', 0.0):.2f};"
            f"orders={risk_snapshot.get('order_count', 0)}"
        )
        cooldown = risk_snapshot.get("loss_cooldown_until")
        if cooldown:
            risk_detail += f";cooldown_until={cooldown}"
        modules.append(
            ModuleStatus(
                "Risk engine",
                risk_state,
                risk_detail,
                risk_severity,
                "error" if risk_state == "HALTED" else "running",
            )
        )

        exec_severity = "warning" if self._execution_engine.paper else "normal"
        exec_detail = "paper" if self._execution_engine.paper else "live"
        modules.append(
            ModuleStatus(
                "Execution",
                self._execution_engine.mode.upper(),
                f"mode={exec_detail}",
                exec_severity,
                "running" if not self._execution_engine.paper else "inactive",
            )
        )

        storage_detail = (
            f"signals={self._config.dashboard.repository_path.name};"
            f"history={self._config.storage.base_path}"
        )
        modules.append(
            ModuleStatus("Storage", "READY", storage_detail, status="running")
        )

        connectivity = self._connectivity.snapshot()
        modules.append(
            ModuleStatus(
                "Connectivity",
                connectivity.get("active_channel", "n/a"),
                (
                    f"failures={connectivity.get('failure_count', 0)};"
                    f"success_streak={connectivity.get('success_streak', 0)}"
                ),
                "warning" if connectivity.get("failure_count", 0) else "normal",
                "running",
            )
        )

        heartbeat_info = self._heartbeat.diagnostics()
        heartbeat_state = "ON" if heartbeat_info.get("enabled") else "OFF"
        last = heartbeat_info.get("last_beat")
        heartbeat_detail = (
            f"last={last.isoformat()}" if isinstance(last, datetime) else "no data"
        )
        modules.append(
            ModuleStatus(
                "Heartbeat",
                heartbeat_state,
                heartbeat_detail,
                "warning" if heartbeat_info.get("misses") else "normal",
                "running" if heartbeat_info.get("enabled") else "inactive",
            )
        )

        errors: list[str] = []
        if risk_snapshot.get("emergency_reason"):
            errors.append(f"Emergency halt: {risk_snapshot['emergency_reason']}")
        if risk_snapshot.get("calibration_required"):
            expiry = risk_snapshot.get("calibration_expiry")
            if expiry:
                errors.append(f"Calibration required until {expiry}")
            else:
                errors.append("Calibration required")
        if manual_state.halted:
            reason = manual_state.reason or "manual override"
            if manual_state.expires_at:
                reason += f" until {manual_state.expires_at.isoformat()}"
            errors.append(f"Manual override active: {reason}")
        if self._latency_events:
            alert = self._latency_events[-1]
            errors.append(
                (
                    f"Latency alert {alert.stage}: {alert.latency_ms:.1f}ms > "
                    f"{alert.threshold_ms:.1f}ms ({alert.severity})"
                )
            )
        heartbeat_failure = heartbeat_info.get("last_failure_reason")
        if heartbeat_failure:
            errors.append(f"Heartbeat failures: {heartbeat_failure}")
        if fallback_last:
            errors.append(f"Fallback rejection: {fallback_last}")

        processes: list[str] = []

        def describe_task(task: Optional[asyncio.Task], label: str) -> None:
            if task is None:
                return
            if task.cancelled():
                state = "cancelled"
            elif task.done():
                exc = task.exception()
                if exc:
                    errors.append(f"{label} failed: {exc}")
                    state = "error"
                else:
                    state = "finished"
            else:
                state = "running"
            processes.append(f"{label}: {state}")

        describe_task(self._checkpoint_task, "Checkpoint loop")
        describe_task(self._heartbeat_task, "Heartbeat monitor")
        describe_task(self._training_task, "Training scheduler")
        for task in self._risk_tasks:
            label = self._risk_task_labels.get(task, task.get_name() or "Risk task")
            describe_task(task, label)

        processes.append(
            f"Disaster recovery: {'enabled' if self._config.disaster_recovery.enabled else 'disabled'}"
        )
        processes.append(
            f"Performance reporter: {'enabled' if self._config.reporting.enabled else 'disabled'}"
        )

        return DashboardStatus(
            modules=modules,
            processes=processes,
            errors=errors,
            ensemble=self._ml_engine.ensemble_overview(),
        )

    def request_restart(self) -> None:
        """Signal that the orchestrator should restart after the current loop."""

        if not self._restart_callback_logged:
            LOGGER.warning("Dashboard requested orchestrator restart")
            self._restart_callback_logged = True
        self._pending_restart = True
        event = self._restart_event
        loop = self._loop
        if event is None:
            return
        if loop is None:
            event.set()
        else:
            loop.call_soon_threadsafe(event.set)

    def _restore_from_checkpoint(self) -> None:
        state = self._checkpoint_manager.load()
        if not state:
            return
        data_state = state.get("data")
        if isinstance(data_state, dict):
            self._data_engine.restore(data_state)
            LOGGER.info("Restored data engine state from checkpoint")
        risk_state = state.get("risk")
        if isinstance(risk_state, dict):
            self._risk_engine.restore(risk_state)
            LOGGER.info("Restored risk engine state from checkpoint")

    async def _prepare_initial_state(self) -> None:
        if not self._rest_api:
            return
        try:
            await self._rest_api.ping()
        except Exception as exc:  # pragma: no cover - network guard
            LOGGER.warning("Initial broker API ping failed: %s", exc)
            return
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        window_minutes = min(1440, max(1, self._config.datafeed.candle_history_size))
        start = now - timedelta(minutes=window_minutes)
        for figi in self._data_engine.active_instruments():
            try:
                order_book = await self._rest_api.fetch_order_book(
                    figi, depth=self._config.datafeed.depth
                )
            except Exception as exc:  # pragma: no cover - network guard
                LOGGER.warning("Failed to bootstrap order book for %s: %s", figi, exc)
            else:
                self._data_engine.ingest_order_book(order_book)
                with suppress(Exception):
                    self._historical_storage.store_order_book(order_book)
            try:
                trades = await self._rest_api.fetch_trades(
                    figi, limit=self._config.datafeed.trade_history_size
                )
                self._data_engine.ingest_trades(figi, trades)
                with suppress(Exception):
                    self._historical_storage.store_trades(trades)
            except Exception as exc:  # pragma: no cover - network guard
                LOGGER.debug("Failed to fetch trades for %s: %s", figi, exc)
            try:
                candles = await self._rest_api.fetch_candles(
                    figi,
                    start,
                    now,
                    self._config.datafeed.candle_interval,
                )
                self._data_engine.ingest_candles(figi, candles)
                with suppress(Exception):
                    self._historical_storage.store_candles(candles)
            except Exception as exc:  # pragma: no cover - network guard
                LOGGER.debug("Failed to fetch candles for %s: %s", figi, exc)
        try:
            portfolio = await self._rest_api.fetch_portfolio()
        except Exception as exc:  # pragma: no cover - network guard
            LOGGER.debug("Portfolio snapshot unavailable: %s", exc)
            return
        if not portfolio:
            return
        for position in portfolio.get("positions", []):
            figi = position.get("figi")
            quantity = int(position.get("quantity", 0) or 0)
            price = float(position.get("average_price", 0.0) or 0.0)
            if not figi or not quantity:
                continue
            try:
                self._risk_engine.update_position(figi, quantity, price)
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.debug("Failed to seed position for %s: %s", figi, exc)

    async def _training_scheduler_loop(self) -> None:
        schedule = self._config.training_schedule
        while schedule.enabled:
            now = datetime.utcnow()
            target = self._compute_next_training_time(now)
            await asyncio.sleep(max(1.0, (target - now).total_seconds()))
            await self._run_scheduled_training()

    def _compute_next_training_time(self, now: datetime) -> datetime:
        schedule = self._config.training_schedule
        hour, minute = (int(part) for part in schedule.daily_time.split(":", 1))
        for offset in range(0, 8):
            candidate = (
                now + timedelta(days=offset)
            ).replace(hour=hour, minute=minute, second=0, microsecond=0)
            if candidate <= now:
                continue
            if schedule.weekdays and candidate.weekday() not in schedule.weekdays:
                continue
            return candidate
        return now + timedelta(hours=24)

    async def _run_scheduled_training(self) -> None:
        schedule = self._config.training_schedule
        if not schedule.enabled:
            return
        if schedule.require_forward_window:
            session_state = self._session_guard.evaluate()
            if session_state.active:
                LOGGER.info("Skipping scheduled training while session is active")
                return
        trainer = ModelTrainer(self._config.training)
        try:
            report = await asyncio.to_thread(trainer.train)
        except Exception as exc:  # pragma: no cover - training failures
            LOGGER.exception("Scheduled training failed: %s", exc)
            self._audit_logger.log("TRAINING", "FAILED", str(exc))
            return
        self._ml_engine.update_weights(report.weights)
        self._audit_logger.log(
            "TRAINING",
            "COMPLETED",
            (
                f"samples={report.samples};epochs={report.epochs};"
                f"loss={report.training_loss:.4f};val={report.validation_loss:.4f}"
            ),
        )
        try:
            self.reload_models(report.model_dir)
        except Exception as exc:  # pragma: no cover - reload issues
            LOGGER.exception("Model reload after training failed: %s", exc)

    def _start_risk_tasks(self) -> None:
        if self._risk_schedule.rotation_interval_seconds > 0:
            task = asyncio.create_task(
                self._instrument_rotation_loop(
                    self._risk_schedule.rotation_interval_seconds
                ),
                name="risk_rotation",
            )
            self._risk_tasks.append(task)
            self._risk_task_labels[task] = "Instrument rotation"
        if self._risk_schedule.position_check_interval_seconds > 0:
            task = asyncio.create_task(
                self._position_management_loop(
                    self._risk_schedule.position_check_interval_seconds
                ),
                name="risk_position_checks",
            )
            self._risk_tasks.append(task)
            self._risk_task_labels[task] = "Position management"
        if self._risk_schedule.hedging_interval_seconds > 0:
            task = asyncio.create_task(
                self._hedging_loop(self._risk_schedule.hedging_interval_seconds),
                name="risk_hedging",
            )
            self._risk_tasks.append(task)
            self._risk_task_labels[task] = "Hedging loop"
        if self._risk_schedule.drift_check_interval_seconds > 0:
            task = asyncio.create_task(
                self._drift_review_loop(
                    self._risk_schedule.drift_check_interval_seconds
                ),
                name="risk_drift_checks",
            )
            self._risk_tasks.append(task)
            self._risk_task_labels[task] = "Drift review"

    def _collect_risk_snapshots(self) -> dict[str, dict[str, float]]:
        snapshots: dict[str, dict[str, float]] = {}
        for figi in self._data_engine.active_instruments():
            book = self._data_engine.get_order_book(figi)
            if not book:
                continue
            snapshots[figi] = {
                "price": book.mid_price(),
                "spread": book.spread(),
                "atr": self._data_engine.atr(figi, periods=5),
            }
        return snapshots

    async def _instrument_rotation_loop(self, interval: int) -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                replacements = self._data_engine.rotate_instruments()
                if replacements:
                    detail = ",".join(f"{old}->{new}" for old, new in replacements)
                    LOGGER.info("Scheduled rotation replaced instruments: %s", detail)
                    self._audit_logger.log("RISK", "ROTATION", detail)
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            raise

    async def _position_management_loop(self, interval: int) -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                snapshots = self._collect_risk_snapshots()
                if not snapshots:
                    continue
                adjustments = self._risk_engine.manage_positions(snapshots)
                for adjustment in adjustments:
                    self._record_position_adjustment(adjustment)
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            raise

    async def _hedging_loop(self, interval: int) -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                snapshots = self._collect_risk_snapshots()
                if not snapshots:
                    continue
                prices = {figi: data["price"] for figi, data in snapshots.items()}
                adjustments = self._risk_engine.hedge_portfolio(prices)
                for adjustment in adjustments:
                    self._record_position_adjustment(adjustment)
                liquidity = self._risk_engine.forecast_liquidity(snapshots)
                self._metrics.record_liquidity(liquidity)
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            raise

    async def _drift_review_loop(self, interval: int) -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                for figi in self._data_engine.active_instruments():
                    history = self._data_engine.history(figi)
                    if len(history) < 2:
                        continue
                    reference = [level.price for level in history[0].bids]
                    current = [level.price for level in history[-1].bids]
                    report = self._drift_detector.evaluate(reference, current)
                    if report.triggered:
                        self._risk_engine.record_drift(report)
                        self._audit_logger.log(
                            "RISK",
                            "DRIFT_REVIEW",
                            f"figi={figi};severity={report.severity};p={report.p_value:.4f}",
                        )
                        if report.alert:
                            await self._notifications.notify_circuit_breaker(figi, report.severity)
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            raise

    def _record_position_adjustment(self, adjustment: PositionAdjustment) -> None:
        detail = (
            f"figi={adjustment.figi};action={adjustment.action};"
            f"qty={adjustment.quantity};reason={adjustment.reason}"
        )
        LOGGER.info("Risk adjustment: %s", detail)
        self._audit_logger.log("RISK", f"POSITION_{adjustment.action.upper()}", detail)

    async def _await_startup_window(self) -> None:
        start = self._config.system.startup_time
        if not start:
            return
        try:
            hour, minute = (int(part) for part in start.split(":", 1))
        except ValueError:
            LOGGER.warning("Invalid startup_time format: %s", start)
            return
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now >= target:
            return
        wait_seconds = (target - now).total_seconds()
        LOGGER.info("Waiting %.0f seconds for startup window", wait_seconds)
        await asyncio.sleep(wait_seconds)

    def _snapshot_state(self) -> dict:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": self._config.system.mode,
            "risk": self._risk_engine.snapshot(),
            "data": self._data_engine.snapshot(),
            "connectivity": self._connectivity.snapshot(),
        }

    async def _persist_checkpoint(self) -> None:
        state = self._snapshot_state()
        await asyncio.to_thread(self._checkpoint_manager.persist, state)

    async def _checkpoint_loop(self, interval: int) -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                await self._persist_checkpoint()
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            raise

    async def _process_latency_alerts(self) -> None:
        while self._latency_events:
            alert = self._latency_events.popleft()
            LOGGER.warning(
                "Latency %s breach stage=%s latency=%.2f threshold=%.2f",
                alert.severity,
                alert.stage,
                alert.latency_ms,
                alert.threshold_ms,
            )
            self._audit_logger.log(
                "RESOURCE",
                "LATENCY",
                f"stage={alert.stage};latency={alert.latency_ms:.2f};threshold={alert.threshold_ms:.2f};severity={alert.severity}",
            )
            if alert.severity == "critical":
                self._risk_engine.trigger_emergency_halt(f"latency:{alert.stage}")
            await self._notifications.notify_latency_violation(
                alert.stage,
                alert.latency_ms,
                alert.threshold_ms,
                alert.severity,
            )

    async def _handle_heartbeat_timeout(self, status: HeartbeatStatus) -> None:
        reason = status.last_failure_reason or "no data"
        last_seen = status.last_beat.isoformat() if status.last_beat else "unknown"
        context = status.last_context or "unknown"
        LOGGER.error(
            "Heartbeat missed intervals=%s elapsed=%.2fs context=%s reason=%s",
            status.missed_intervals,
            status.elapsed_seconds,
            context,
            reason,
        )
        self._audit_logger.log(
            "NETWORK",
            "HEARTBEAT_MISSED",
            f"misses={status.missed_intervals};last={last_seen};context={context};reason={reason}",
        )
        self._risk_engine.trigger_emergency_halt("heartbeat")
        await self._notifications.notify_heartbeat_missed(
            status.missed_intervals,
            status.last_beat,
            reason,
        )

    async def _process_manual_override(self) -> bool:
        status = self._manual_override.status()
        if status.halted:
            reason = status.reason or "manual override"
            if not self._manual_override_active or reason != self._manual_override_reason:
                LOGGER.warning("Manual override active: %s", reason)
                self._audit_logger.log("CONTROL", "HALTED", reason)
                await self._notifications.notify_manual_override(reason, status.expires_at)
            self._manual_override_active = True
            self._manual_override_reason = reason
            await asyncio.sleep(self._manual_override.poll_interval)
            return True
        if self._manual_override_active:
            LOGGER.info("Manual override cleared")
            self._audit_logger.log("CONTROL", "RESUMED", "manual override cleared")
            await self._notifications.notify_manual_override_cleared()
        self._manual_override_active = False
        self._manual_override_reason = None
        return False

    async def _process_session_schedule(self) -> bool:
        state = self._session_guard.evaluate()
        if state.changed:
            if state.active:
                next_pause = (
                    state.next_transition.isoformat()
                    if state.next_transition
                    else "unknown"
                )
                self._audit_logger.log(
                    "CONTROL",
                    "SESSION_RESUMED",
                    f"next_pause={next_pause}",
                )
                await self._notifications.notify_session_resumed(
                    state.next_transition
                )
            else:
                reason = state.reason or "schedule"
                resume_at = (
                    state.next_transition.isoformat()
                    if state.next_transition
                    else "unspecified"
                )
                self._audit_logger.log(
                    "CONTROL",
                    "SESSION_SUSPENDED",
                    f"reason={reason};resume_at={resume_at}",
                )
                await self._notifications.notify_session_suspended(
                    reason,
                    state.next_transition,
                )
        if not state.active:
            await asyncio.sleep(0)
            return False
        return True

    async def _handle_stream_failure(self, exc: Exception) -> None:
        reason = str(exc)
        self._audit_logger.log(
            "NETWORK", "STREAM_ERROR", f"channel={self._connectivity.active_channel};reason={reason}"
        )
        self._heartbeat.record_failure(reason)
        event, delay = self._connectivity.record_failure(reason)
        if event and event.kind == "failover":
            LOGGER.error(
                "Connectivity failover to %s due to %s", event.channel, reason
            )
            self._audit_logger.log(
                "NETWORK", "FAILOVER", f"channel={event.channel};reason={reason}"
            )
            self._data_engine.resynchronise()
            await self._notifications.notify_connectivity_failover(event.channel, reason)
        if delay > 0:
            await asyncio.sleep(delay)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._restart_event = asyncio.Event()
        if self._pending_restart:
            self._restart_event.set()
        self._start_dashboard_if_needed()
        await self._await_startup_window()
        interval = int(self._config.system.checkpoint_interval_seconds)
        if interval > 0:
            self._checkpoint_task = asyncio.create_task(self._checkpoint_loop(interval))
        if self._heartbeat.enabled:
            poll = max(self._config.monitoring.heartbeat_interval_seconds / 2, 0.5)
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat.monitor(
                    self._handle_heartbeat_timeout,
                    poll_interval=poll,
                )
            )
        if self._config.training_schedule.enabled:
            self._training_task = asyncio.create_task(self._training_scheduler_loop())
        self._start_risk_tasks()
        await self._prepare_initial_state()
        stream_factory = self._market_stream_factory()
        async def integrity_probe() -> bool:
            return await check_data_integrity(self._data_engine)

        try:
            async for order_book in iterate_stream(
                stream_factory,
                delay=self._config.datafeed.reconnect_delay,
                integrity_check=integrity_probe,
                on_failure=self._handle_stream_failure,
                stop_event=self._restart_event,
            ):
                await self._process_latency_alerts()
                snapshot, mitigations = self._resource_monitor.check_thresholds()
                if mitigations:
                    LOGGER.warning(
                        "Resource pressure detected cpu=%.1f mem=%.1f gpu=%s actions=%s",
                        snapshot.cpu_percent,
                        snapshot.memory_percent,
                        f"{snapshot.gpu_memory_percent:.1f}"
                        if snapshot.gpu_memory_percent is not None
                        else "n/a",
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
                with suppress(Exception):
                    self._historical_storage.store_order_book(order_book)
                self._heartbeat.record_beat(order_book.figi)
                recovery = self._connectivity.record_success()
                if recovery and recovery.kind == "recovery":
                    LOGGER.info("Connectivity restored on %s", recovery.channel)
                    self._audit_logger.log(
                        "NETWORK", "RECOVERED", f"channel={recovery.channel}"
                    )
                    self._data_engine.resynchronise()
                    await self._notifications.notify_connectivity_recovered(
                        recovery.channel
                    )
                if await self._process_manual_override():
                    continue
                if not await self._process_session_schedule():
                    continue
                mid_price = order_book.mid_price()
                spread = order_book.spread()
                atr = self._data_engine.atr(order_book.figi, periods=5)
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
                with timed("features", self._handle_latency):
                    market = self._data_engine.market_indicators()
                    features = self._feature_pipeline.transform(window, market=market)
                with suppress(Exception):
                    self._historical_storage.store_features(features)
                signals: list[MLSignal] = []
                try:
                    with timed("ml", self._handle_latency):
                        signals = self._ml_engine.infer([features])
                except Exception as exc:
                    context = f"exception:{type(exc).__name__}"
                    LOGGER.exception("ML inference failed; attempting fallback: %s", exc)
                    fallback = self._fallback.generate(window, reason=context)
                    if fallback is None:
                        detail = self._fallback.last_rejection or "unavailable"
                        self._audit_logger.log(
                            "ML",
                            "FALLBACK_UNAVAILABLE",
                            f"figi={order_book.figi};reason={context};detail={detail}",
                        )
                        continue
                    remaining = self._fallback.remaining_allowance()
                    self._audit_logger.log(
                        "ML",
                        "FALLBACK_SIGNAL",
                        (
                            f"figi={fallback.signal.figi};reason={fallback.reason};"
                            f"imbalance={fallback.imbalance:.3f};volatility={fallback.volatility:.4f};"
                            f"quota_remaining={remaining}"
                        ),
                    )
                    await self._notifications.notify_fallback_signal(
                        fallback.signal.figi,
                        fallback.reason,
                        fallback.signal.confidence,
                    )
                    signals = [fallback.signal]
                failover_detail = self._ml_engine.drain_failover_event()
                if failover_detail:
                    mode = self._ml_engine.device_mode
                    LOGGER.error("GPU failure detected, running in %s mode", mode)
                    self._audit_logger.log(
                        "RESOURCE", "GPU_FAILOVER", f"mode={mode};detail={failover_detail}"
                    )
                    await self._notifications.notify_gpu_failover(mode, failover_detail)
                delay = self._ml_engine.throttle_delay
                if delay > 0:
                    await asyncio.sleep(delay)
                if not signals:
                    fallback = self._fallback.generate(window, reason="no_signal")
                    if fallback is None:
                        detail = self._fallback.last_rejection or "no_fallback"
                        self._audit_logger.log(
                            "ML",
                            "NO_SIGNAL",
                            f"figi={order_book.figi};detail={detail}",
                        )
                        continue
                    remaining = self._fallback.remaining_allowance()
                    self._audit_logger.log(
                        "ML",
                        "FALLBACK_SIGNAL",
                        (
                            f"figi={fallback.signal.figi};reason={fallback.reason};"
                            f"imbalance={fallback.imbalance:.3f};volatility={fallback.volatility:.4f};"
                            f"quota_remaining={remaining}"
                        ),
                    )
                    await self._notifications.notify_fallback_signal(
                        fallback.signal.figi,
                        fallback.reason,
                        fallback.signal.confidence,
                    )
                    signals = [fallback.signal]
                for signal in signals:
                    self._metrics.record_signal(signal.figi, signal.confidence)
                    self._repository.persist_signal(signal.figi, signal.direction, signal.confidence)
                    with timed("risk", self._handle_latency):
                        plan = self._risk_engine.evaluate_signal(
                            signal,
                            mid_price,
                            spread=spread,
                            atr=atr,
                            market_volatility=market.market_volatility,
                        )
                    if not plan:
                        LOGGER.info("Signal rejected by risk engine")
                        self._audit_logger.log(
                            "ORDER",
                            "REJECTED",
                            f"figi={signal.figi};confidence={signal.confidence:.3f}",
                        )
                        continue
                    report = await self._execution_engine.execute_plan(
                        signal, plan, mid_price
                    )
                    if report.accepted:
                        LOGGER.info("Order executed for %s", signal.figi)
                        self._audit_logger.log(
                            "ORDER",
                            "EXECUTED",
                            (
                                f"figi={signal.figi};confidence={signal.confidence:.3f};"
                                f"qty={plan.quantity};strategy={plan.strategy};stop={plan.stop_loss:.2f}"
                            ),
                        )
                        await self._notifications.notify_order_filled(
                            signal.figi, order_book.mid_price(), signal.confidence
                        )
                    else:
                        LOGGER.warning("Order not executed: %s", report.reason)
                        self._audit_logger.log(
                            "ORDER", "FAILED", f"figi={signal.figi};reason={report.reason}"
                        )
            risk_metrics = self._risk_engine.metrics({order_book.figi: order_book.mid_price()})
            self._metrics.record_risk(order_book.figi, risk_metrics.var_value)
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
            await self._performance_reporter.maybe_emit()
            await self._disaster_recovery.maybe_replicate()
        finally:
            if self._checkpoint_task:
                self._checkpoint_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._checkpoint_task
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._heartbeat_task
            if self._training_task:
                self._training_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._training_task
            for task in self._risk_tasks:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                self._risk_task_labels.pop(task, None)
            await self._process_latency_alerts()
            await self._persist_checkpoint()
            await self._performance_reporter.maybe_emit()
        if self._restart_event and self._restart_event.is_set():
            raise RestartRequested()

    def reload_models(self, model_dir: Path) -> None:
        """Reload quantised models and clear cached feature state."""

        self._feature_pipeline.reset_cache()
        self._ml_engine.reload_models(model_dir)
        self._risk_engine.notify_model_reload()

    def _start_dashboard_if_needed(self) -> None:
        if self._dashboard_thread is not None:
            return
        if not self._config.dashboard.auto_start:
            return
        try:
            thread = run_dashboard(
                self._config.dashboard.repository_path,
                status_provider=self.dashboard_status,
                refresh_interval_ms=self._config.dashboard.refresh_interval_ms,
                signal_limit=self._config.dashboard.signal_limit,
                headless=self._config.dashboard.headless,
                title=self._config.dashboard.title,
                background=True,
                config_path=self._config_path,
                restart_callback=self.request_restart,
            )
        except OSError as exc:
            LOGGER.error(
                "Failed to launch dashboard UI: %s", exc
            )
            self._audit_logger.log(
                "UI",
                "DASHBOARD_ERROR",
                f"error={exc}",
            )
            return
        if thread is None:
            LOGGER.warning("Dashboard launcher returned no thread; UI not started")
            return
        self._dashboard_thread = thread
        LOGGER.info("Dashboard UI started with Tkinter renderer")
        self._audit_logger.log("UI", "DASHBOARD_STARTED", "renderer=tkinter")


def run_from_yaml(path: str | Path) -> None:
    config_path = Path(path).expanduser()
    first_iteration = True

    while True:
        config = load_config(config_path)
        level = getattr(logging, config.logging.level)
        if first_iteration:
            logging.basicConfig(level=level)
            first_iteration = False
        else:
            logging.getLogger().setLevel(level)

        token = (
            config.datafeed.sandbox_token
            if config.datafeed.use_sandbox
            else config.datafeed.production_token
        )
        token_missing = is_placeholder_token(token)
        allow_tokenless = bool(getattr(config.datafeed, "allow_tokenless", False))
        requires_sdk = (
            config.system.mode == "production"
            or not allow_tokenless
            or not token_missing
        )

        if requires_sdk:
            try:
                ensure_sdk_available()
            except TinkoffSDKUnavailable as exc:
                LOGGER.error(
                    "Cannot start orchestrator without the tinkoff-investments SDK: %s",
                    exc,
                )
                LOGGER.error(
                    "Install the bundled wheel via scripts/install_vendor.py or pip before running."
                )
                raise SystemExit(1) from exc
        else:
            try:
                ensure_sdk_available()
            except TinkoffSDKUnavailable:
                LOGGER.warning(
                    "tinkoff-investments SDK not installed; continuing in offline mode."
                )
            else:
                LOGGER.info(
                    "tinkoff-investments SDK detected; offline mode may stream cached data."
                )

        orchestrator = Orchestrator(config, config_path=config_path)
        try:
            asyncio.run(orchestrator.run())
        except RestartRequested:
            LOGGER.info("Restart requested by dashboard; reloading configuration.")
            continue
        break


__all__ = ["Orchestrator", "run_from_yaml"]
