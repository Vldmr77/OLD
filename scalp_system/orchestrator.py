"""Main orchestrator for the scalping system."""
from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Optional

from .config.base import OrchestratorConfig
from .config.loader import load_config
from .control.manual_override import ManualOverrideGuard
from .control.session import SessionGuard
from .data.engine import DataEngine
from .data.streams import MarketDataStream, iterate_stream
from .execution.client import BrokerClient
from .execution.executor import ExecutionEngine
from .features.pipeline import FeaturePipeline
from .ml.engine import MLSignal, MLEngine
from .ml.fallback import FallbackSignalGenerator
from .ml.calibration import CalibrationCoordinator
from .monitoring.audit import AuditLogger
from .monitoring.drift import DriftDetector
from .monitoring.latency import LatencyAlert, LatencyMonitor
from .monitoring.metrics import MetricsRegistry
from .monitoring.heartbeat import HeartbeatMonitor, HeartbeatStatus
from .monitoring.notifications import NotificationDispatcher
from .monitoring.reporting import PerformanceReporter
from .monitoring.resource import ResourceMonitor
from .network import ConnectivityFailover
from .risk.engine import RiskEngine
from .security import KeyManager
from .storage.checkpoint import CheckpointManager
from .storage.disaster_recovery import DisasterRecoveryManager
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
        self._data_engine.update_monitored_pool(config.datafeed.monitored_instruments)
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

    def _handle_latency(self, stage: str, latency_ms: float) -> None:
        self._metrics.record_latency(stage, latency_ms)
        alert = self._latency_monitor.observe(stage, latency_ms)
        if alert:
            self._latency_events.append(alert)

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
        stream_factory = self._market_stream_factory()
        broker_factory = self._broker_factory()
        rotation_counter = 0
        async def integrity_probe() -> bool:
            return await check_data_integrity(self._data_engine)

        try:
            async for order_book in iterate_stream(
                stream_factory,
                delay=self._config.datafeed.reconnect_delay,
                integrity_check=integrity_probe,
                on_failure=self._handle_stream_failure,
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
                    report = await ExecutionEngine(
                        broker_factory, self._risk_engine
                    ).execute_plan(signal, plan, mid_price)
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
            await self._process_latency_alerts()
            await self._persist_checkpoint()
            await self._performance_reporter.maybe_emit()

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
