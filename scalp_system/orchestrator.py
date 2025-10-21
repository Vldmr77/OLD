"""Main orchestrator for the scalping system."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from .config.base import OrchestratorConfig
from .config.loader import load_config
from .data.streams import MarketDataStream, RollingOrderBookBuffer, iterate_stream
from .execution.client import BrokerClient
from .execution.executor import ExecutionEngine
from .features.pipeline import FeaturePipeline
from .ml.engine import MLEngine
from .monitoring.drift import DriftDetector
from .monitoring.metrics import MetricsRegistry
from .risk.engine import RiskEngine
from .storage.repository import SQLiteRepository
from .utils.timing import timed

LOGGER = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config
        self._feature_pipeline = FeaturePipeline(config.features)
        self._ml_engine = MLEngine(config.ml)
        self._risk_engine = RiskEngine(config.risk)
        self._metrics = MetricsRegistry()
        self._repository = SQLiteRepository(config.storage.base_path / "signals.db")
        self._drift_detector = DriftDetector(
            threshold=config.ml.drift_threshold,
            history_path=config.storage.base_path / "drift_metrics.log",
        )

    def _market_stream_factory(self) -> Callable[[], MarketDataStream]:
        token = (
            self._config.datafeed.sandbox_token
            if self._config.datafeed.use_sandbox
            else self._config.datafeed.production_token
        )
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
        if not token:
            raise RuntimeError("Broker token is required")

        def factory() -> BrokerClient:
            return BrokerClient(token, sandbox=self._config.datafeed.use_sandbox, account_id=self._config.execution.account_id)

        return factory

    async def run(self) -> None:
        buffer = RollingOrderBookBuffer(maxlen=self._config.features.lob_levels)
        stream_factory = self._market_stream_factory()
        broker_factory = self._broker_factory()
        async for order_book in iterate_stream(stream_factory):
            buffer.append(order_book)
            if not buffer.is_ready():
                continue
            with timed("features", self._metrics.record_latency):
                features = self._feature_pipeline.transform(buffer.window())
            with timed("ml", self._metrics.record_latency):
                signals = self._ml_engine.infer([features])
            for signal in signals:
                self._metrics.record_signal(signal.figi, signal.confidence)
                self._repository.persist_signal(signal.figi, signal.direction, signal.confidence)
                if not self._risk_engine.evaluate_signal(signal, order_book.mid_price()):
                    LOGGER.info("Signal rejected by risk engine")
                    continue
                report = await ExecutionEngine(broker_factory, self._risk_engine).execute_signal(
                    signal, order_book.mid_price()
                )
                if report.accepted:
                    LOGGER.info("Order executed for %s", signal.figi)
                else:
                    LOGGER.warning("Order not executed: %s", report.reason)
            drift = self._drift_detector.evaluate(
                reference=[level.price for level in buffer.window()[0].bids],
                current=[level.price for level in order_book.bids],
            )
            if drift.triggered:
                LOGGER.warning("Data drift detected p=%.4f mean_diff=%.4f", drift.p_value, drift.mean_diff)


def run_from_yaml(path: str | Path) -> None:
    config = load_config(path)
    logging.basicConfig(level=getattr(logging, config.logging.level))
    orchestrator = Orchestrator(config)
    asyncio.run(orchestrator.run())


__all__ = ["Orchestrator", "run_from_yaml"]
