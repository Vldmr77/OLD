"""Order execution logic."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ..ml.engine import MLSignal
from ..risk.engine import RiskEngine
from .client import BrokerClient, OrderRequest

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from tinkoff.invest import OrderDirection, OrderType
except ImportError:  # pragma: no cover
    OrderDirection = None  # type: ignore
    OrderType = None  # type: ignore


@dataclass
class ExecutionReport:
    figi: str
    accepted: bool
    reason: Optional[str] = None


class ExecutionEngine:
    def __init__(self, broker_factory, risk_engine: RiskEngine) -> None:
        if OrderDirection is None or OrderType is None:
            raise RuntimeError("tinkoff-investments SDK is required for execution")
        self._broker_factory = broker_factory
        self._risk_engine = risk_engine

    async def execute_signal(self, signal: MLSignal, price: float) -> ExecutionReport:
        if not self._risk_engine.evaluate_signal(signal, price):
            return ExecutionReport(figi=signal.figi, accepted=False, reason="Risk check failed")
        direction = OrderDirection.ORDER_DIRECTION_BUY if signal.direction > 0 else OrderDirection.ORDER_DIRECTION_SELL
        order = OrderRequest(
            figi=signal.figi,
            quantity=1,
            price=price,
            direction=direction,
            order_type=OrderType.ORDER_TYPE_MARKET,
        )
        async with self._broker_factory() as broker:
            try:
                await broker.place_order(order)
                self._risk_engine.register_order()
                self._risk_engine.update_position(signal.figi, signal.direction, price)
                return ExecutionReport(figi=signal.figi, accepted=True)
            except Exception as exc:  # pragma: no cover - defensive branch
                LOGGER.exception("Order placement failed: %s", exc)
                return ExecutionReport(figi=signal.figi, accepted=False, reason=str(exc))


__all__ = ["ExecutionEngine", "ExecutionReport"]
