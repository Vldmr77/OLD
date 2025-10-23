"""Order execution logic."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional

from ..ml.engine import MLSignal
from ..risk.engine import OrderPlan, RiskEngine
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
    def __init__(self, broker_factory, risk_engine: RiskEngine, *, mode: str = "production") -> None:
        self._broker_factory = broker_factory
        self._risk_engine = risk_engine
        self._mode: Literal["production", "forward-test", "development"] = (
            mode if mode in {"production", "forward-test", "development"} else "production"
        )
        self._paper = self._mode != "production"
        self._last_broker_latency_ms: Optional[float] = None
        if not self._paper and (OrderDirection is None or OrderType is None):
            raise RuntimeError("tinkoff-investments SDK is required for execution")

    async def execute_plan(
        self, signal: MLSignal, plan: OrderPlan, price: float
    ) -> ExecutionReport:
        if self._paper:
            self._last_broker_latency_ms = None
            self._risk_engine.register_order(plan.slices)
            self._risk_engine.update_position(
                signal.figi, signal.direction * plan.quantity, price, plan.stop_loss
            )
            return ExecutionReport(figi=signal.figi, accepted=True)
        start = time.perf_counter()
        direction = (
            OrderDirection.ORDER_DIRECTION_BUY
            if signal.direction > 0
            else OrderDirection.ORDER_DIRECTION_SELL
        )
        async with self._broker_factory() as broker:
            try:
                if plan.strategy == "vwap":
                    await self._execute_vwap(broker, signal, plan, price, direction)
                else:
                    await self._execute_ioc(broker, signal, plan, price, direction)
                self._risk_engine.register_order(plan.slices)
                self._risk_engine.update_position(
                    signal.figi,
                    signal.direction * plan.quantity,
                    price,
                    plan.stop_loss,
                )
                return ExecutionReport(figi=signal.figi, accepted=True)
            except Exception as exc:  # pragma: no cover - defensive branch
                LOGGER.exception("Order placement failed: %s", exc)
                return ExecutionReport(figi=signal.figi, accepted=False, reason=str(exc))
            finally:
                self._last_broker_latency_ms = (time.perf_counter() - start) * 1000.0

    async def cancel_all_orders(self) -> int:
        if self._paper:
            LOGGER.info("Paper mode active; no broker orders to cancel")
            return 0
        async with self._broker_factory() as broker:
            try:
                return await broker.cancel_all_orders()
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.exception("Failed to cancel orders: %s", exc)
                return 0

    async def _execute_ioc(
        self,
        broker: BrokerClient,
        signal: MLSignal,
        plan: OrderPlan,
        price: float,
        direction,
    ) -> None:
        order = OrderRequest(
            figi=signal.figi,
            quantity=plan.quantity,
            price=price,
            direction=direction,
            order_type=OrderType.ORDER_TYPE_MARKET,
        )
        await broker.place_order(order)

    async def _execute_vwap(
        self,
        broker: BrokerClient,
        signal: MLSignal,
        plan: OrderPlan,
        price: float,
        direction,
    ) -> None:
        aggressive_qty = max(1, int(round(plan.quantity * plan.aggressiveness)))
        aggressive_qty = min(plan.quantity, aggressive_qty)
        passive_qty = plan.quantity - aggressive_qty
        market_order = OrderRequest(
            figi=signal.figi,
            quantity=aggressive_qty,
            price=price,
            direction=direction,
            order_type=OrderType.ORDER_TYPE_MARKET,
        )
        await broker.place_order(market_order)
        if passive_qty <= 0:
            return
        price_adjustment = 0.0005 * price
        limit_price = price + price_adjustment if signal.direction > 0 else price - price_adjustment
        limit_order = OrderRequest(
            figi=signal.figi,
            quantity=passive_qty,
            price=limit_price,
            direction=direction,
            order_type=OrderType.ORDER_TYPE_LIMIT,
        )
        await broker.place_order(limit_order)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def paper(self) -> bool:
        return self._paper

    def set_broker_factory(self, factory) -> None:
        self._broker_factory = factory

    @property
    def last_broker_latency_ms(self) -> Optional[float]:
        return self._last_broker_latency_ms


__all__ = ["ExecutionEngine", "ExecutionReport"]
