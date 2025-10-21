"""Risk management subsystem."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

from ..config.base import RiskLimits
from ..ml.engine import MLSignal


@dataclass
class Position:
    figi: str
    quantity: int = 0
    average_price: float = 0.0


@dataclass
class RiskMetrics:
    timestamp: datetime
    unrealized_pnl: float
    gross_exposure: float
    var_value: float


class RiskEngine:
    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits
        self._positions: Dict[str, Position] = {}
        self._order_count: int = 0
        self._last_reset = datetime.utcnow()

    def evaluate_signal(self, signal: MLSignal, price: float) -> bool:
        position = self._positions.get(signal.figi, Position(figi=signal.figi))
        projected_qty = position.quantity + signal.direction
        if abs(projected_qty) > self._limits.max_position:
            return False
        projected_exposure = abs(projected_qty * price)
        if projected_exposure > self._limits.max_gross_exposure:
            return False
        if self._order_count >= self._limits.max_order_rate_per_minute:
            return False
        return True

    def register_order(self) -> None:
        now = datetime.utcnow()
        if (now - self._last_reset).total_seconds() > 60:
            self._order_count = 0
            self._last_reset = now
        self._order_count += 1

    def update_position(self, figi: str, quantity_delta: int, price: float) -> None:
        position = self._positions.setdefault(figi, Position(figi=figi))
        new_qty = position.quantity + quantity_delta
        if new_qty == 0:
            position.quantity = 0
            position.average_price = 0.0
        else:
            position.average_price = (
                (position.average_price * position.quantity + price * quantity_delta) / new_qty
            )
            position.quantity = new_qty

    def metrics(self, market_prices: Dict[str, float]) -> RiskMetrics:
        unrealized = 0.0
        gross = 0.0
        prices = []
        for position in self._positions.values():
            price = market_prices.get(position.figi, position.average_price)
            prices.append(price)
            gross += abs(position.quantity * price)
            unrealized += (price - position.average_price) * position.quantity
        if not prices:
            prices = [0.0]
        mean_price = sum(prices) / len(prices)
        variance = sum((price - mean_price) ** 2 for price in prices) / len(prices)
        var_value = math.sqrt(variance) * 2 * self._limits.var_horizon_minutes
        return RiskMetrics(
            timestamp=datetime.utcnow(),
            unrealized_pnl=unrealized,
            gross_exposure=gross,
            var_value=var_value,
        )


__all__ = ["RiskEngine", "RiskMetrics", "Position"]
