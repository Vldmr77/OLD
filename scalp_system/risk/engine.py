"""Risk management subsystem."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..config.base import RiskLimits
from ..ml.engine import MLSignal
from ..monitoring.drift import DriftReport


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
        self._session_start = datetime.utcnow()
        self._realized_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._loss_cooldown_until: Optional[datetime] = None
        self._daily_loss_triggered = False
        self._halt_due_to_drift = False
        self._last_drift: Optional[DriftReport] = None
        self._halt_trading = False
        self._calibration_required = False
        self._calibration_expiry: Optional[datetime] = None
        self._emergency_reason: Optional[str] = None

    def evaluate_signal(self, signal: MLSignal, price: float) -> bool:
        self._reset_if_new_day()
        if self._loss_cooldown_until and datetime.utcnow() >= self._loss_cooldown_until:
            self._loss_cooldown_until = None
        if self._halt_trading:
            return False
        if self._loss_cooldown_until and datetime.utcnow() < self._loss_cooldown_until:
            return False
        if (
            self._limits.max_daily_loss > 0
            and self._daily_loss_triggered
            and self._realized_pnl <= -self._limits.max_daily_loss
        ):
            return False
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
        self._reset_if_new_day()
        now = datetime.utcnow()
        if (now - self._last_reset).total_seconds() > 60:
            self._order_count = 0
            self._last_reset = now
        self._order_count += 1

    def update_position(self, figi: str, quantity_delta: int, price: float) -> None:
        self._reset_if_new_day()
        position = self._positions.setdefault(figi, Position(figi=figi))
        prev_qty = position.quantity
        prev_avg = position.average_price
        if prev_qty != 0 and quantity_delta != 0 and prev_qty * quantity_delta < 0:
            closed_qty = min(abs(prev_qty), abs(quantity_delta))
            sign = 1 if prev_qty > 0 else -1
            realized = (price - prev_avg) * closed_qty * sign
            self._update_loss_state(realized)
            remaining = prev_qty + quantity_delta
            if remaining == 0:
                position.quantity = 0
                position.average_price = 0.0
                return
            if prev_qty * remaining > 0:
                position.quantity = remaining
                position.average_price = prev_avg
                return
            quantity_delta = remaining
            prev_qty = 0
            prev_avg = 0.0

        new_qty = prev_qty + quantity_delta
        if new_qty == 0:
            position.quantity = 0
            position.average_price = 0.0
        elif prev_qty == 0:
            position.quantity = new_qty
            position.average_price = price
        else:
            position.average_price = (
                (prev_avg * prev_qty + price * quantity_delta) / new_qty
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

    def record_drift(self, report: DriftReport) -> None:
        self._last_drift = report
        if report.alert:
            self._halt_trading = True
            self._halt_due_to_drift = True
        elif report.calibrate:
            self._calibration_required = True
            self._calibration_expiry = datetime.utcnow() + timedelta(minutes=15)

    def calibration_required(self) -> bool:
        if not self._calibration_required:
            return False
        if self._calibration_expiry and datetime.utcnow() > self._calibration_expiry:
            self._calibration_required = False
            self._calibration_expiry = None
            return False
        return True

    def acknowledge_calibration(self) -> None:
        self._calibration_required = False
        self._calibration_expiry = None

    def last_drift(self) -> Optional[DriftReport]:
        return self._last_drift

    def trading_halted(self) -> bool:
        return self._halt_trading

    def notify_model_reload(self) -> None:
        """Resume trading safeguards after successful model reload."""

        self._halt_due_to_drift = False
        if not self._daily_loss_triggered:
            self._halt_trading = False
        self._calibration_required = False
        self._calibration_expiry = None
        self._emergency_reason = None

    def trigger_emergency_halt(self, reason: str) -> None:
        self._halt_trading = True
        self._emergency_reason = reason

    def _reset_if_new_day(self) -> None:
        now = datetime.utcnow()
        if now.date() != self._session_start.date():
            self._session_start = now
            self._order_count = 0
            self._last_reset = now
            self._realized_pnl = 0.0
            self._consecutive_losses = 0
            self._loss_cooldown_until = None
            self._daily_loss_triggered = False
            if not self._halt_due_to_drift:
                self._halt_trading = False

    def _update_loss_state(self, realized: float) -> None:
        self._realized_pnl += realized
        if self._limits.max_daily_loss > 0 and self._realized_pnl <= -self._limits.max_daily_loss:
            self._daily_loss_triggered = True
            self._halt_trading = True
        if realized < 0:
            self._consecutive_losses += 1
            if (
                self._limits.max_consecutive_losses > 0
                and self._consecutive_losses >= self._limits.max_consecutive_losses
                and self._limits.loss_cooldown_minutes > 0
            ):
                self._loss_cooldown_until = datetime.utcnow() + timedelta(
                    minutes=self._limits.loss_cooldown_minutes
                )

    def snapshot(self) -> dict:
        positions: List[dict] = [
            {
                "figi": pos.figi,
                "quantity": pos.quantity,
                "average_price": pos.average_price,
            }
            for pos in self._positions.values()
        ]
        payload = {
            "positions": positions,
            "realized_pnl": self._realized_pnl,
            "consecutive_losses": self._consecutive_losses,
            "loss_cooldown_until": self._loss_cooldown_until.isoformat()
            if self._loss_cooldown_until
            else None,
            "daily_loss_triggered": self._daily_loss_triggered,
            "halt_trading": self._halt_trading,
            "order_count": self._order_count,
            "last_reset": self._last_reset.isoformat(),
            "session_start": self._session_start.isoformat(),
            "emergency_reason": self._emergency_reason,
            "calibration_required": self._calibration_required,
            "calibration_expiry": self._calibration_expiry.isoformat()
            if self._calibration_expiry
            else None,
        }
        return payload

    def restore(self, payload: dict) -> None:
        self._positions.clear()
        for item in payload.get("positions", []):
            try:
                figi = item["figi"]
                qty = int(item.get("quantity", 0))
                price = float(item.get("average_price", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            self._positions[figi] = Position(figi=figi, quantity=qty, average_price=price)
        self._realized_pnl = float(payload.get("realized_pnl", 0.0))
        self._consecutive_losses = int(payload.get("consecutive_losses", 0))
        loss_until = payload.get("loss_cooldown_until")
        self._loss_cooldown_until = (
            datetime.fromisoformat(loss_until) if isinstance(loss_until, str) else None
        )
        self._daily_loss_triggered = bool(payload.get("daily_loss_triggered", False))
        self._halt_trading = bool(payload.get("halt_trading", False))
        self._order_count = int(payload.get("order_count", 0))
        last_reset = payload.get("last_reset")
        if isinstance(last_reset, str):
            self._last_reset = datetime.fromisoformat(last_reset)
        session_start = payload.get("session_start")
        if isinstance(session_start, str):
            self._session_start = datetime.fromisoformat(session_start)
        self._emergency_reason = payload.get("emergency_reason")
        self._calibration_required = bool(payload.get("calibration_required", False))
        calibration_expiry = payload.get("calibration_expiry")
        self._calibration_expiry = (
            datetime.fromisoformat(calibration_expiry)
            if isinstance(calibration_expiry, str)
            else None
        )

    def performance_snapshot(self) -> dict:
        return {
            "realized_pnl": self._realized_pnl,
            "consecutive_losses": self._consecutive_losses,
            "daily_loss_triggered": self._daily_loss_triggered,
            "halt_trading": self._halt_trading,
        }


__all__ = ["RiskEngine", "RiskMetrics", "Position"]
