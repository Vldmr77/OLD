"""Risk management subsystem."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Literal

from ..config.base import RiskLimits
from ..ml.engine import MLSignal
from ..monitoring.drift import DriftReport


@dataclass
class Position:
    figi: str
    quantity: int = 0
    average_price: float = 0.0
    stop_loss: float = 0.0


@dataclass
class RiskMetrics:
    timestamp: datetime
    unrealized_pnl: float
    gross_exposure: float
    var_value: float


@dataclass
class OrderPlan:
    quantity: int
    strategy: str
    stop_loss: float
    aggressiveness: float
    slices: int


@dataclass
class PositionAdjustment:
    figi: str
    action: Literal["tighten_stop", "close", "reduce", "hedge"]
    quantity: int
    reason: str


class RiskEngine:
    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits
        self._limits.ensure()
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

        self._capital_base = self._limits.capital_base

    def evaluate_signal(
        self,
        signal: MLSignal,
        price: float,
        *,
        spread: float,
        atr: float,
        market_volatility: float,
    ) -> Optional[OrderPlan]:
        self._reset_if_new_day()
        if self._loss_cooldown_until and datetime.utcnow() >= self._loss_cooldown_until:
            self._loss_cooldown_until = None
        if self._halt_trading:
            return None
        if self._loss_cooldown_until and datetime.utcnow() < self._loss_cooldown_until:
            return None
        if self._limits.vix_threshold > 0 and market_volatility >= self._limits.vix_threshold:
            self.trigger_emergency_halt("vix_threshold")
            return None
        if self._daily_loss_triggered and self._realized_pnl <= -self._daily_loss_limit_value():
            return None
        position = self._positions.get(signal.figi, Position(figi=signal.figi))
        plan = self._build_order_plan(position, signal, price, spread, atr)
        if plan is None:
            return None
        if self._order_count + plan.slices > self._limits.max_order_rate_per_minute:
            return None
        return plan

    def calculate_position_size(
        self,
        signal: MLSignal,
        price: float,
        *,
        spread: float,
        atr: float,
    ) -> int:
        position = self._positions.get(signal.figi, Position(figi=signal.figi))
        plan = self._build_order_plan(position, signal, price, spread, atr)
        return plan.quantity if plan else 0

    def validate_order(self, figi: str, quantity: int, price: float) -> bool:
        if quantity <= 0 or price <= 0:
            return False
        if abs(quantity) > self._limits.max_position:
            return False
        exposure = abs(quantity * price)
        if 0 < self._limits.max_gross_exposure < exposure:
            return False
        pct_limit = self._limits.max_exposure_pct * self._capital_base
        if pct_limit > 0 and exposure > pct_limit:
            return False
        return True

    def register_order(self, slices: int = 1) -> None:
        self._reset_if_new_day()
        now = datetime.utcnow()
        if (now - self._last_reset).total_seconds() > 60:
            self._order_count = 0
            self._last_reset = now
        self._order_count += max(1, slices)

    def update_position(
        self, figi: str, quantity_delta: int, price: float, stop_loss: Optional[float] = None
    ) -> None:
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
                position.stop_loss = 0.0
                return
            if prev_qty * remaining > 0:
                position.quantity = remaining
                position.average_price = prev_avg
                if stop_loss is not None:
                    position.stop_loss = stop_loss
                return
            quantity_delta = remaining
            prev_qty = 0
            prev_avg = 0.0

        new_qty = prev_qty + quantity_delta
        if new_qty == 0:
            position.quantity = 0
            position.average_price = 0.0
            position.stop_loss = 0.0
        elif prev_qty == 0:
            position.quantity = new_qty
            position.average_price = price
            position.stop_loss = stop_loss or position.stop_loss
        else:
            position.average_price = (
                (prev_avg * prev_qty + price * quantity_delta) / new_qty
            )
            position.quantity = new_qty
            if stop_loss is not None:
                position.stop_loss = stop_loss

    def process_trade_result(
        self, figi: str, filled_price: float, quantity: int, *, stop_loss: Optional[float] = None
    ) -> None:
        self.update_position(figi, quantity, filled_price, stop_loss=stop_loss)

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

    def reset_halts(self, *, reset_daily_limit: bool = True) -> None:
        """Clear manual halts so trading can resume when safe."""

        self._halt_trading = False
        self._halt_due_to_drift = False
        self._emergency_reason = None
        self._loss_cooldown_until = None
        if reset_daily_limit:
            self._daily_loss_triggered = False
            self._consecutive_losses = 0

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
        daily_limit = self._daily_loss_limit_value()
        if daily_limit > 0 and self._realized_pnl <= -daily_limit:
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
                "stop_loss": pos.stop_loss,
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
            "capital_base": self._capital_base,
        }
        return payload

    def restore(self, payload: dict) -> None:
        self._positions.clear()
        for item in payload.get("positions", []):
            try:
                figi = item["figi"]
                qty = int(item.get("quantity", 0))
                price = float(item.get("average_price", 0.0))
                stop = float(item.get("stop_loss", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            self._positions[figi] = Position(
                figi=figi, quantity=qty, average_price=price, stop_loss=stop
            )
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
        self._capital_base = float(payload.get("capital_base", self._limits.capital_base))

    def performance_snapshot(self) -> dict:
        return {
            "realized_pnl": self._realized_pnl,
            "consecutive_losses": self._consecutive_losses,
            "daily_loss_triggered": self._daily_loss_triggered,
            "halt_trading": self._halt_trading,
        }

    def manage_positions(self, snapshots: Dict[str, Dict[str, float]]) -> List[PositionAdjustment]:
        adjustments: List[PositionAdjustment] = []
        for figi, position in self._positions.items():
            if position.quantity == 0:
                continue
            snapshot = snapshots.get(figi)
            if not snapshot:
                continue
            price = float(snapshot.get("price", position.average_price or 0.0))
            if price <= 0:
                continue
            entry_price = position.average_price or price
            entry_price = max(entry_price, 1e-6)
            raw_spread = max(float(snapshot.get("spread", 0.0)), 0.0)
            spread_ratio = raw_spread / entry_price
            spread_ratio = max(spread_ratio, 0.01)
            atr_value = max(float(snapshot.get("atr", 0.0)), 0.0)
            atr_ratio = atr_value / entry_price
            offset = 1.8 * atr_ratio + 0.5 * spread_ratio
            direction = 1 if position.quantity > 0 else -1
            if direction > 0:
                desired_stop = entry_price * (1 - offset)
            else:
                desired_stop = entry_price * (1 + offset)
            desired_stop = max(desired_stop, 0.01)
            tightened = False
            if position.quantity > 0:
                if position.stop_loss == 0.0 or desired_stop > position.stop_loss:
                    position.stop_loss = desired_stop
                    tightened = True
            else:
                if position.stop_loss == 0.0 or desired_stop < position.stop_loss:
                    position.stop_loss = desired_stop
                    tightened = True
            if tightened:
                adjustments.append(
                    PositionAdjustment(
                        figi=figi,
                        action="tighten_stop",
                        quantity=abs(position.quantity),
                        reason="trailing_stop",
                    )
                )
            stop = position.stop_loss
            if stop:
                if position.quantity > 0 and price <= stop:
                    adjustments.append(
                        PositionAdjustment(
                            figi=figi,
                            action="close",
                            quantity=abs(position.quantity),
                            reason="stop_loss_trigger",
                        )
                    )
                    self.trigger_emergency_halt("stop_loss_trigger")
                elif position.quantity < 0 and price >= stop:
                    adjustments.append(
                        PositionAdjustment(
                            figi=figi,
                            action="close",
                            quantity=abs(position.quantity),
                            reason="stop_loss_trigger",
                        )
                    )
                    self.trigger_emergency_halt("stop_loss_trigger")
        return adjustments

    def forecast_liquidity(self, snapshots: Dict[str, Dict[str, float]]) -> float:
        if not snapshots:
            return 1.0
        scores: List[float] = []
        for snapshot in snapshots.values():
            price = max(float(snapshot.get("price", 0.0)), 1e-6)
            spread = max(float(snapshot.get("spread", 0.0)), 0.0)
            atr = max(float(snapshot.get("atr", 0.0)), 0.0)
            penalty = min(1.0, (spread / price) + atr)
            scores.append(max(0.0, 1.0 - penalty))
        return sum(scores) / len(scores)

    def hedge_portfolio(self, market_prices: Dict[str, float]) -> List[PositionAdjustment]:
        adjustments: List[PositionAdjustment] = []
        exposures: List[tuple[str, float, float, int]] = []
        for figi, position in self._positions.items():
            if position.quantity == 0:
                continue
            price = float(market_prices.get(figi, position.average_price or 0.0))
            if price <= 0:
                continue
            exposure = price * position.quantity
            exposures.append((figi, exposure, price, position.quantity))
        if not exposures:
            return adjustments
        gross = sum(abs(item[1]) for item in exposures)
        if gross <= 0:
            return adjustments
        net = sum(item[1] for item in exposures)
        imbalance = abs(net) / gross
        threshold = max(0.05, min(0.25, self._limits.max_risk_per_instrument * 2))
        if imbalance < threshold:
            return adjustments
        target_figi, _, price, qty = max(exposures, key=lambda item: abs(item[1]))
        hedge_qty = int(abs(net) / max(price, 1.0))
        hedge_qty = min(hedge_qty, self._limits.max_position)
        if hedge_qty > 0:
            adjustments.append(
                PositionAdjustment(
                    figi=target_figi,
                    action="hedge",
                    quantity=hedge_qty,
                    reason=f"exposure_imbalance_{imbalance:.3f}_{'long' if net > 0 else 'short'}",
                )
            )
        exposure_limit = self._limits.max_gross_exposure
        if exposure_limit > 0 and abs(qty) * price > exposure_limit:
            reduce_qty = max(1, abs(qty) - int(exposure_limit / max(price, 1.0)))
            adjustments.append(
                PositionAdjustment(
                    figi=target_figi,
                    action="reduce",
                    quantity=reduce_qty,
                    reason="gross_exposure_limit",
                )
            )
        return adjustments

    def calculate_order_plan(
        self, signal: MLSignal, price: float, *, spread: float, atr: float
    ) -> Optional[OrderPlan]:
        position = self._positions.get(signal.figi, Position(figi=signal.figi))
        return self._build_order_plan(position, signal, price, spread, atr)

    def update_capital(self, equity: float) -> None:
        self._capital_base = max(equity, 1.0)


    def _build_order_plan(
        self,
        position: Position,
        signal: MLSignal,
        price: float,
        spread: float,
        atr: float,
    ) -> Optional[OrderPlan]:
        spread = max(spread, 0.01)
        atr = max(atr, 0.0)
        stop_loss = price * (1 + (1.8 * atr + 0.5 * spread) * signal.direction)
        stop_distance = abs(stop_loss - price)
        if stop_distance <= 0:
            stop_distance = max(price * 0.001, 0.01)

        capacity = self._capacity_for(position, signal.direction)
        if capacity <= 0:
            return None

        risk_budget = self._capital_base * self._limits.max_risk_per_instrument
        if risk_budget <= 0:
            return None
        raw_quantity = int(risk_budget / stop_distance)
        quantity = max(1, min(raw_quantity, capacity))

        if position.quantity * signal.direction >= 0 and quantity <= 0:
            return None

        limit_value = min(
            self._limits.max_gross_exposure,
            self._limits.max_exposure_pct * self._capital_base,
        )
        if limit_value > 0:
            max_quantity = int(limit_value / price) if price > 0 else 0
            if position.quantity and abs(position.quantity) > max_quantity:
                return None
            remaining_capacity = max_quantity - abs(position.quantity)
            if position.quantity * signal.direction >= 0:
                if remaining_capacity <= 0:
                    return None
                quantity = min(quantity, remaining_capacity)
        if quantity <= 0:
            return None

        strategy = "vwap" if quantity > 5 else "ioc"
        aggressiveness = 0.8 if strategy == "vwap" else 1.0
        slices = 2 if strategy == "vwap" and quantity > 5 else 1

        projected_qty = position.quantity + signal.direction * quantity
        if abs(projected_qty) > self._limits.max_position:
            return None
        projected_exposure = abs(projected_qty * price)
        exposure_limit = min(
            self._limits.max_gross_exposure,
            self._limits.max_exposure_pct * self._capital_base,
        )
        if exposure_limit > 0 and projected_exposure > exposure_limit:
            return None

        return OrderPlan(
            quantity=quantity,
            strategy=strategy,
            stop_loss=stop_loss,
            aggressiveness=aggressiveness,
            slices=slices,
        )

    def _capacity_for(self, position: Position, direction: int) -> int:
        if direction > 0:
            return self._limits.max_position - position.quantity
        if direction < 0:
            return self._limits.max_position + position.quantity
        return 0

    def _daily_loss_limit_value(self) -> float:
        pct_limit = self._capital_base * self._limits.daily_loss_limit_pct
        return max(self._limits.max_daily_loss, pct_limit)


__all__ = [
    "RiskEngine",
    "RiskMetrics",
    "Position",
    "OrderPlan",
    "PositionAdjustment",
]
