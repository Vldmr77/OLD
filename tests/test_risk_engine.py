from datetime import datetime, timedelta

from scalp_system.config.base import RiskLimits
from scalp_system.ml.engine import MLSignal
from scalp_system.monitoring.drift import DriftReport
from scalp_system.risk.engine import RiskEngine


def test_risk_engine_respects_limits():
    engine = RiskEngine(
        RiskLimits(
            max_position=1,
            max_gross_exposure=1000,
            capital_base=1000,
            max_risk_per_instrument=0.001,
            max_exposure_pct=0.5,
        )
    )
    signal = MLSignal(figi="FIGI", direction=1, confidence=0.9)
    plan = engine.evaluate_signal(
        signal,
        price=100,
        spread=0.05,
        atr=0.0,
        market_volatility=0.1,
    )
    assert plan is not None
    engine.update_position("FIGI", signal.direction * plan.quantity, price=100, stop_loss=plan.stop_loss)
    disallowed = engine.evaluate_signal(
        signal,
        price=100,
        spread=0.05,
        atr=0.0,
        market_volatility=0.1,
    )
    assert disallowed is None


def test_risk_engine_halts_on_critical_drift():
    engine = RiskEngine(RiskLimits())
    engine.record_drift(
        DriftReport(
            p_value=0.01,
            mean_diff=10.0,
            sigma=0.1,
            severity="critical",
            triggered=True,
            calibrate=True,
            alert=True,
        )
    )
    blocked = engine.evaluate_signal(
        MLSignal(figi="FIGI", direction=1, confidence=0.5),
        price=1,
        spread=0.01,
        atr=0.0,
        market_volatility=0.5,
    )
    assert blocked is None
    assert engine.trading_halted() is True


def test_risk_engine_reset_halts_clears_state():
    engine = RiskEngine(RiskLimits())
    engine.trigger_emergency_halt("manual")
    assert engine.trading_halted() is True

    engine.reset_halts()

    assert engine.trading_halted() is False


def test_risk_engine_triggers_vwap_strategy():
    limits = RiskLimits(
        max_position=500,
        max_gross_exposure=1_000_000,
        capital_base=1_000_000,
        max_risk_per_instrument=0.01,
        max_exposure_pct=1.0,
    )
    engine = RiskEngine(limits)
    signal = MLSignal(figi="FIGI", direction=1, confidence=0.95)
    plan = engine.evaluate_signal(
        signal,
        price=100,
        spread=0.5,
        atr=0.0,
        market_volatility=0.1,
    )
    assert plan is not None
    assert plan.strategy == "vwap"
    assert plan.slices == 2


def test_risk_engine_halts_on_vix_threshold():
    limits = RiskLimits(vix_threshold=0.3, capital_base=1000, max_risk_per_instrument=0.001)
    engine = RiskEngine(limits)
    blocked = engine.evaluate_signal(
        MLSignal(figi="FIGI", direction=1, confidence=0.6),
        price=100,
        spread=0.05,
        atr=0.0,
        market_volatility=0.5,
    )
    assert blocked is None
    assert engine.trading_halted() is True


def test_risk_engine_blocks_after_daily_loss():
    limits = RiskLimits(
        max_position=10,
        max_daily_loss=50,
        capital_base=1000,
        max_risk_per_instrument=0.001,
    )
    engine = RiskEngine(limits)
    engine.update_position("FIGI", 1, price=100)
    engine.update_position("FIGI", -1, price=40)
    disallowed = engine.evaluate_signal(
        MLSignal(figi="FIGI", direction=1, confidence=0.8),
        price=100,
        spread=0.05,
        atr=0.0,
        market_volatility=0.1,
    )
    assert disallowed is None


def test_risk_engine_cooldown_after_consecutive_losses(monkeypatch):
    limits = RiskLimits(
        max_position=5,
        max_consecutive_losses=1,
        loss_cooldown_minutes=1,
        capital_base=1000,
        max_risk_per_instrument=0.001,
    )
    engine = RiskEngine(limits)
    engine.update_position("FIGI", 1, price=100)
    engine.update_position("FIGI", -1, price=90)
    denied = engine.evaluate_signal(
        MLSignal(figi="FIGI", direction=1, confidence=0.8),
        price=100,
        spread=0.05,
        atr=0.0,
        market_volatility=0.1,
    )
    assert denied is None

    cooldown_expired = datetime.utcnow() - timedelta(minutes=2)
    monkeypatch.setattr(engine, "_loss_cooldown_until", cooldown_expired)
    allowed = engine.evaluate_signal(
        MLSignal(figi="FIGI", direction=1, confidence=0.8),
        price=100,
        spread=0.05,
        atr=0.0,
        market_volatility=0.1,
    )
    assert allowed is not None


def test_risk_engine_snapshot_and_restore_preserves_state():
    limits = RiskLimits(max_position=5, max_daily_loss=100)
    engine = RiskEngine(limits)
    engine.update_position("FIGI", 1, price=100)
    engine.update_position("FIGI", -1, price=90)
    engine.trigger_emergency_halt("latency:features")
    snapshot = engine.snapshot()

    restored = RiskEngine(limits)
    restored.restore(snapshot)

    assert restored.trading_halted() is True
    restored_state = restored.snapshot()
    assert restored_state["positions"][0]["figi"] == "FIGI"
    assert restored_state["realized_pnl"] == snapshot["realized_pnl"]


def test_risk_engine_calculates_and_validates_order():
    limits = RiskLimits(
        max_position=50,
        max_gross_exposure=100_000,
        capital_base=200_000,
        max_risk_per_instrument=0.05,
    )
    engine = RiskEngine(limits)
    signal = MLSignal(figi="FIGI", direction=1, confidence=0.9)
    size = engine.calculate_position_size(signal, price=100, spread=0.05, atr=0.2)
    assert size > 0
    assert engine.validate_order("FIGI", size, 100)


def test_risk_engine_manage_positions_trailing_and_closure():
    limits = RiskLimits(max_position=10, capital_base=10_000, max_risk_per_instrument=0.05)
    engine = RiskEngine(limits)
    engine.update_position("FIGI", 3, price=100, stop_loss=90)
    snapshots = {"FIGI": {"price": 120.0, "spread": 0.1, "atr": 0.5}}
    adjustments = engine.manage_positions(snapshots)
    assert any(adj.action == "tighten_stop" for adj in adjustments)
    updated_stop = engine.snapshot()["positions"][0]["stop_loss"]
    assert updated_stop > 90
    trigger_snapshots = {"FIGI": {"price": updated_stop - 0.1, "spread": 0.1, "atr": 0.5}}
    closing = engine.manage_positions(trigger_snapshots)
    assert any(adj.action == "close" for adj in closing)
    assert engine.trading_halted() is True


def test_risk_engine_liquidity_forecast_and_hedge():
    limits = RiskLimits(
        max_position=100,
        capital_base=100_000,
        max_risk_per_instrument=0.05,
        max_gross_exposure=150_000,
    )
    engine = RiskEngine(limits)
    engine.update_position("FIGI1", 50, price=100)
    engine.update_position("FIGI2", -10, price=50)
    snapshots = {
        "FIGI1": {"price": 100.0, "spread": 0.2, "atr": 0.4},
        "FIGI2": {"price": 50.0, "spread": 0.1, "atr": 0.3},
    }
    liquidity = engine.forecast_liquidity(snapshots)
    assert 0.0 <= liquidity <= 1.0
    adjustments = engine.hedge_portfolio({figi: data["price"] for figi, data in snapshots.items()})
    assert adjustments
    assert any(adj.action in {"hedge", "reduce"} for adj in adjustments)
