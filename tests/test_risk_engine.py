from datetime import datetime, timedelta

from scalp_system.config.base import RiskLimits
from scalp_system.ml.engine import MLSignal
from scalp_system.monitoring.drift import DriftReport
from scalp_system.risk.engine import RiskEngine


def test_risk_engine_respects_limits():
    engine = RiskEngine(RiskLimits(max_position=2, max_gross_exposure=1000))
    signal = MLSignal(figi="FIGI", direction=1, confidence=0.9)
    allowed = engine.evaluate_signal(signal, price=100)
    assert allowed is True
    engine.update_position("FIGI", 2, price=100)
    disallowed = engine.evaluate_signal(signal, price=100)
    assert disallowed is False


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
    blocked = engine.evaluate_signal(MLSignal(figi="FIGI", direction=1, confidence=0.5), price=1)
    assert blocked is False
    assert engine.trading_halted() is True


def test_risk_engine_blocks_after_daily_loss():
    limits = RiskLimits(max_position=10, max_daily_loss=50)
    engine = RiskEngine(limits)
    engine.update_position("FIGI", 1, price=100)
    engine.update_position("FIGI", -1, price=40)
    disallowed = engine.evaluate_signal(MLSignal(figi="FIGI", direction=1, confidence=0.8), price=100)
    assert disallowed is False


def test_risk_engine_cooldown_after_consecutive_losses(monkeypatch):
    limits = RiskLimits(
        max_position=5,
        max_consecutive_losses=1,
        loss_cooldown_minutes=1,
    )
    engine = RiskEngine(limits)
    engine.update_position("FIGI", 1, price=100)
    engine.update_position("FIGI", -1, price=90)
    denied = engine.evaluate_signal(MLSignal(figi="FIGI", direction=1, confidence=0.8), price=100)
    assert denied is False

    cooldown_expired = datetime.utcnow() - timedelta(minutes=2)
    monkeypatch.setattr(engine, "_loss_cooldown_until", cooldown_expired)
    allowed = engine.evaluate_signal(MLSignal(figi="FIGI", direction=1, confidence=0.8), price=100)
    assert allowed is True


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
