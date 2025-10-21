from scalp_system.config.base import RiskLimits
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
