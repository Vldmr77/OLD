from scalp_system.config.base import RiskLimits
from scalp_system.ml.engine import MLSignal
from scalp_system.risk.engine import RiskEngine


def test_risk_engine_respects_limits():
    engine = RiskEngine(RiskLimits(max_position=2, max_gross_exposure=1000))
    signal = MLSignal(figi="FIGI", direction=1, confidence=0.9)
    allowed = engine.evaluate_signal(signal, price=100)
    assert allowed is True
    engine.update_position("FIGI", 2, price=100)
    disallowed = engine.evaluate_signal(signal, price=100)
    assert disallowed is False
