from scalp_system.config.base import MLConfig
from scalp_system.features.pipeline import FeatureVector
from scalp_system.ml.engine import MLEngine


def test_ml_engine_produces_signal():
    config = MLConfig()
    engine = MLEngine(config)
    vector = FeatureVector(figi="FIGI", timestamp=0.0, features=[1.0] * 16)
    signals = engine.infer([vector])
    assert len(signals) == 1
    signal = signals[0]
    assert signal.figi == "FIGI"
    assert signal.direction in (-1, 1)
    assert 0.0 <= signal.confidence <= 1.0
