from scalp_system.config.base import MLConfig
from scalp_system.features.pipeline import FeatureVector
from scalp_system.ml.engine import MLEngine, _MIN_MODEL_SIZE


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


def test_ml_engine_reload_recovers_corrupted(tmp_path):
    config = MLConfig()
    engine = MLEngine(config)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for idx, (name, filename) in enumerate(engine._model_files.items()):  # type: ignore[attr-defined]
        path = model_dir / filename
        size = _MIN_MODEL_SIZE * 2 if idx == 0 else 256
        path.write_bytes(b"1" * size)
    engine.reload_models(model_dir)
    for filename in engine._model_files.values():  # type: ignore[attr-defined]
        assert (model_dir / filename).stat().st_size >= _MIN_MODEL_SIZE * 2
