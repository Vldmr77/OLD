import pytest
import json

import pytest

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


def test_ml_engine_reload_recovers_corrupted(tmp_path):
    config = MLConfig()
    engine = MLEngine(config)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for idx, (name, filename) in enumerate(engine._model_files.items()):  # type: ignore[attr-defined]
        path = model_dir / filename
        if idx == 0:
            path.write_text("{\"weights\": []}", encoding="utf-8")
        else:
            path.write_text("not-json", encoding="utf-8")
    engine.reload_models(model_dir)
    for name, filename in engine._model_files.items():  # type: ignore[attr-defined]
        payload = json.loads((model_dir / filename).read_text(encoding="utf-8"))
        if name == "gbdt_feat":
            assert "stages" in payload and payload["stages"]
        else:
            assert "weights" in payload and payload["weights"]


def test_gpu_failure_triggers_cpu_fallback(monkeypatch):
    config = MLConfig()
    engine = MLEngine(config)
    vector = FeatureVector(figi="FIGI", timestamp=0.0, features=[1.0] * 8)

    original_predict = engine._models["lstm_ob"].predict  # type: ignore[attr-defined]
    call_count = {"calls": 0}

    def flaky_predict(batch):
        call_count["calls"] += 1
        if call_count["calls"] == 1:
            raise RuntimeError("CUDA device lost")
        return original_predict(batch)

    engine._models["lstm_ob"].predict = flaky_predict  # type: ignore[attr-defined]

    signals = engine.infer([vector])
    assert signals
    assert engine.device_mode == "cpu"
    assert pytest.approx(engine.throttle_delay, rel=1e-6) == 0.2
    detail = engine.drain_failover_event()
    assert detail is not None and "cuda" in detail.lower()
