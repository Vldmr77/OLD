import json
from pathlib import Path

import pytest

from scalp_system.cli.model_trainer import run_daemon
from scalp_system.config.base import TrainingConfig
from scalp_system.ml.training import ModelTrainer


def _write_dataset(path, samples):
    lines = [json.dumps(sample) for sample in samples]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_model_trainer_outputs_weights(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    samples = [
        {"figi": "FIGI", "features": [0.1 * i for i in range(1, 5)], "label": 0.2},
        {"figi": "FIGI", "features": [0.2 * i for i in range(1, 5)], "label": 0.5},
        {"figi": "FIGI", "features": [0.3 * i for i in range(1, 5)], "label": 0.8},
        {"figi": "FIGI", "features": [0.4 * i for i in range(1, 5)], "label": 0.1},
        {"figi": "FIGI", "features": [0.5 * i for i in range(1, 5)], "label": -0.1},
        {"figi": "FIGI", "features": [0.6 * i for i in range(1, 5)], "label": 0.3},
        {"figi": "FIGI", "features": [0.7 * i for i in range(1, 5)], "label": 0.4},
        {"figi": "FIGI", "features": [0.8 * i for i in range(1, 5)], "label": 0.6},
        {"figi": "FIGI", "features": [0.9 * i for i in range(1, 5)], "label": -0.2},
        {"figi": "FIGI", "features": [1.0 * i for i in range(1, 5)], "label": 0.9},
    ]
    _write_dataset(dataset, samples)
    output_dir = tmp_path / "models"
    config = TrainingConfig(dataset_path=dataset, output_dir=output_dir, epochs=3, min_samples=5)
    trainer = ModelTrainer(config)

    report = trainer.train()

    assert report.samples == len(samples)
    assert report.training_loss >= 0.0
    assert report.validation_loss >= 0.0
    assert report.output_path.exists()
    weights = json.loads(report.output_path.read_text(encoding="utf-8"))
    assert set(weights) == {
        "lstm_ob",
        "gbdt_features",
        "transformer_temporal",
        "svm_volatility",
    }
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-6)
    assert report.model_dir.exists()
    expected_files = {
        "lstm_ob.tflite": "weights",
        "gbdt_features.tflite": "stages",
        "temporal_transformer.tflite": "weights",
        "volatility_svm.tflite": "weights",
    }
    for filename, key in expected_files.items():
        payload = json.loads((report.model_dir / filename).read_text(encoding="utf-8"))
        assert key in payload
    assert report.quantization_plan is not None
    plan_entries = json.loads(report.quantization_plan.read_text(encoding="utf-8"))
    assert plan_entries
    assert {entry["precision"] for entry in plan_entries} <= {"int8", "float16"}
    assert len(report.quantization_decisions) == 4
    for decision in report.quantization_decisions:
        assert decision.precision in {"int8", "float16"}


def test_model_trainer_requires_enough_samples(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    samples = [
        {"figi": "FIGI", "features": [0.1, 0.2], "label": 0.3},
        {"figi": "FIGI", "features": [0.4, 0.5], "label": 0.6},
    ]
    _write_dataset(dataset, samples)
    config = TrainingConfig(dataset_path=dataset, output_dir=tmp_path, min_samples=5)
    trainer = ModelTrainer(config)

    with pytest.raises(ValueError):
        trainer.train()


def test_run_daemon_honours_iterations(tmp_path):
    emitted = []

    def fake_runner(config_path: Path, overrides: dict[str, object]):
        artefact = tmp_path / f"artefact_{len(emitted)}.json"
        artefact.write_text("{}", encoding="utf-8")
        emitted.append(artefact)
        return artefact

    results = list(
        run_daemon(
            tmp_path / "config.yaml",
            {},
            interval_minutes=0,
            iterations=2,
            runner=fake_runner,
        )
    )

    assert results == emitted
    assert len(emitted) == 2
