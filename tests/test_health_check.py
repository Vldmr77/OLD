import pytest

from scalp_system.cli.health_check import CHECKS, run_health_checks
from scalp_system.config.base import MLConfig, OrchestratorConfig, StorageConfig


def _make_config(tmp_path):
    storage = StorageConfig(base_path=tmp_path / "runtime")
    ml = MLConfig(model_dir=tmp_path / "models")
    config = OrchestratorConfig(storage=storage, ml=ml)
    return config


def test_run_health_checks_all_modules(tmp_path):
    config = _make_config(tmp_path)
    results = run_health_checks(module="all", config=config)
    assert len(results) == len(CHECKS)
    assert any(entry.startswith("storage:") for entry in results)
    assert any(entry.startswith("data:") for entry in results)
    assert any(entry.startswith("ml:") for entry in results)
    assert any(entry.startswith("notifications:") for entry in results)
    assert any(entry.startswith("risk:") for entry in results)
    assert any(entry.startswith("calibration:") for entry in results)


def test_run_health_checks_single_module(tmp_path):
    config = _make_config(tmp_path)
    results = run_health_checks(module="storage", config=config)
    assert len(results) == 1
    assert results[0].startswith("storage:")


def test_run_health_checks_unknown_module(tmp_path):
    config = _make_config(tmp_path)
    with pytest.raises(ValueError):
        run_health_checks(module="unknown", config=config)

