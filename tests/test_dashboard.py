from pathlib import Path

import pytest

from scalp_system.config.base import OrchestratorConfig
from scalp_system.orchestrator import Orchestrator
from scalp_system.storage.repository import SQLiteRepository
from scalp_system.ui.dashboard import DashboardStatus, DashboardUI, ModuleStatus


def test_dashboard_refresh_headless(tmp_path):
    repo_path = tmp_path / "signals.sqlite3"
    repository = SQLiteRepository(repo_path)
    repository.persist_signal("BBG000000001", 1, 0.87)

    status = DashboardStatus(
        modules=[ModuleStatus("Data engine", "RUNNING", "detail")],
        processes=["stream"],
        errors=["none"],
        ensemble={"base_weights": {"lstm_ob": 0.4}},
    )

    ui = DashboardUI(repository, status_provider=lambda: status, headless=True)
    snapshot = ui.refresh_once()

    assert snapshot.summary["total_signals"] == 1
    assert snapshot.status.modules[0].name == "Data engine"
    assert snapshot.status.ensemble["base_weights"]["lstm_ob"] == pytest.approx(0.4)


def _build_config(tmp_path: Path, auto_start: bool = True) -> OrchestratorConfig:
    runtime_dir = tmp_path / "runtime"
    config = OrchestratorConfig.from_dict(
        {
            "storage": {"base_path": str(runtime_dir)},
            "dashboard": {
                "auto_start": auto_start,
                "repository_path": str(runtime_dir / "signals.sqlite3"),
                "refresh_interval_ms": 750,
                "signal_limit": 15,
                "title": "Test Dashboard",
                "headless": True,
            },
            "reporting": {
                "report_path": str(tmp_path / "reports" / "performance.jsonl")
            },
            "training": {
                "dataset_path": str(tmp_path / "data" / "training.jsonl"),
                "output_dir": str(tmp_path / "models"),
            },
            "backtest": {
                "dataset_path": str(tmp_path / "data" / "backtest.jsonl")
            },
            "disaster_recovery": {
                "replication_path": str(tmp_path / "backups"),
                "enabled": False,
            },
        }
    )
    return config


def test_orchestrator_starts_dashboard_when_enabled(monkeypatch, tmp_path):
    config = _build_config(tmp_path, auto_start=True)
    calls: dict[str, object] = {}

    def fake_run_dashboard(
        path: Path,
        *,
        status_provider,
        refresh_interval_ms: int,
        signal_limit: int,
        headless: bool,
        title: str,
        background: bool,
        config_path=None,
        restart_callback=None,
        token_status_provider=None,
        token_writer=None,
    ):
        calls.update(
            {
                "path": path,
                "status_provider": status_provider,
                "refresh_interval_ms": refresh_interval_ms,
                "signal_limit": signal_limit,
                "headless": headless,
                "title": title,
                "background": background,
                "config_path": config_path,
                "restart_callback": restart_callback,
                "token_status_provider": token_status_provider,
                "token_writer": token_writer,
            }
        )

        class DummyThread:
            pass

        return DummyThread()

    monkeypatch.setattr("scalp_system.orchestrator.run_dashboard", fake_run_dashboard)

    orchestrator = Orchestrator(config)
    orchestrator._start_dashboard_if_needed()

    assert calls["path"] == config.dashboard.repository_path
    assert calls["refresh_interval_ms"] == config.dashboard.refresh_interval_ms
    assert calls["signal_limit"] == config.dashboard.signal_limit
    assert calls["headless"] is True
    assert calls["title"] == "Test Dashboard"
    assert calls["background"] is True
    assert calls["config_path"] is None
    assert callable(calls["restart_callback"])
    provider = calls["status_provider"]
    assert provider.__self__ is orchestrator  # bound method


def test_orchestrator_skips_dashboard_when_disabled(monkeypatch, tmp_path):
    config = _build_config(tmp_path, auto_start=False)

    def unexpected(*args, **kwargs):  # pragma: no cover - failure helper
        pytest.fail("run_dashboard should not be invoked when auto_start is disabled")

    monkeypatch.setattr("scalp_system.orchestrator.run_dashboard", unexpected)

    orchestrator = Orchestrator(config)
    orchestrator._start_dashboard_if_needed()

    assert orchestrator._dashboard_thread is None


def test_dashboard_status_contains_modules(tmp_path):
    config = _build_config(tmp_path)
    orchestrator = Orchestrator(config)
    status = orchestrator.dashboard_status()

    names = {module.name for module in status.modules}
    assert "Data engine" in names
    assert "ML engine" in names
    assert "Execution" in names
    assert status.ensemble["base_weights"]
    assert any(proc.startswith("Disaster recovery") for proc in status.processes)
