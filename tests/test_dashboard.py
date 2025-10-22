from pathlib import Path

import pytest

from scalp_system.config.base import OrchestratorConfig
from scalp_system.orchestrator import Orchestrator
from scalp_system.storage.repository import SQLiteRepository
from scalp_system.ui.dashboard import create_dashboard_app


def test_dashboard_routes(tmp_path):
    repo_path = tmp_path / "signals.sqlite3"
    repository = SQLiteRepository(repo_path)
    repository.persist_signal("BBG000000001", 1, 0.87)
    app = create_dashboard_app(repository)
    client = app.test_client()

    response = client.get("/api/summary")
    assert response.status_code == 200
    summary = response.json()
    assert summary["total_signals"] == 1
    assert summary["latest"]["figi"] == "BBG000000001"

    response = client.get("/api/signals")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload[0]["figi"] == "BBG000000001"

    response = client.get("/")
    assert response.status_code == 200
    assert "Scalp System Dashboard" in response.get_data(as_text=True)


def _build_config(tmp_path: Path, auto_start: bool = True) -> OrchestratorConfig:
    runtime_dir = tmp_path / "runtime"
    config = OrchestratorConfig.from_dict(
        {
            "storage": {"base_path": str(runtime_dir)},
            "dashboard": {
                "auto_start": auto_start,
                "host": "127.0.0.1",
                "port": 8765,
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

    def fake_run_dashboard(path: Path, *, host: str, port: int, background: bool):
        calls.update({
            "path": path,
            "host": host,
            "port": port,
            "background": background,
        })

        class DummyThread:
            pass

        return DummyThread()

    monkeypatch.setattr("scalp_system.orchestrator.run_dashboard", fake_run_dashboard)

    orchestrator = Orchestrator(config)
    orchestrator._start_dashboard_if_needed()

    assert calls["path"] == config.dashboard.repository_path
    assert calls["host"] == config.dashboard.host
    assert calls["port"] == config.dashboard.port
    assert calls["background"] is True
    assert orchestrator._dashboard_thread is not None


def test_orchestrator_skips_dashboard_when_disabled(monkeypatch, tmp_path):
    config = _build_config(tmp_path, auto_start=False)
    def unexpected(*args, **kwargs):  # pragma: no cover - failure helper
        pytest.fail("run_dashboard should not be invoked when auto_start is disabled")

    monkeypatch.setattr("scalp_system.orchestrator.run_dashboard", unexpected)

    orchestrator = Orchestrator(config)
    orchestrator._start_dashboard_if_needed()

    assert orchestrator._dashboard_thread is None
