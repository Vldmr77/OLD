import socket
from pathlib import Path

from scalp_system.cli import dashboard as dashboard_cli
from scalp_system.config.base import OrchestratorConfig
from scalp_system.orchestrator import Orchestrator
from scalp_system.ui import DashboardApp


def test_dashboard_refresh_headless():
    status_payload = {
        "build": {"version": "test", "uptime_s": 0, "mode": "development"},
        "modules": [
            {"name": "Data engine", "state": "ready", "detail": "ok"},
            {"name": "ML engine", "state": "ready", "detail": "gpu"},
        ],
        "alerts": [],
        "signals": [
            {"figi": "FIGI1", "score": 0.87, "timestamp": "2024-01-01T00:00:00Z"}
        ],
        "orders": [],
        "metrics": {"cpu": 10.0},
        "data": {"queues": {"ingest": 1}, "errors": 0},
        "ml": {"models": [], "jobs": [], "metrics": {}, "artifacts": []},
        "risk": {"profile": {}, "exposure": [], "params": {}, "alerts": [], "log": []},
        "execution": {"queue": [], "mode": {"mode": "dev", "paper": True}, "fills": [], "stats": {}},
        "adapter": {"quota": {}, "errors": {}, "diagnostics": {}},
        "bus": {"state": "down", "host": "127.0.0.1", "port": 8787, "rps": 0, "lag_ms": 0, "topics": [], "events": []},
        "logs": {"tail": [], "metrics": {}},
    }

    app = DashboardApp(headless=True, status_provider=lambda: status_payload)
    app.run()
    snapshot = app._state.snapshot()

    modules = {mod["name"]: mod for mod in snapshot["modules"]}
    assert modules["Data engine"]["state"] == "ready"
    assert snapshot["signals"][0]["figi"] == "FIGI1"


def _build_config(tmp_path: Path, auto_start: bool = True) -> OrchestratorConfig:
    runtime_dir = tmp_path / "runtime"
    config = OrchestratorConfig.from_dict(
        {
            "storage": {"base_path": str(runtime_dir)},
            "dashboard": {
                "auto_start": auto_start,
                "repository_path": str(runtime_dir / "signals.db"),
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
        instrument_replace_callback=None,
        sandbox_forward_callback=None,
        backtest_callback=None,
        training_callback=None,
        bus_address=None,
        events_endpoint=None,
        status_endpoint=None,
        **_: object,
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
                "instrument_replace_callback": instrument_replace_callback,
                "sandbox_forward_callback": sandbox_forward_callback,
                "backtest_callback": backtest_callback,
                "training_callback": training_callback,
                "bus_address": bus_address,
                "events_endpoint": events_endpoint,
                "status_endpoint": status_endpoint,
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
    assert provider.__self__ is orchestrator
    assert calls["bus_address"] is None
    assert calls["events_endpoint"] is None
    assert calls["status_endpoint"] is None


def _get_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def test_orchestrator_provides_bus_address_when_bus_running(monkeypatch, tmp_path):
    config = _build_config(tmp_path, auto_start=True)
    config.bus.enabled = True
    config.bus.port = _get_free_port()

    calls: dict[str, object] = {}

    def fake_run_dashboard(*args, **kwargs):
        calls.update(kwargs)

        class DummyThread:
            pass

        return DummyThread()

    monkeypatch.setattr("scalp_system.orchestrator.run_dashboard", fake_run_dashboard)

    orchestrator = Orchestrator(config)
    orchestrator._start_event_bus_if_needed()
    try:
        orchestrator._start_dashboard_if_needed()
        assert calls["bus_address"] == (
            config.bus.host,
            orchestrator._event_bus.port,  # type: ignore[attr-defined]
        )
    finally:
        orchestrator._stop_event_bus()


def test_cli_dashboard_omits_bus_when_unavailable(monkeypatch, tmp_path):
    config = _build_config(tmp_path)
    config.bus.enabled = True

    monkeypatch.setattr("scalp_system.cli.dashboard.load_config", lambda path: config)

    captured: dict[str, object] = {}

    def fake_run_dashboard(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("scalp_system.cli.dashboard.run_dashboard", fake_run_dashboard)

    class DummyBus:
        def __init__(self, *args, **kwargs):
            pass

        def check_available(self) -> bool:
            return False

    monkeypatch.setattr("scalp_system.cli.dashboard.BusClient", DummyBus)

    dashboard_cli.main(["--repository", str(tmp_path / "signals.db"), "--config", "dummy.yaml"])

    assert captured["bus_address"] is None
