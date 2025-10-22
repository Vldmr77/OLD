import socket
import sys
from pathlib import Path

import pytest

from scalp_system.cli import dashboard as dashboard_cli
from scalp_system.config.base import OrchestratorConfig
from scalp_system.orchestrator import Orchestrator
from scalp_system.ui import DashboardApp
from scalp_system.ui.bus_client import DashboardBusClient


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
        token_writer=None,
        token_status_provider=None,
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
                "token_writer": token_writer,
                "token_status_provider": token_status_provider,
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
    assert callable(calls["token_writer"])
    assert callable(calls["token_status_provider"])
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


def test_orchestrator_launches_dashboard_process(monkeypatch, tmp_path):
    config = _build_config(tmp_path, auto_start=True)
    config.dashboard.headless = False
    config.dashboard.title = "Proc UI"
    config.dashboard.repository_path = tmp_path / "runtime" / "signals.db"

    monkeypatch.setenv("DISPLAY", ":0")

    captured: dict[str, object] = {}
    status_calls: dict[str, object] = {}

    class DummyProcess:
        def __init__(self, args: list[str]):
            self.args = args
            self._terminated = False
            self.pid = 1234

        def poll(self) -> int | None:
            return 0 if self._terminated else None

        def terminate(self) -> None:
            self._terminated = True

        def wait(self, timeout: float | None = None) -> int:
            self._terminated = True
            return 0

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return DummyProcess(args)

    class DummyStatusServer:
        def __init__(self, host: str, port: int, status_provider):
            status_calls["host"] = host
            status_calls["port"] = port
            status_calls["provider"] = status_provider
            status_calls["start_count"] = 0
            self._endpoint = f"http://{host}:{port}/status"

        def start(self) -> None:
            status_calls["start_count"] = status_calls.get("start_count", 0) + 1

        def stop(self) -> None:
            status_calls["stopped"] = True

        @property
        def status_endpoint(self) -> str:
            return self._endpoint

    monkeypatch.setattr("scalp_system.orchestrator.subprocess.Popen", fake_popen)
    monkeypatch.setattr("scalp_system.orchestrator.sys.executable", sys.executable)
    monkeypatch.setattr(
        "scalp_system.orchestrator.DashboardStatusServer", DummyStatusServer
    )

    orchestrator = Orchestrator(config, config_path=tmp_path / "config.yaml")
    orchestrator._start_dashboard_if_needed()

    assert captured["args"][0] == sys.executable
    assert "scalp_system.cli.dashboard" in captured["args"]
    assert "--repository" in captured["args"]
    assert "--status-endpoint" in captured["args"]
    idx = captured["args"].index("--status-endpoint")
    assert captured["args"][idx + 1] == "http://127.0.0.1:5000/status"
    assert status_calls["start_count"] == 1


def test_orchestrator_skips_dashboard_when_no_display(monkeypatch, tmp_path, caplog):
    config = _build_config(tmp_path, auto_start=True)
    config.dashboard.headless = False

    def fail_run_dashboard(*_, **__):  # pragma: no cover - should not be called
        raise AssertionError("dashboard should not launch without display")

    monkeypatch.setattr("scalp_system.orchestrator.run_dashboard", fail_run_dashboard)
    monkeypatch.setenv("DISPLAY", "")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(
        "scalp_system.orchestrator.Orchestrator._display_available",
        staticmethod(lambda: False),
    )

    orchestrator = Orchestrator(config)

    with caplog.at_level("WARNING"):
        orchestrator._start_dashboard_if_needed()

    assert "Display not detected" in caplog.text

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
    assert callable(captured["token_writer"])
    assert callable(captured["token_status_provider"])


def test_dashboard_bus_client_caches_bus_state(monkeypatch):
    checks: dict[str, int] = {"count": 0}

    class DummyCoreBus:
        def __init__(self, *_, **__):
            pass

        def check_available(self) -> bool:
            checks["count"] += 1
            return True

        def emit_event(self, event):  # pragma: no cover - behaviour mocked
            return None

    monkeypatch.setattr("scalp_system.ui.bus_client.CoreBusClient", DummyCoreBus)

    client = DashboardBusClient("127.0.0.1", 9999, availability_ttl=10.0)

    assert client.is_up() is True
    assert client.is_up() is True  # cached result avoids extra socket checks
    assert checks["count"] == 1


def test_dashboard_bus_client_marks_bus_down_after_emit_failure(monkeypatch):
    class DummyCoreBus:
        def __init__(self, *_, **__):
            pass

        def check_available(self) -> bool:
            raise AssertionError("check_available should not be called when cached")

        def emit_event(self, event):  # pragma: no cover - behaviour mocked
            raise OSError("bus offline")

    monkeypatch.setattr("scalp_system.ui.bus_client.CoreBusClient", DummyCoreBus)

    client = DashboardBusClient("127.0.0.1", 9999, availability_ttl=10.0)

    with pytest.raises(OSError):
        client.emit("system.restart")

    assert client.is_up() is False
