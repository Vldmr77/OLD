import logging
import socket
from datetime import datetime, timezone

from scalp_system.config.base import OrchestratorConfig
from scalp_system.data.models import OrderBook, OrderBookLevel

from scalp_system.orchestrator import Orchestrator


def _get_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _build_config(tmp_path, *, bus_enabled: bool = False) -> OrchestratorConfig:
    runtime = tmp_path / "runtime"
    config = OrchestratorConfig.from_dict(
        {
            "storage": {"base_path": str(runtime)},
            "dashboard": {
                "auto_start": False,
                "repository_path": str(runtime / "signals.db"),
                "headless": True,
            },
            "reporting": {"report_path": str(tmp_path / "reports" / "perf.jsonl")},
            "training": {
                "dataset_path": str(tmp_path / "data" / "training.jsonl"),
                "output_dir": str(tmp_path / "models"),
            },
            "backtest": {"dataset_path": str(tmp_path / "data" / "backtest.jsonl")},
            "manual_override": {
                "enabled": True,
                "flag_path": str(runtime / "manual.flag"),
            },
        }
    )
    config.dashboard.auto_start = False
    config.dashboard.headless = True
    config.bus.enabled = bus_enabled
    if bus_enabled:
        config.bus.port = _get_free_port()
    return config


def test_orchestrator_pause_and_resume_toggle_manual_override(tmp_path):
    config = _build_config(tmp_path)
    orchestrator = Orchestrator(config)

    ok, message = orchestrator.pause_trading()
    assert ok is True
    assert "paused" in message.lower()
    assert config.manual_override.flag_path.exists()
    assert orchestrator._risk_engine.trading_halted() is True  # type: ignore[attr-defined]

    ok, _ = orchestrator.resume_trading()
    assert ok is True
    assert not config.manual_override.flag_path.exists()
    assert orchestrator._risk_engine.trading_halted() is False  # type: ignore[attr-defined]


def test_orchestrator_resync_and_rebuild_clear_state(tmp_path):
    config = _build_config(tmp_path)
    orchestrator = Orchestrator(config)

    book = OrderBook(
        figi="FIGI",
        timestamp=datetime.now(timezone.utc),
        bids=(OrderBookLevel(price=100.0, quantity=1.0),),
        asks=(OrderBookLevel(price=100.5, quantity=1.0),),
        depth=1,
    )
    orchestrator._data_engine.ingest_order_book(book)  # type: ignore[attr-defined]
    assert orchestrator._data_engine.last_window("FIGI")  # type: ignore[attr-defined]

    orchestrator.resynchronise_data()
    assert orchestrator._data_engine.last_window("FIGI") == ()  # type: ignore[attr-defined]

    orchestrator._feature_pipeline.transform([book])  # type: ignore[attr-defined]
    assert orchestrator._feature_pipeline._last_vector is not None  # type: ignore[attr-defined]
    orchestrator.rebuild_features()
    assert orchestrator._feature_pipeline._last_vector is None  # type: ignore[attr-defined]


def test_event_bus_registers_dashboard_commands(tmp_path):
    config = _build_config(tmp_path, bus_enabled=True)
    orchestrator = Orchestrator(config)
    orchestrator._start_event_bus_if_needed()
    try:
        handlers = orchestrator._event_bus._handlers  # type: ignore[attr-defined]
        expected = {
            "system.restart",
            "system.pause",
            "system.resume",
            "system.shutdown",
            "system.backtest",
            "system.backtest.create",
            "system.train",
            "ml.train",
            "ml.validate",
            "ml.rollback",
            "system.sandbox_forward",
            "system.forwardtest",
            "system.forwardtest.start",
            "system.production",
            "system.production.start",
            "risk.reset_stops",
            "exec.cancel_all",
            "features.resync",
            "features.rebuild",
            "adapter.refresh_tokens",
            "adapter.switch_sandbox",
            "bus.restart",
            "bus.test_event",
            "instrument.replace",
        }
        assert expected.issubset(set(handlers.keys()))
    finally:
        orchestrator._stop_event_bus()


def test_start_production_mode_requires_token(tmp_path):
    config = _build_config(tmp_path)
    config.datafeed.allow_tokenless = False  # force token requirement
    orchestrator = Orchestrator(config)

    ok, message = orchestrator.start_production_mode()
    assert ok is False
    assert "Production token" in message

    orchestrator._config.datafeed.production_token = "prod-real-token-123"  # type: ignore[attr-defined]
    orchestrator._config.system.mode = "forward-test"  # type: ignore[attr-defined]
    orchestrator._config.datafeed.use_sandbox = True  # type: ignore[attr-defined]

    ok, message = orchestrator.start_production_mode()
    assert ok is True
    assert "Production mode" in message
    assert orchestrator._config.system.mode == "production"  # type: ignore[attr-defined]
    assert orchestrator._config.datafeed.use_sandbox is False  # type: ignore[attr-defined]


def test_start_production_mode_tokenless_allowed(tmp_path):
    config = _build_config(tmp_path)
    config.datafeed.allow_tokenless = True
    orchestrator = Orchestrator(config)

    ok, message = orchestrator.start_production_mode()
    assert ok is True
    assert "paper mode" in message.lower()


def test_dashboard_status_exposes_adapter_logs(tmp_path):
    config = _build_config(tmp_path)
    orchestrator = Orchestrator(config)

    adapter_logger = logging.getLogger("scalp_system.broker.tinkoff")
    adapter_logger.setLevel(logging.INFO)
    adapter_logger.info("adapter connected")

    payload = orchestrator.dashboard_status()
    adapter_section = payload.get("adapter", {})
    assert "logs" in adapter_section
    assert any("adapter connected" in line for line in adapter_section.get("logs", []))
