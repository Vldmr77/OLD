"""Run the monitoring dashboard."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import loader as config_loader
from ..config.loader import load_config
from ..config.token_prompt import store_tokens, token_status
from ..control.bus import BusClient
from ..ui import run_dashboard

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the scalping dashboard UI")
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("./runtime/signals.db"),
        help="Path to the SQLite repository file",
    )
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=1000,
        help="Refresh interval in milliseconds",
    )
    parser.add_argument(
        "--signal-limit",
        type=int,
        default=25,
        help="Maximum number of signals to display",
    )
    parser.add_argument(
        "--title",
        default="Scalp System Dashboard",
        help="Window title override",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a Tk window (useful for tests)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to the YAML configuration for token management",
    )
    parser.add_argument(
        "--status-endpoint",
        help="Optional HTTP endpoint that exposes orchestrator status",
    )
    parser.add_argument(
        "--bus-host",
        help="Override control bus host (defaults to config)",
    )
    parser.add_argument(
        "--bus-port",
        type=int,
        help="Override control bus port (defaults to config)",
    )
    args = parser.parse_args(argv)

    config_path = args.config
    if config_path is None:
        config_path = config_loader._discover_default_path()
    cfg = load_config(config_path)

    bus_address: tuple[str, int] | None = None
    if cfg.bus.enabled:
        bus_host = args.bus_host or cfg.bus.host
        bus_port = args.bus_port or cfg.bus.port
        bus_address = (bus_host, bus_port)
        candidate = BusClient(host=bus_host, port=bus_port)
        if candidate.check_available():
            LOGGER.debug("Control bus reachable at %s:%s", bus_host, bus_port)
        else:
            LOGGER.warning(
                "Control bus unavailable at %s:%s; controls will retry in UI",
                bus_host,
                bus_port,
            )

    def _token_writer(sandbox: str | None, production: str | None) -> bool:
        try:
            return store_tokens(config_path, sandbox=sandbox, production=production)
        except RuntimeError as exc:
            LOGGER.error("Token persistence failed: %s", exc)
            return False

    def _token_status() -> dict[str, bool]:
        try:
            return token_status(config_path)
        except RuntimeError as exc:
            LOGGER.error("Token status lookup failed: %s", exc)
            return {"sandbox": False, "production": False}

    run_dashboard(
        args.repository,
        refresh_interval_ms=args.refresh_interval,
        signal_limit=args.signal_limit,
        title=args.title,
        headless=args.headless,
        config_path=config_path,
        token_writer=_token_writer,
        token_status_provider=_token_status,
        bus_address=bus_address,
        status_endpoint=args.status_endpoint,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
