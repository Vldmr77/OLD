"""Run the monitoring dashboard."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import loader as config_loader
from ..config.loader import load_config
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
    args = parser.parse_args(argv)

    config_path = args.config
    if config_path is None:
        config_path = config_loader._discover_default_path()
    cfg = load_config(config_path)

    bus_client: BusClient | None = None
    if cfg.bus.enabled:
        candidate = BusClient(host=cfg.bus.host, port=cfg.bus.port)
        if candidate.check_available():
            bus_client = candidate
        else:
            LOGGER.warning(
                "Control bus unavailable at %s:%s; dashboard controls disabled",
                cfg.bus.host,
                cfg.bus.port,
            )

    run_dashboard(
        args.repository,
        refresh_interval_ms=args.refresh_interval,
        signal_limit=args.signal_limit,
        title=args.title,
        headless=args.headless,
        config_path=config_path,
        bus_address=(cfg.bus.host, cfg.bus.port)
        if bus_client is not None
        else None,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
