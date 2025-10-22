"""Run the monitoring dashboard."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..ui import run_dashboard


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the scalping dashboard UI")
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("./runtime/signals.sqlite3"),
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

    run_dashboard(
        args.repository,
        refresh_interval_ms=args.refresh_interval,
        signal_limit=args.signal_limit,
        title=args.title,
        headless=args.headless,
        config_path=args.config,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
