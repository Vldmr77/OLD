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
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to serve on")
    args = parser.parse_args(argv)

    run_dashboard(args.repository, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
