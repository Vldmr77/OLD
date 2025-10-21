"""Basic health checks for system components."""
from __future__ import annotations

import argparse
from pathlib import Path

from scalp_system.config.loader import load_config
from scalp_system.storage.repository import SQLiteRepository


def run_health_checks(config_path: Path) -> None:
    config = load_config(config_path)
    SQLiteRepository(config.storage.base_path / "health_check.db")
    print("OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run system health checks")
    parser.add_argument("--config", type=Path, default=Path("config.example.yaml"))
    args = parser.parse_args()
    run_health_checks(args.config)


if __name__ == "__main__":
    main()
