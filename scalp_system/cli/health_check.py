"""Basic health checks for system components."""
from __future__ import annotations

import argparse
from pathlib import Path

from scalp_system.config.loader import load_config
from scalp_system.ml.calibration import CalibrationCoordinator
from scalp_system.monitoring.drift import DriftDetector
from scalp_system.storage.repository import SQLiteRepository


def run_health_checks(config_path: Path) -> None:
    """Validate that critical runtime components can be initialised."""
    config = load_config(config_path)
    SQLiteRepository(config.storage.base_path / "health_check.db")
    DriftDetector(
        threshold=config.ml.drift_threshold,
        history_dir=config.storage.base_path / "drift_metrics",
    )
    CalibrationCoordinator(
        queue_path=config.storage.base_path / "calibration_queue.jsonl"
    )
    print("OK")


def main() -> None:
    """CLI entry point for running health checks."""
    parser = argparse.ArgumentParser(description="Run system health checks")
    parser.add_argument("--config", type=Path, default=Path("config.example.yaml"))
    args = parser.parse_args()
    run_health_checks(args.config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
