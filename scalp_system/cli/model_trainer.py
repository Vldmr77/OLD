"""Offline model trainer stub."""
from __future__ import annotations

import argparse
from pathlib import Path

from scalp_system.config.loader import load_config


def _normalise(values: list[float]) -> list[float]:
    total = sum(values)
    if not total:
        return [1.0 / len(values) for _ in values]
    return [value / total for value in values]


def calibrate(days: int, config_path: Path) -> None:
    """Normalise ensemble weights and persist them next to the config."""
    config = load_config(config_path)
    weights = [
        config.ml.weights.lstm_ob,
        config.ml.weights.gbdt_features,
        config.ml.weights.transformer_temporal,
        config.ml.weights.svm_volatility,
    ]
    normalised = _normalise(weights)
    config.ml.weights.lstm_ob = normalised[0]
    config.ml.weights.gbdt_features = normalised[1]
    config.ml.weights.transformer_temporal = normalised[2]
    config.ml.weights.svm_volatility = normalised[3]
    output = config_path.with_suffix(".calibrated.json")
    output.write_text(config.json(indent=2), encoding="utf-8")
    print(f"Calibrated weights saved to {output}")


def main() -> None:
    """CLI entry point for running calibration stubs."""
    parser = argparse.ArgumentParser(description="Model trainer")
    parser.add_argument("calibrate", nargs="?")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--config", type=Path, default=Path("config.example.yaml"))
    args = parser.parse_args()
    calibrate(args.days, args.config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
