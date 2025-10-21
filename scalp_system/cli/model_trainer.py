"""CLI entry points for model calibration and training."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from scalp_system.config import DEFAULT_CONFIG_PATH
from scalp_system.config.loader import load_config
from scalp_system.ml.training import ModelTrainer


def calibrate(config_path: Path) -> Path:
    """Normalise ensemble weights and persist them next to the config."""
    config = load_config(config_path)
    config.ml.weights.normalise()
    output = config_path.with_suffix(".calibrated.json")
    output.write_text(config.json(indent=2), encoding="utf-8")
    return output


def run_training(config_path: Path, overrides: dict[str, Any]) -> Path:
    """Execute offline training based on the provided configuration overrides."""
    config = load_config(config_path, **overrides)
    trainer = ModelTrainer(config.training)
    report = trainer.train()
    summary = config_path.with_suffix(".training.json")
    summary.write_text(
        json.dumps(
            {
                "samples": report.samples,
                "epochs": report.epochs,
                "training_loss": report.training_loss,
                "validation_loss": report.validation_loss,
                "weights": {
                    "lstm_ob": report.weights.lstm_ob,
                    "gbdt_features": report.weights.gbdt_features,
                    "transformer_temporal": report.weights.transformer_temporal,
                    "svm_volatility": report.weights.svm_volatility,
                },
                "artefact": str(report.output_path),
                "quantization_plan": str(report.quantization_plan) if report.quantization_plan else None,
                "quantization": [
                    {
                        "model": decision.model_name,
                        "precision": decision.precision,
                        "weight": decision.weight,
                        "converter": decision.converter_directive,
                    }
                    for decision in report.quantization_decisions
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report.output_path


def run_daemon(
    config_path: Path,
    overrides: dict[str, Any],
    *,
    interval_minutes: float = 60.0,
    iterations: Optional[int] = None,
    runner: Callable[[Path, dict[str, Any]], Path] = run_training,
) -> Iterable[Path]:
    """Continuously execute training jobs on a fixed interval.

    Parameters
    ----------
    config_path:
        Path to the configuration file used for each run.
    overrides:
        Additional configuration overrides that should be applied to each
        invocation. These mirror the arguments accepted by ``run_training``.
    interval_minutes:
        Delay between subsequent runs. The default of 60 minutes matches the
        operator runbook in the technical specification.
    iterations:
        Optional maximum number of iterations. When ``None`` the daemon runs
        until it is interrupted.
    runner:
        Callable used to perform the training step. Injected in tests to avoid
        heavy work while preserving behaviour.

    Yields
    ------
    Path
        Location of the most recent training artefact produced by ``runner``.
    """

    interval_seconds = max(0.0, float(interval_minutes) * 60.0)
    completed = 0
    while iterations is None or completed < iterations:
        artefact = runner(config_path, overrides)
        completed += 1
        yield artefact
        if iterations is not None and completed >= iterations:
            break
        if interval_seconds:
            time.sleep(interval_seconds)


def main() -> None:
    """CLI entry point for running calibration and training workflows."""
    parser = argparse.ArgumentParser(description="Model trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = subparsers.add_parser("calibrate", help="Normalise ensemble weights")
    calibrate_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config file (defaults to packaged example)",
    )

    train_parser = subparsers.add_parser("train", help="Run offline training")
    train_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config file (defaults to packaged example)",
    )
    train_parser.add_argument(
        "--dataset", type=Path, help="Override training dataset path"
    )
    train_parser.add_argument(
        "--output", type=Path, help="Override directory for trained artefacts"
    )
    train_parser.add_argument("--epochs", type=int, help="Override number of epochs")
    train_parser.add_argument(
        "--learning-rate", dest="learning_rate", type=float, help="Override learning rate"
    )
    train_parser.add_argument(
        "--validation-split", dest="validation_split", type=float, help="Override validation split"
    )
    train_parser.add_argument(
        "--min-samples", dest="min_samples", type=int, help="Override minimum sample count"
    )

    daemon_parser = subparsers.add_parser(
        "daemon", help="Run continuous training cycles as a background helper"
    )
    daemon_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config file (defaults to packaged example)",
    )
    daemon_parser.add_argument(
        "--dataset", type=Path, help="Override training dataset path"
    )
    daemon_parser.add_argument(
        "--output", type=Path, help="Override directory for trained artefacts"
    )
    daemon_parser.add_argument(
        "--epochs", type=int, help="Override number of epochs"
    )
    daemon_parser.add_argument(
        "--learning-rate", dest="learning_rate", type=float, help="Override learning rate"
    )
    daemon_parser.add_argument(
        "--validation-split", dest="validation_split", type=float, help="Override validation split"
    )
    daemon_parser.add_argument(
        "--min-samples", dest="min_samples", type=int, help="Override minimum sample count"
    )
    daemon_parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Minutes to wait between training iterations (default: 60)",
    )
    daemon_parser.add_argument(
        "--max-iterations",
        type=int,
        dest="max_iterations",
        help="Run the daemon for a fixed number of iterations (default: infinite)",
    )

    args = parser.parse_args()

    if args.command == "calibrate":
        output = calibrate(args.config)
        print(f"Calibrated weights saved to {output}")
        return

    overrides: dict[str, Any] = {}
    training_overrides: dict[str, Any] = {}
    if getattr(args, "dataset", None):
        training_overrides["dataset_path"] = str(args.dataset)
    if getattr(args, "output", None):
        training_overrides["output_dir"] = str(args.output)
    if getattr(args, "epochs", None) is not None:
        training_overrides["epochs"] = args.epochs
    if getattr(args, "learning_rate", None) is not None:
        training_overrides["learning_rate"] = args.learning_rate
    if getattr(args, "validation_split", None) is not None:
        training_overrides["validation_split"] = args.validation_split
    if getattr(args, "min_samples", None) is not None:
        training_overrides["min_samples"] = args.min_samples
    if training_overrides:
        overrides["training"] = training_overrides

    if args.command == "train":
        artefact = run_training(args.config, overrides)
        print(f"Training artefacts saved to {artefact}")
        return

    try:
        for artefact in run_daemon(
            args.config,
            overrides,
            interval_minutes=args.interval,
            iterations=args.max_iterations,
        ):
            print(f"Training artefacts saved to {artefact}")
    except KeyboardInterrupt:  # pragma: no cover - interactive guard
        print("Daemon interrupted by user")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
