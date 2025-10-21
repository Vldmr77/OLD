"""CLI entry points for model calibration and training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def main() -> None:
    """CLI entry point for running calibration and training workflows."""
    parser = argparse.ArgumentParser(description="Model trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = subparsers.add_parser("calibrate", help="Normalise ensemble weights")
    calibrate_parser.add_argument(
        "--config", type=Path, default=Path("config.example.yaml"), help="Path to config file"
    )

    train_parser = subparsers.add_parser("train", help="Run offline training")
    train_parser.add_argument(
        "--config", type=Path, default=Path("config.example.yaml"), help="Path to config file"
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

    args = parser.parse_args()

    if args.command == "calibrate":
        output = calibrate(args.config)
        print(f"Calibrated weights saved to {output}")
    else:
        overrides: dict[str, Any] = {}
        training_overrides: dict[str, Any] = {}
        if args.dataset:
            training_overrides["dataset_path"] = str(args.dataset)
        if args.output:
            training_overrides["output_dir"] = str(args.output)
        if args.epochs is not None:
            training_overrides["epochs"] = args.epochs
        if args.learning_rate is not None:
            training_overrides["learning_rate"] = args.learning_rate
        if args.validation_split is not None:
            training_overrides["validation_split"] = args.validation_split
        if args.min_samples is not None:
            training_overrides["min_samples"] = args.min_samples
        if training_overrides:
            overrides["training"] = training_overrides
        artefact = run_training(args.config, overrides)
        print(f"Training artefacts saved to {artefact}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
