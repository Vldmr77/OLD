"""Initialises configuration files for different environments."""
from __future__ import annotations

import argparse
from pathlib import Path

from scalp_system.config.base import OrchestratorConfig
from scalp_system.config.loader import load_config


def init_config(env: str, output: Path) -> None:
    base_config = load_config()
    base_config.environment = env
    output.write_text(base_config.json(indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise configuration")
    parser.add_argument("--env", default="development")
    parser.add_argument("--output", type=Path, default=Path("config.json"))
    args = parser.parse_args()
    init_config(args.env, args.output)


if __name__ == "__main__":
    main()
