from __future__ import annotations

import argparse
from pathlib import Path

from scalp_system.config import DEFAULT_CONFIG_PATH
from scalp_system.orchestrator import run_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Scalping system orchestrator")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration (defaults to packaged example)",
    )
    args = parser.parse_args()
    run_from_yaml(args.config)


if __name__ == "__main__":
    main()
