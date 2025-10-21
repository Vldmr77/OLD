from __future__ import annotations

import argparse
from pathlib import Path

from .orchestrator import run_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Scalping system orchestrator")
    parser.add_argument("config", type=Path, help="Path to YAML configuration")
    args = parser.parse_args()
    run_from_yaml(args.config)


if __name__ == "__main__":
    main()
