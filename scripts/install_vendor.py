#!/usr/bin/env python3
"""Install project dependencies into a bundled vendor directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

DEFAULT_TARGET = Path(__file__).resolve().parents[1] / "scalp_system" / "vendor"
DEFAULT_SDK = Path(__file__).resolve().parents[1] / "tinkoff_investments-0.2.0b117-py3-none-any.whl"

# Essential dependencies required for the runtime to start.
CORE_REQUIREMENTS: Sequence[str] = ("PyYAML>=6.0",)

# Optional extras that provide enhanced functionality but are not mandatory for
# a smoke run of the system.
OPTIONAL_REQUIREMENTS: Sequence[str] = (
    "cryptography>=41.0",  # encrypted token storage
    "psutil>=5.9",  # resource monitoring helpers
)


def _pip_install(args: Iterable[str]) -> None:
    """Invoke pip with the provided arguments."""

    command: List[str] = [sys.executable, "-m", "pip", "install", *args]
    subprocess.check_call(command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Install runtime dependencies into a local vendor directory so the system "
            "can run without global site-packages."
        )
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Directory where dependencies will be installed (default: scalp_system/vendor)",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=None,
        help=(
            "Optional path to an extra requirements file. Use this to pin bespoke"
            " dependencies beyond the curated runtime set."
        ),
    )
    parser.add_argument(
        "--sdk-wheel",
        type=Path,
        default=DEFAULT_SDK,
        help="Path to the tinkoff investments SDK wheel",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=(),
        help="Additional pip requirement specifiers to install",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help=(
            "Install optional extras (encryption, resource monitoring). These pull"
            " in heavier dependencies like cryptography."
        ),
    )
    parser.add_argument(
        "--no-sdk",
        action="store_true",
        help="Skip installing the tinkoff SDK wheel",
    )
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    install_args = ["--target", str(target)]

    # Install the essential requirements one by one to keep transitive
    # dependencies to the necessary minimum.
    for requirement in CORE_REQUIREMENTS:
        _pip_install([*install_args, requirement])

    if args.include_optional:
        for requirement in OPTIONAL_REQUIREMENTS:
            _pip_install([*install_args, requirement])

    if args.requirements and args.requirements.exists():
        _pip_install([*install_args, "-r", str(args.requirements)])

    if not args.no_sdk and args.sdk_wheel.exists():
        _pip_install([*install_args, str(args.sdk_wheel)])

    if args.extra:
        _pip_install([*install_args, *args.extra])

    print(f"Dependencies installed into {target}")


if __name__ == "__main__":
    main()
