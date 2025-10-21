#!/usr/bin/env python3
"""Install project dependencies into a bundled vendor directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

DEFAULT_TARGET = Path(__file__).resolve().parents[1] / "scalp_system" / "vendor"
DEFAULT_REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"
DEFAULT_SDK = Path(__file__).resolve().parents[1] / "tinkoff_investments-0.2.0b117-py3-none-any.whl"


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
        default=DEFAULT_REQUIREMENTS,
        help="Path to requirements file to install before the SDK wheel",
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
        "--no-requirements",
        action="store_true",
        help="Skip installing requirements.txt",
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

    if not args.no_requirements and args.requirements.exists():
        _pip_install([*install_args, "-r", str(args.requirements)])

    if not args.no_sdk and args.sdk_wheel.exists():
        _pip_install([*install_args, str(args.sdk_wheel)])

    if args.extra:
        _pip_install([*install_args, *args.extra])

    print(f"Dependencies installed into {target}")


if __name__ == "__main__":
    main()
