#!/usr/bin/env python3
"""Convenience launcher for the scalping system and dashboard UI.

The script keeps the repository root on ``sys.path`` so it can be started from an
IDE shortcut or a double-click. By default it behaves like ``python -m
scalp_system`` and accepts the same arguments. Passing ``--dashboard`` (or
``-d``) switches the launcher into UI mode and forwards the remaining arguments
to ``python -m scalp_system.cli.dashboard``.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def _prepare_environment() -> None:
    """Ensure the repository root is on ``sys.path`` and set as CWD."""

    repo_root = Path(__file__).resolve().parent
    package_root = repo_root / "scalp_system"

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    if Path.cwd() != repo_root:
        os.chdir(repo_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the scalping system orchestrator or dashboard. Any additional "
            "arguments are forwarded to the selected entry point."
        )
    )
    parser.add_argument(
        "-d",
        "--dashboard",
        action="store_true",
        help="Launch the dashboard UI instead of the trading orchestrator",
    )
    return parser


def build_invocation(argv: Sequence[str] | None = None) -> Tuple[str, List[str]]:
    """Return the module name and arguments that should be executed."""

    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    parsed, remainder = parser.parse_known_args(list(argv))

    target_module = (
        "scalp_system.cli.dashboard" if parsed.dashboard else "scalp_system"
    )

    forwarded: List[str] = list(remainder)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    return target_module, forwarded


def _launch_module(module: str, argv: Iterable[str]) -> None:
    sys.argv = [module, *argv]
    if module == "scalp_system":
        from scalp_system.__main__ import main as orchestrator_main

        orchestrator_main(list(argv))
    elif module == "scalp_system.cli.dashboard":
        from scalp_system.cli.dashboard import main as dashboard_main

        dashboard_main(list(argv))
    else:  # pragma: no cover - future proofing
        sys.modules.pop(module, None)
        runpy.run_module(module, run_name="__main__")


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point used by both terminal and shortcut launches."""

    _prepare_environment()
    module, forwarded = build_invocation(argv)
    _launch_module(module, forwarded)


if __name__ == "__main__":
    main()
