#!/usr/bin/env python3
"""Convenience launcher for the scalping system orchestrator."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


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


def main() -> None:
    """Launch the orchestrator exactly like ``python -m scalp_system``."""

    _prepare_environment()
    runpy.run_module("scalp_system", run_name="__main__")


if __name__ == "__main__":
    main()
