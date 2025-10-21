"""Configuration package utilities and defaults."""
from __future__ import annotations

from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.example.yaml"

from .loader import load_config  # noqa: E402  # circular-safe import

__all__ = ["DEFAULT_CONFIG_PATH", "load_config"]
