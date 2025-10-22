"""Scalping system package initialisation."""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_PATH = Path(__file__).resolve().parent / "vendor"
if _VENDOR_PATH.exists() and str(_VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(_VENDOR_PATH))

__all__ = []
