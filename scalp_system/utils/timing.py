"""Utility helpers."""
from __future__ import annotations

import contextlib
import time
from typing import Iterator


@contextlib.contextmanager
def timed(stage: str, callback):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        callback(stage, elapsed)


__all__ = ["timed"]
