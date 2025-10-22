"""Confirmation dialog helper."""
from __future__ import annotations

try:  # pragma: no cover - tkinter optional
    from tkinter import messagebox  # type: ignore
except Exception:  # pragma: no cover
    messagebox = None  # type: ignore


def ask_confirmation(title: str, message: str) -> bool:
    if messagebox is None:  # pragma: no cover - headless fallback
        return False
    return bool(messagebox.askokcancel(title, message))


__all__ = ["ask_confirmation"]
