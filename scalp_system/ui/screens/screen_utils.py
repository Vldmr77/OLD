"""Utility helpers shared between screens."""
from __future__ import annotations

try:  # pragma: no cover - tkinter optional
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    ttk = None  # type: ignore


def create_section(parent, title: str, *, pack: bool = True, **pack_kwargs):
    if ttk is None:
        return None
    frame = ttk.LabelFrame(parent, text=title)
    if pack:
        options = {"fill": "both", "expand": True, "padx": 8, "pady": 6}
        options.update(pack_kwargs)
        frame.pack(**options)
    elif pack_kwargs:
        frame.pack(**pack_kwargs)
    return frame


__all__ = ["create_section"]
