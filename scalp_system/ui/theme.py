"""Shared colour palette and ttk theme helpers for the dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:  # pragma: no cover - tkinter optional in CI
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    ttk = None  # type: ignore


@dataclass(frozen=True)
class Palette:
    background: str = "#0f172a"
    surface: str = "#111827"
    surface_alt: str = "#1f2937"
    outline: str = "#1e293b"
    text: str = "#e5e7eb"
    text_muted: str = "#94a3b8"
    accent: str = "#3b82f6"
    accent_alt: str = "#22c55e"
    warning: str = "#f59e0b"
    error: str = "#ef4444"


PALETTE = Palette()


def apply_theme(root: "tk.Misc") -> None:  # type: ignore[name-defined]
    """Apply the dashboard ttk theme to the given root widget."""

    if ttk is None:  # pragma: no cover - tk unavailable
        return

    style = ttk.Style(root)
    try:  # pragma: no cover - fallback on Windows classic theme
        style.theme_use("clam")
    except tk.TclError:  # type: ignore[attr-defined]
        pass

    base_font = ("Segoe UI", 10)
    heading_font = ("Segoe UI Semibold", 11)

    if hasattr(root, "option_add"):
        root.option_add("*Font", base_font)

    root.configure(bg=PALETTE.background)

    style.configure("TFrame", background=PALETTE.background)
    style.configure("TLabel", background=PALETTE.surface, foreground=PALETTE.text)
    style.configure(
        "TButton",
        font=heading_font,
        padding=8,
        background=PALETTE.accent,
        foreground="white",
        borderwidth=0,
        focusthickness=0,
        relief="flat",
    )
    style.map(
        "TButton",
        background=[("active", PALETTE.accent_alt), ("disabled", PALETTE.surface_alt)],
        foreground=[("disabled", PALETTE.text_muted)],
    )

    style.configure(
        "Dashboard.TNotebook",
        background=PALETTE.background,
        borderwidth=0,
    )
    style.configure(
        "Dashboard.TNotebook.Tab",
        font=heading_font,
        padding=(12, 6),
        background=PALETTE.surface,
        foreground=PALETTE.text_muted,
    )
    style.map(
        "Dashboard.TNotebook.Tab",
        background=[("selected", PALETTE.surface_alt)],
        foreground=[("selected", PALETTE.text)],
    )

    style.configure(
        "Dashboard.TLabelframe",
        background=PALETTE.surface,
        foreground=PALETTE.text,
        bordercolor=PALETTE.outline,
        relief="groove",
        padding=12,
    )
    style.configure(
        "Dashboard.TLabelframe.Label",
        background=PALETTE.surface,
        foreground=PALETTE.accent,
        font=("Segoe UI Semibold", 10),
        padding=(4, 0, 4, 0),
    )

    style.configure(
        "Dashboard.Section.TFrame",
        background=PALETTE.surface,
        relief="flat",
        padding=8,
    )

    style.configure(
        "Dashboard.Treeview",
        background=PALETTE.surface,
        foreground=PALETTE.text,
        fieldbackground=PALETTE.surface,
        borderwidth=0,
        rowheight=26,
    )
    style.map(
        "Dashboard.Treeview",
        background=[("selected", PALETTE.accent)],
        foreground=[("selected", "white")],
    )
    style.configure(
        "Dashboard.Treeview.Heading",
        background=PALETTE.surface_alt,
        foreground=PALETTE.text,
        relief="flat",
        padding=(6, 4),
        font=("Segoe UI Semibold", 10),
    )
    style.map(
        "Dashboard.Treeview.Heading",
        background=[("active", PALETTE.accent)],
        foreground=[("active", "white")],
    )

    style.configure(
        "DashboardBadge.TFrame",
        background=PALETTE.surface_alt,
        relief="flat",
        padding=10,
    )
    style.configure(
        "DashboardBadge.Title.TLabel",
        background=PALETTE.surface_alt,
        foreground=PALETTE.text_muted,
        font=("Segoe UI", 9),
    )
    style.configure(
        "DashboardBadge.Value.TLabel",
        background=PALETTE.surface_alt,
        foreground=PALETTE.text,
        font=("Segoe UI Semibold", 12),
    )

    style.configure(
        "Dashboard.Stage.TLabel",
        background=PALETTE.surface_alt,
        foreground=PALETTE.text,
        font=("Segoe UI", 10),
        anchor="center",
        padding=(10, 8),
    )

    style.configure(
        "DashboardBanner.TFrame",
        background=PALETTE.surface,
        padding=(16, 12),
    )
    style.configure(
        "DashboardBanner.Title.TLabel",
        background=PALETTE.surface,
        foreground=PALETTE.text,
        font=("Segoe UI Semibold", 16),
    )
    style.configure(
        "DashboardBanner.Subtitle.TLabel",
        background=PALETTE.surface,
        foreground=PALETTE.text_muted,
        font=("Segoe UI", 10),
    )


def tag_colours() -> Dict[str, str]:
    return {
        "odd": PALETTE.surface,
        "even": PALETTE.surface_alt,
        "error": PALETTE.error,
        "warning": PALETTE.warning,
        "ok": PALETTE.accent_alt,
    }


__all__ = ["PALETTE", "apply_theme", "tag_colours"]
