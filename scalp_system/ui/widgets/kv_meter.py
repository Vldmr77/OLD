"""Key-value metric badges for the dashboard."""
from __future__ import annotations

from ..i18n import RU

try:  # pragma: no cover - tkinter optional
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    ttk = None  # type: ignore


class KeyValueMeter(ttk.Frame):  # type: ignore[misc]
    """Display metrics as compact coloured badges."""

    def __init__(self, master: tk.Misc) -> None:  # type: ignore[valid-type]
        super().__init__(master, style="Dashboard.Section.TFrame")
        self._cards: dict[str, tuple[ttk.Label, ttk.Label]] = {}

    def update_metrics(self, metrics: dict[str, object]) -> None:
        self._cards = {}
        for widget in self.winfo_children():
            widget.destroy()

        for index, (name, value) in enumerate(metrics.items()):
            row, column = 0, index
            card = ttk.Frame(self, style="DashboardBadge.TFrame")
            card.grid(row=row, column=column, padx=6, pady=6, sticky="nsew")
            key = str(name)
            display_name = RU.get(f"metric_{key}", key.replace("_", " ").upper())
            title = ttk.Label(card, text=display_name, style="DashboardBadge.Title.TLabel")
            title.pack(anchor="w")
            value_label = ttk.Label(card, text=str(value), style="DashboardBadge.Value.TLabel")
            value_label.pack(anchor="w", pady=(2, 0))
            self._cards[name] = (title, value_label)

        for column in range(len(metrics)):
            self.columnconfigure(column, weight=1)


__all__ = ["KeyValueMeter"]
