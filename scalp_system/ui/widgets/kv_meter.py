"""Key-value metric badges for the dashboard."""
from __future__ import annotations

try:  # pragma: no cover - tkinter optional
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    ttk = None  # type: ignore


class KeyValueMeter(ttk.Frame):  # type: ignore[misc]
    """Display a small table of metrics."""

    def __init__(self, master: tk.Misc) -> None:  # type: ignore[valid-type]
        super().__init__(master)
        self._labels: dict[str, tk.StringVar] = {}

    def update_metrics(self, metrics: dict[str, object]) -> None:
        for index, (name, value) in enumerate(metrics.items()):
            if name not in self._labels:
                label = ttk.Label(self, text=f"{name}: ")
                var = tk.StringVar(value=str(value))
                value_label = ttk.Label(self, textvariable=var)
                label.grid(row=index, column=0, sticky="w", padx=4)
                value_label.grid(row=index, column=1, sticky="w", padx=4)
                self._labels[name] = var
            else:
                self._labels[name].set(str(value))


__all__ = ["KeyValueMeter"]
