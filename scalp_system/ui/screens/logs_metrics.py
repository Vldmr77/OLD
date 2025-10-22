"""Logs and metrics screen."""
from __future__ import annotations

from ..i18n import RU
from ..widgets.log_tail import LogTail
from ..widgets.kv_meter import KeyValueMeter
from .screen_utils import create_section

try:  # pragma: no cover
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    ttk = None  # type: ignore


class LogsMetricsScreen:
    def __init__(self, context) -> None:
        if ttk is None:
            return
        self._context = context
        frame = ttk.Frame(context.notebook)
        context.notebook.add(frame, text=context.strings["tab_logs"])

        filters = create_section(frame, RU["logs_filters"])
        ttk.Label(filters, text="Модуль").grid(row=0, column=0, padx=4, pady=2)
        self._module_var = tk.StringVar()
        ttk.Entry(filters, textvariable=self._module_var, width=20).grid(row=0, column=1, padx=4, pady=2)
        ttk.Label(filters, text="Уровень").grid(row=0, column=2, padx=4, pady=2)
        self._level_var = tk.StringVar()
        ttk.Entry(filters, textvariable=self._level_var, width=10).grid(row=0, column=3, padx=4, pady=2)
        ttk.Label(filters, text="Текст").grid(row=0, column=4, padx=4, pady=2)
        self._text_var = tk.StringVar()
        ttk.Entry(filters, textvariable=self._text_var, width=20).grid(row=0, column=5, padx=4, pady=2)

        logs_frame = create_section(frame, RU["logs_tail"])
        self._tail = LogTail(logs_frame)
        self._tail.pack(fill="both", expand=True)

        metrics_frame = create_section(frame, RU["metrics_header"])
        self._metrics = KeyValueMeter(metrics_frame)
        self._metrics.pack(anchor="w", padx=8, pady=4)

    def update(self) -> None:
        if ttk is None:
            return
        payload = self._context.state.get("logs", {})
        lines = payload.get("tail", [])
        if isinstance(lines, list):
            self._tail.extend(lines)
        self._metrics.update_metrics(payload.get("metrics", {}))


__all__ = ["LogsMetricsScreen"]
