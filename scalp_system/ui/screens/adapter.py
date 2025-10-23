"""Tinkoff adapter screen."""
from __future__ import annotations

from ..i18n import RU
from ..widgets.kv_meter import KeyValueMeter
from ..widgets.log_tail import LogTail
from .screen_utils import create_section

try:  # pragma: no cover
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    ttk = None  # type: ignore


class AdapterScreen:
    def __init__(self, context) -> None:
        if ttk is None:
            return
        self._context = context
        frame = ttk.Frame(context.notebook, style="Dashboard.Section.TFrame")
        context.notebook.add(frame, text=context.strings["tab_adapter"])

        tokens_frame = create_section(frame, RU["adapter_tokens"])
        self._sandbox_var = tk.StringVar()
        self._production_var = tk.StringVar()
        ttk.Label(tokens_frame, text=RU["token_sandbox"]).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        sandbox_entry = ttk.Entry(tokens_frame, textvariable=self._sandbox_var, width=60)
        sandbox_entry.grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        ttk.Label(tokens_frame, text=RU["token_production"]).grid(row=1, column=0, sticky="w", padx=4, pady=2)
        production_entry = ttk.Entry(tokens_frame, textvariable=self._production_var, width=60)
        production_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        tokens_frame.columnconfigure(1, weight=1)
        for entry in (sandbox_entry, production_entry):
            entry.bind("<Control-c>", lambda e: e.widget.event_generate("<<Copy>>"))
            entry.bind("<Control-v>", lambda e: e.widget.event_generate("<<Paste>>"))

        ttk.Label(tokens_frame, text=RU["token_hint"]).grid(row=2, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Button(tokens_frame, text=RU["btn_save_tokens"], command=self._save_tokens).grid(
            row=3, column=0, padx=4, pady=4, sticky="w"
        )

        quota_frame = create_section(frame, RU["adapter_quota"])
        self._quota = KeyValueMeter(quota_frame)
        self._quota.pack(anchor="w", padx=8, pady=4)

        error_frame = create_section(frame, RU["adapter_errors"])
        self._errors = KeyValueMeter(error_frame)
        self._errors.pack(anchor="w", padx=8, pady=4)

        diag_frame = create_section(frame, RU["adapter_diag"])
        self._diag = KeyValueMeter(diag_frame)
        self._diag.pack(anchor="w", padx=8, pady=4)

        logs_frame = create_section(frame, RU["adapter_logs"])
        self._log_tail = LogTail(logs_frame)
        self._log_tail.pack(fill="both", expand=True)
        self._last_log_count = 0

        buttons = ttk.Frame(frame, style="Dashboard.Section.TFrame")
        buttons.pack(fill="x", padx=8, pady=4)
        self._bus_buttons = [
            ttk.Button(buttons, text=RU["btn_update_quota"], command=lambda: context.emit("adapter.refresh_tokens")),
            ttk.Button(buttons, text=RU["btn_switch_sandbox"], command=lambda: context.emit("adapter.switch_sandbox")),
        ]
        for button in self._bus_buttons:
            button.pack(side="left", padx=4)
        self._refresh_tokens()

    def _refresh_tokens(self) -> None:
        status = self._context.token_status()
        sandbox = status.get("sandbox", False)
        production = status.get("production", False)
        if sandbox:
            self._sandbox_var.set("••••••")
        if production:
            self._production_var.set("••••••")

    def _save_tokens(self) -> None:
        sandbox = self._sandbox_var.get() or None
        production = self._production_var.get() or None
        self._context.save_tokens(sandbox, production)
        self._refresh_tokens()

    def update(self) -> None:
        if ttk is None:
            return
        payload = self._context.state.get("adapter", {})
        self._quota.update_metrics(payload.get("quota", {}))
        self._errors.update_metrics(payload.get("errors", {}))
        self._diag.update_metrics(payload.get("diagnostics", {}))
        lines = payload.get("logs", []) if isinstance(payload, dict) else []
        if isinstance(lines, list):
            if len(lines) < getattr(self, "_last_log_count", 0):
                self._log_tail.set_lines(lines)
            else:
                self._log_tail.extend(lines[getattr(self, "_last_log_count", 0) :])
            self._last_log_count = len(lines)
        for button in getattr(self, "_bus_buttons", []):
            button.configure(state="normal" if self._context.bus.is_up() else "disabled")


__all__ = ["AdapterScreen"]
