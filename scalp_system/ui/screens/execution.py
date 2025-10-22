"""Execution screen for order management."""
from __future__ import annotations

from ..i18n import RU
from ..widgets.table import SimpleTable
from ..widgets.kv_meter import KeyValueMeter
from ..widgets.confirm import ask_confirmation
from .screen_utils import create_section

try:  # pragma: no cover
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    ttk = None  # type: ignore


class ExecutionScreen:
    def __init__(self, context) -> None:
        if ttk is None:
            return
        self._context = context
        frame = ttk.Frame(context.notebook, style="Dashboard.Section.TFrame")
        context.notebook.add(frame, text=context.strings["tab_execution"])

        queue_frame = create_section(frame, RU["exec_queue"])
        self._queue_table = SimpleTable(queue_frame, ["id", "figi", "state", "latency"], height=6)
        self._queue_table.pack(fill="both", expand=True)

        mode_frame = create_section(frame, RU["exec_mode"])
        self._mode = KeyValueMeter(mode_frame)
        self._mode.pack(anchor="w", padx=8, pady=4)

        trades_frame = create_section(frame, RU["exec_trades"])
        self._trades = SimpleTable(trades_frame, ["timestamp", "figi", "qty", "price"], height=5)
        self._trades.pack(fill="both", expand=True)

        stats_frame = create_section(frame, RU["exec_stats"])
        self._stats = KeyValueMeter(stats_frame)
        self._stats.pack(anchor="w", padx=8, pady=4)

        buttons = ttk.Frame(frame, style="Dashboard.Section.TFrame")
        buttons.pack(fill="x", padx=8, pady=4)
        self._buttons = [
            ttk.Button(buttons, text=RU["btn_pause"], command=lambda: context.emit("system.pause")),
            ttk.Button(buttons, text=RU["btn_resume"], command=lambda: context.emit("system.resume")),
            ttk.Button(buttons, text=RU["btn_cancel_orders"], command=self._cancel_all),
        ]
        for button in self._buttons:
            button.pack(side="left", padx=4)

    def _cancel_all(self) -> None:
        if ask_confirmation(RU["btn_cancel_orders"], RU["confirm_cancel_orders"]):
            self._context.emit("exec.cancel_all")

    def update(self) -> None:
        if ttk is None:
            return
        payload = self._context.state.get("execution", {})
        self._queue_table.update_rows(payload.get("queue", []))
        self._mode.update_metrics(payload.get("mode", {}))
        self._trades.update_rows(payload.get("fills", []))
        self._stats.update_metrics(payload.get("stats", {}))
        bus_up = self._context.bus.is_up()
        for button in getattr(self, "_buttons", []):
            button.configure(state="normal" if bus_up else "disabled")


__all__ = ["ExecutionScreen"]
