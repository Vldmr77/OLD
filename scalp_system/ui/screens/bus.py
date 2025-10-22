"""Event bus diagnostics screen."""
from __future__ import annotations

from ..i18n import RU
from ..widgets.table import SimpleTable
from ..widgets.kv_meter import KeyValueMeter
from .screen_utils import create_section

try:  # pragma: no cover
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    ttk = None  # type: ignore


class BusScreen:
    def __init__(self, context) -> None:
        if ttk is None:
            return
        self._context = context
        frame = ttk.Frame(context.notebook)
        context.notebook.add(frame, text=context.strings["tab_bus"])

        topics = create_section(frame, RU["bus_topics"])
        self._topics = SimpleTable(topics, ["topic", "subscribers"], height=5)
        self._topics.pack(fill="both", expand=True)

        metrics = create_section(frame, RU["bus_metrics"])
        self._metrics = KeyValueMeter(metrics)
        self._metrics.pack(anchor="w", padx=8, pady=4)

        events = create_section(frame, RU["bus_events"])
        self._events = SimpleTable(events, ["timestamp", "type", "payload"], height=6)
        self._events.pack(fill="both", expand=True)

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", padx=8, pady=4)
        self._buttons = [
            ttk.Button(buttons, text=RU["btn_restart_bus"], command=lambda: context.emit("bus.restart")),
            ttk.Button(buttons, text=RU["btn_test_event"], command=lambda: context.emit("bus.test_event")),
        ]
        for button in self._buttons:
            button.pack(side="left", padx=4)

    def update(self) -> None:
        if ttk is None:
            return
        payload = self._context.state.get("bus", {})
        self._topics.update_rows(payload.get("topics", []))
        self._metrics.update_metrics({k: v for k, v in payload.items() if k not in {"topics", "events"}})
        self._events.update_rows(payload.get("events", []))
        bus_up = self._context.bus.is_up()
        for button in getattr(self, "_buttons", []):
            button.configure(state="normal" if bus_up else "disabled")


__all__ = ["BusScreen"]
