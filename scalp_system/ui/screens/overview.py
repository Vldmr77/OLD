"""System overview screen."""
from __future__ import annotations

from typing import Dict

from ..i18n import RU
from ..state import State
from ..widgets.status_graph import StatusGraph
from ..widgets.table import SimpleTable
from ..widgets.kv_meter import KeyValueMeter
from .screen_utils import create_section

try:  # pragma: no cover - optional in headless envs
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    ttk = None  # type: ignore


MODULE_MAP = {
    "data": "Data engine",
    "features": "Feature pipeline",
    "ml": "ML engine",
    "risk": "Risk engine",
    "execution": "Execution",
    "adapter": "Tinkoff Adapter",
    "storage": "Storage",
    "heartbeat": "Heartbeat",
    "bus": "Bus",
}


class OverviewScreen:
    def __init__(self, context) -> None:
        if ttk is None:
            return
        self._context = context
        self._frame = ttk.Frame(context.notebook, style="Dashboard.Section.TFrame")
        context.notebook.add(self._frame, text=context.strings["tab_overview"])
        self._state: State = context.state
        self._graph = StatusGraph(self._frame)
        self._graph.pack(fill="x", padx=8, pady=12)

        modules_frame = create_section(self._frame, RU["module_header"])
        self._module_table = SimpleTable(modules_frame, ["module", "state", "detail"], height=6)
        self._module_table.pack(fill="both", expand=True)

        signals_frame = create_section(self._frame, RU["signals_header"])
        self._signals_table = SimpleTable(
            signals_frame,
            ["figi", "score", "timestamp"],
            height=6,
        )
        self._signals_table.pack(fill="both", expand=True)

        orders_frame = create_section(self._frame, RU["orders_header"])
        self._orders_table = SimpleTable(
            orders_frame,
            ["order_id", "figi", "status", "price"],
            height=6,
        )
        self._orders_table.pack(fill="both", expand=True)

        metrics_frame = create_section(self._frame, RU["metrics_header"])
        self._metrics = KeyValueMeter(metrics_frame)
        self._metrics.pack(anchor="w", padx=8, pady=4)

        actions = ttk.Frame(self._frame, style="Dashboard.Section.TFrame")
        actions.pack(fill="x", padx=8, pady=4)
        self._buttons: list[tuple[object, bool]] = []
        for text_key, command, requires_bus in [
            ("btn_restart", "system.restart", True),
            ("btn_pause", "system.pause", True),
            ("btn_resume", "system.resume", True),
            ("btn_reset_risk", "risk.reset_stops", True),
            ("btn_refresh", None, False),
        ]:
            btn = ttk.Button(
                actions,
                text=RU[text_key],
                command=(lambda cmd=command: self._context.emit(cmd)),
            )
            btn.pack(side="left", padx=4, pady=4)
            self._buttons.append((btn, requires_bus))

    def update(self) -> None:
        if ttk is None:
            return
        modules = self._state.modules()
        table_rows = []
        graph_payload: Dict[str, Dict[str, str]] = {}
        for module in modules:
            name = module.get("name", "")
            state = module.get("state", "unknown")
            detail = module.get("detail", "")
            table_rows.append({"module": name, "state": state, "detail": detail})
            for graph_key, label in MODULE_MAP.items():
                if label.lower() == name.lower() or graph_key.lower() == name.lower():
                    graph_payload[graph_key] = {"state": state, "detail": detail}
                    break
        self._module_table.update_rows(table_rows)
        self._graph.update_modules(graph_payload)
        self._signals_table.update_rows(self._state.signals())
        self._orders_table.update_rows(self._state.orders())
        self._metrics.update_metrics(self._state.metrics())
        bus_up = self._context.bus.is_up()
        for button, requires_bus in getattr(self, "_buttons", []):
            state = "normal" if (bus_up or not requires_bus) else "disabled"
            button.configure(state=state)


__all__ = ["OverviewScreen"]
