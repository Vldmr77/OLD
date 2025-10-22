"""Data and feature pipeline screen."""
from __future__ import annotations

from ..i18n import RU
from ..widgets.table import SimpleTable
from ..widgets.kv_meter import KeyValueMeter
from .screen_utils import create_section

try:  # pragma: no cover - tkinter optional
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    ttk = None  # type: ignore


class DataFeaturesScreen:
    def __init__(self, context) -> None:
        if ttk is None:
            return
        self._context = context
        frame = ttk.Frame(context.notebook, style="Dashboard.Section.TFrame")
        context.notebook.add(frame, text=context.strings["tab_data_features"])
        self._frame = frame

        pipeline = ttk.Frame(frame, style="Dashboard.Section.TFrame")
        pipeline.pack(fill="x", padx=8, pady=8)
        for idx, text_key in enumerate(
            [
                "data_sources",
                "data_ingest",
                "data_clean",
                "data_features",
                "data_cache",
                "data_validate",
                "data_rt",
                "data_batch",
                "data_export",
            ]
        ):
            box = ttk.Label(pipeline, text=RU[text_key], style="Dashboard.Stage.TLabel")
            box.grid(row=idx // 3, column=idx % 3, padx=4, pady=4, sticky="ew")
            pipeline.columnconfigure(idx % 3, weight=1)

        metrics_frame = create_section(frame, RU["metrics_header"])
        self._metrics = KeyValueMeter(metrics_frame)
        self._metrics.pack(anchor="w", padx=8, pady=4)

        queue_frame = create_section(frame, "Очереди")
        self._queues = SimpleTable(queue_frame, ["queue", "size"], height=6)
        self._queues.pack(fill="both", expand=True)

        buttons = ttk.Frame(frame, style="Dashboard.Section.TFrame")
        buttons.pack(fill="x", padx=8, pady=4)
        self._buttons = [
            ttk.Button(buttons, text=RU["btn_resync"], command=lambda: context.emit("features.resync")),
            ttk.Button(buttons, text=RU["btn_rebuild"], command=lambda: context.emit("features.rebuild")),
        ]
        for button in self._buttons:
            button.pack(side="left", padx=4)
        self._frame.columnconfigure(0, weight=1)

    def update(self) -> None:
        if ttk is None:
            return
        metrics = self._context.state.data_metrics()
        queues = metrics.get("queues", {}) if isinstance(metrics, dict) else {}
        queue_rows = [{"queue": name, "size": value} for name, value in queues.items()]
        self._metrics.update_metrics({k: v for k, v in metrics.items() if k != "queues"})
        self._queues.update_rows(queue_rows)
        bus_up = self._context.bus.is_up()
        for button in getattr(self, "_buttons", []):
            button.configure(state="normal" if bus_up else "disabled")


__all__ = ["DataFeaturesScreen"]
