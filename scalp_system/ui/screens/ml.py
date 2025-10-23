"""Machine-learning ensemble screen."""
from __future__ import annotations

from ..i18n import RU
from ..widgets.table import SimpleTable
from ..widgets.kv_meter import KeyValueMeter
from .screen_utils import create_section

try:  # pragma: no cover
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    ttk = None  # type: ignore


class MLScreen:
    def __init__(self, context) -> None:
        if ttk is None:
            return
        self._context = context
        frame = ttk.Frame(context.notebook, style="Dashboard.Section.TFrame")
        context.notebook.add(frame, text=context.strings["tab_ml"])
        self._frame = frame

        models_frame = create_section(frame, RU["ml_models"])
        self._model_table = SimpleTable(
            models_frame,
            ["name", "scope", "weight", "roc", "f1", "updated"],
            height=6,
        )
        self._model_table.pack(fill="both", expand=True)

        jobs_frame = create_section(frame, RU["ml_jobs"])
        self._jobs_table = SimpleTable(jobs_frame, ["id", "strategy", "progress", "eta", "status"], height=5)
        self._jobs_table.pack(fill="both", expand=True)

        metrics_frame = create_section(frame, RU["ml_metrics"])
        self._metrics = KeyValueMeter(metrics_frame)
        self._metrics.pack(anchor="w", padx=8, pady=4)

        artifacts_frame = create_section(frame, RU["ml_artifacts"])
        self._artifacts_table = SimpleTable(artifacts_frame, ["version", "created", "note"], height=4)
        self._artifacts_table.pack(fill="both", expand=True)

        buttons = ttk.Frame(frame, style="Dashboard.Section.TFrame")
        buttons.pack(fill="x", padx=8, pady=4)
        self._buttons = [
            ttk.Button(buttons, text=RU["btn_train"], command=lambda: context.emit("ml.train")),
            ttk.Button(buttons, text=RU["btn_validate"], command=lambda: context.emit("ml.validate")),
            ttk.Button(buttons, text=RU["btn_rollback"], command=lambda: context.emit("ml.rollback")),
        ]
        for button in self._buttons:
            button.pack(side="left", padx=4)

    def update(self) -> None:
        if ttk is None:
            return
        payload = self._context.state.get("ml", {})
        self._model_table.update_rows(payload.get("models", []))
        self._jobs_table.update_rows(payload.get("jobs", []))
        self._metrics.update_metrics(payload.get("metrics", {}))
        self._artifacts_table.update_rows(payload.get("artifacts", []))
        bus_up = self._context.bus.is_up()
        for button in getattr(self, "_buttons", []):
            button.configure(state="normal" if bus_up else "disabled")


__all__ = ["MLScreen"]
