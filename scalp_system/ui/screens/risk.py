"""Risk management screen."""
from __future__ import annotations

from ..i18n import RU
from ..widgets.table import SimpleTable
from ..widgets.kv_meter import KeyValueMeter
from .screen_utils import create_section

try:  # pragma: no cover
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    ttk = None  # type: ignore


class RiskScreen:
    def __init__(self, context) -> None:
        if ttk is None:
            return
        self._context = context
        frame = ttk.Frame(context.notebook)
        context.notebook.add(frame, text=context.strings["tab_risk"])

        profiles = create_section(frame, RU["risk_profiles"])
        self._profiles = KeyValueMeter(profiles)
        self._profiles.pack(anchor="w", padx=8, pady=4)

        exposure = create_section(frame, RU["risk_exposure"])
        self._exposure = SimpleTable(exposure, ["figi", "qty", "pnl", "stop"], height=6)
        self._exposure.pack(fill="both", expand=True)

        params = create_section(frame, RU["risk_params"])
        self._params = KeyValueMeter(params)
        self._params.pack(anchor="w", padx=8, pady=4)

        alerts = create_section(frame, RU["risk_alerts"])
        self._alerts = SimpleTable(alerts, ["id", "severity", "since"], height=4)
        self._alerts.pack(fill="both", expand=True)

        log_frame = create_section(frame, RU["risk_log"])
        self._log = SimpleTable(log_frame, ["timestamp", "decision"], height=4)
        self._log.pack(fill="both", expand=True)

        self._reset_button = ttk.Button(
            frame, text=RU["btn_reset_risk"], command=lambda: context.emit("risk.reset_stops")
        )
        self._reset_button.pack(anchor="w", padx=8, pady=6)

    def update(self) -> None:
        if ttk is None:
            return
        payload = self._context.state.get("risk", {})
        self._profiles.update_metrics(payload.get("profile", {}))
        self._exposure.update_rows(payload.get("exposure", []))
        self._params.update_metrics(payload.get("params", {}))
        alerts = payload.get("alerts", [])
        if not alerts:
            alerts = [{"id": RU["alert_none"], "severity": "info", "since": ""}]
        self._alerts.update_rows(alerts)
        self._log.update_rows(payload.get("log", []))
        if hasattr(self, "_reset_button"):
            self._reset_button.configure(state="normal" if self._context.bus.is_up() else "disabled")


__all__ = ["RiskScreen"]
