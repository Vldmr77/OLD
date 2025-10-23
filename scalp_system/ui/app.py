"""Tkinter dashboard application composed of modular screens."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable, Dict, Optional
from urllib.error import URLError
from urllib.request import urlopen

from .bus_client import DashboardBusClient
from .i18n import RU
from .state import State
from .theme import apply_theme

try:  # pragma: no cover - optional in headless environments
    import tkinter as tk  # type: ignore
    from tkinter import messagebox, ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    ttk = None  # type: ignore
    messagebox = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class ScreenContext:
    root: "tk.Misc"  # type: ignore[name-defined]
    notebook: "ttk.Notebook"  # type: ignore[name-defined]
    state: State
    strings: Dict[str, str]
    emit: Callable[[str | None, Optional[dict]], bool]
    notify: Callable[[str, str, str], None]
    bus: DashboardBusClient
    save_tokens: Callable[[Optional[str], Optional[str]], bool]
    token_status: Callable[[], Dict[str, bool]]
    replace_instrument: Callable[[str, str], tuple[bool, str]]
    sandbox_forward: Callable[[], tuple[bool, str]]
    backtest: Callable[[], tuple[bool, str]]
    training: Callable[[], tuple[bool, str]]
    production: Callable[[], tuple[bool, str]]


class DashboardApp:
    """Coordinator for Tkinter dashboard screens."""

    _COMMAND_ALIASES: Dict[str, str] = {
        "system.backtest.create": "system.backtest",
        "system.forwardtest.start": "system.forwardtest",
        "system.production.start": "system.production",
    }

    def __init__(
        self,
        *,
        status_endpoint: str | None = None,
        status_provider: Callable[[], dict] | None = None,
        bus_host: str = "127.0.0.1",
        bus_port: int = 8787,
        refresh_interval_ms: int = 1000,
        headless: bool = False,
        title: str | None = None,
        restart_callback: Callable[[], tuple[bool, str]] | None = None,
        token_writer: Callable[[Optional[str], Optional[str]], bool] | None = None,
        token_status_provider: Callable[[], Dict[str, bool]] | None = None,
        instrument_replace_callback: Callable[[str, str], tuple[bool, str]] | None = None,
        sandbox_forward_callback: Callable[[], tuple[bool, str]] | None = None,
        backtest_callback: Callable[[], tuple[bool, str]] | None = None,
        training_callback: Callable[[], tuple[bool, str]] | None = None,
        production_callback: Callable[[], tuple[bool, str]] | None = None,
        events_endpoint: str | None = None,
    ) -> None:
        self._state = State()
        self._status_endpoint = status_endpoint
        self._status_provider = status_provider
        self._refresh_interval_ms = max(refresh_interval_ms, 250)
        self._headless = headless or tk is None
        self._title = title or RU["app_title"]
        self._bus_client = DashboardBusClient(bus_host, bus_port, events_endpoint=events_endpoint)
        self._restart_callback = restart_callback
        self._token_writer = token_writer or (lambda s, p: False)
        self._token_status_provider = token_status_provider or (lambda: {"sandbox": False, "production": False})
        self._instrument_replace_callback = instrument_replace_callback or (lambda cur, new: (False, ""))
        self._sandbox_forward_callback = sandbox_forward_callback or (lambda: (False, ""))
        self._backtest_callback = backtest_callback or (lambda: (False, ""))
        self._training_callback = training_callback or (lambda: (False, ""))
        self._production_callback = production_callback or (lambda: (False, ""))
        self._root: tk.Tk | None = None  # type: ignore[assignment]
        self._notebook: ttk.Notebook | None = None  # type: ignore[assignment]
        self._screens: list[object] = []
        self._events_unsubscribe: Callable[[], None] | None = None
        self._status_queue: Queue[dict] = Queue()
        self._status_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._subtitle_var: tk.StringVar | None = None  # type: ignore[attr-defined]
        self._close_requested = False

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        assert tk is not None and ttk is not None
        self._root = tk.Tk()
        apply_theme(self._root)
        self._root.title(self._title)
        self._root.geometry("1280x860")
        self._root.minsize(1120, 720)
        self._root.protocol("WM_DELETE_WINDOW", self._handle_close)

        banner = ttk.Frame(self._root, style="DashboardBanner.TFrame")
        banner.pack(fill="x", padx=16, pady=(16, 8))
        title_label = ttk.Label(banner, text=self._title, style="DashboardBanner.Title.TLabel")
        title_label.pack(side="left")
        self._subtitle_var = tk.StringVar(value="")
        subtitle_label = ttk.Label(
            banner,
            textvariable=self._subtitle_var,
            style="DashboardBanner.Subtitle.TLabel",
        )
        subtitle_label.pack(side="right")

        container = ttk.Frame(self._root, style="Dashboard.Section.TFrame")
        container.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._notebook = ttk.Notebook(container, style="Dashboard.TNotebook")
        self._notebook.pack(fill="both", expand=True)

        from .screens import (
            adapter,
            bus,
            data_features,
            execution,
            logs_metrics,
            ml,
            overview,
            risk,
        )

        context = ScreenContext(
            root=self._root,
            notebook=self._notebook,
            state=self._state,
            strings=RU,
            emit=self._emit_command,
            notify=self._notify,
            bus=self._bus_client,
            save_tokens=self._save_tokens,
            token_status=self._token_status_provider,
            replace_instrument=self._replace_instrument,
            sandbox_forward=self._sandbox_forward,
            backtest=self._backtest,
            training=self._training,
            production=self._production,
        )

        self._screens = [
            overview.OverviewScreen(context),
            data_features.DataFeaturesScreen(context),
            ml.MLScreen(context),
            risk.RiskScreen(context),
            execution.ExecutionScreen(context),
            adapter.AdapterScreen(context),
            bus.BusScreen(context),
            logs_metrics.LogsMetricsScreen(context),
        ]

    # ------------------------------------------------------------------
    def _configure_bindings(self) -> None:
        assert self._root is not None
        self._root.bind("<F5>", lambda *_: self._refresh())
        self._root.bind("r", lambda *_: self._emit_command("system.restart"))
        self._root.bind("p", lambda *_: self._emit_command("system.pause"))
        self._root.bind("b", lambda *_: self._emit_command("system.backtest.create"))
        self._root.bind("t", lambda *_: self._emit_command("ml.train"))

    # ------------------------------------------------------------------
    def run(self) -> None:
        if self._headless:
            payload = self._fetch_status_payload()
            if payload:
                self._state.update_from_status(payload)
            return
        if self._root is None:
            self._build_ui()
            self._configure_bindings()
        assert self._root is not None
        self._stop_event.clear()
        self._status_thread = threading.Thread(target=self._status_worker, daemon=True)
        self._status_thread.start()
        self._pump_ui()
        assert self._root is not None
        self._events_unsubscribe = self._bus_client.subscribe(self._handle_event)
        try:
            self._root.mainloop()
        finally:
            if self._events_unsubscribe:
                self._events_unsubscribe()
            self._stop_event.set()
            if self._status_thread:
                self._status_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    def _handle_close(self) -> None:
        if self._headless or self._root is None:
            return
        if self._close_requested:
            try:
                self._root.destroy()
            except Exception:  # pragma: no cover - defensive
                pass
            return
        self._close_requested = True
        self._stop_event.set()
        try:
            self._bus_client.emit("system.shutdown")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to request shutdown during close: %s", exc)
        destroy = getattr(self._root, "destroy", None)
        after = getattr(self._root, "after", None)
        if callable(after):
            after(100, lambda: destroy() if callable(destroy) else None)
        elif callable(destroy):
            destroy()

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        payload = self._fetch_status_payload()
        if payload:
            self._state.update_from_status(payload)
            self._update_screens()

    # ------------------------------------------------------------------
    def _fetch_status_payload(self) -> dict | None:
        if self._status_endpoint:
            try:
                with urlopen(self._status_endpoint, timeout=2) as response:  # nosec B310
                    return json.load(response)
            except URLError as exc:
                LOGGER.debug("Status endpoint unavailable: %s", exc)
                if not self._status_provider:
                    return {"error": {"source": "endpoint", "message": str(exc)}}
        if self._status_provider:
            try:
                return self._status_provider()
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Status provider failed: %s", exc)
                return {"error": {"source": "provider", "message": str(exc)}}
        return None

    # ------------------------------------------------------------------
    def _status_worker(self) -> None:
        interval = max(self._refresh_interval_ms / 1000.0, 0.25)
        while not self._stop_event.is_set():
            payload = self._fetch_status_payload()
            if payload is not None:
                self._status_queue.put(payload)
            if self._stop_event.wait(interval):
                break

    # ------------------------------------------------------------------
    def _pump_ui(self) -> None:
        if self._headless or self._root is None:
            return
        try:
            while True:
                payload = self._status_queue.get_nowait()
                if payload.get("error"):
                    message = payload["error"].get("message", "")
                    LOGGER.error("Status provider error: %s", message)
                    self._notify("error", RU["evt_err_title"], message or RU["status_error"])
                    continue
                self._state.update_from_status(payload)
                self._update_screens()
        except Empty:
            pass
        finally:
            self._root.after(200, self._pump_ui)

    # ------------------------------------------------------------------
    def _update_screens(self) -> None:
        if self._headless:
            return
        for screen in self._screens:
            update = getattr(screen, "update", None)
            if callable(update):
                update()
        self._update_banner()

    # ------------------------------------------------------------------
    def _emit_command(self, name: str | None, args: Optional[dict] = None) -> bool:
        if name is None:
            self._refresh()
            return True
        canonical = self._COMMAND_ALIASES.get(name, name)
        try:
            if self._bus_client.is_up():
                self._bus_client.emit(canonical, args)
                self._notify("info", RU["evt_ok_title"], RU["evt_ok_msg"].format(name=canonical))
                return True
            LOGGER.debug("Bus unavailable; falling back to callback for %s", name)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to emit command via bus: %s", exc)
            self._notify("error", RU["evt_err_title"], str(exc))
            return False

        fallback_map = {
            "system.restart": self._restart_callback,
            "system.backtest": self._backtest_callback,
            "system.forwardtest": self._sandbox_forward_callback,
            "system.production": self._production_callback,
            "ml.train": self._training_callback,
        }
        callback = fallback_map.get(canonical)
        if callback is None:
            self._notify("warning", RU["bus_down"], RU["bus_down_msg"])
            return False
        ok, message = callback()
        level = "info" if ok else "error"
        self._notify(level, RU["evt_ok_title"], message or RU["status_updated"])
        return ok

    # ------------------------------------------------------------------
    def _save_tokens(self, sandbox: Optional[str], production: Optional[str]) -> bool:
        try:
            result = self._token_writer(sandbox, production)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to persist tokens: %s", exc)
            self._notify("error", RU["evt_err_title"], str(exc))
            return False
        if result:
            self._notify("info", RU["evt_ok_title"], RU["token_saved"])
        else:
            self._notify("warning", RU["evt_warn_title"], RU["token_error"])
        return result

    # ------------------------------------------------------------------
    def _replace_instrument(self, current: str, new: str) -> tuple[bool, str]:
        try:
            return self._instrument_replace_callback(current, new)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Instrument replacement failed: %s", exc)
            return False, str(exc)

    # ------------------------------------------------------------------
    def _sandbox_forward(self) -> tuple[bool, str]:
        try:
            return self._sandbox_forward_callback()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Sandbox forward failed: %s", exc)
            return False, str(exc)

    # ------------------------------------------------------------------
    def _production(self) -> tuple[bool, str]:
        try:
            return self._production_callback()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Production switch failed: %s", exc)
            return False, str(exc)

    # ------------------------------------------------------------------
    def _backtest(self) -> tuple[bool, str]:
        try:
            return self._backtest_callback()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Backtest trigger failed: %s", exc)
            return False, str(exc)

    # ------------------------------------------------------------------
    def _training(self) -> tuple[bool, str]:
        try:
            return self._training_callback()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Training command failed: %s", exc)
            return False, str(exc)

    # ------------------------------------------------------------------
    def _handle_event(self, payload: dict) -> None:
        LOGGER.debug("Dashboard received event: %s", payload)
        if payload.get("type") == "status":
            body = payload.get("payload")
            if isinstance(body, dict):
                self._state.update_from_status(body)
                self._update_screens()

    # ------------------------------------------------------------------
    def _notify(self, level: str, title: str, message: str) -> None:
        if self._headless or messagebox is None:
            LOGGER.log(getattr(logging, level.upper(), logging.INFO), "%s: %s", title, message)
            return
        if level == "error":
            messagebox.showerror(title, message)
        elif level == "warning":
            messagebox.showwarning(title, message)
        else:
            messagebox.showinfo(title, message)

    # ------------------------------------------------------------------
    def _update_banner(self) -> None:
        if self._headless or self._subtitle_var is None:
            return
        build = self._state.get("build", {}) or {}
        mode = str(build.get("mode", "offline")).upper()
        uptime = build.get("uptime_s")
        uptime_text = self._format_uptime(uptime)
        bus_state_raw = str(self._state.get("bus.state", "unknown")).lower()
        if bus_state_raw == "up":
            bus_display = RU["banner_bus_up"]
        elif bus_state_raw == "down":
            bus_display = RU["banner_bus_down"]
        else:
            bus_display = RU["banner_bus_unknown"]
        parts = [f"{RU['status_mode']}: {mode}", f"{RU['banner_bus']}: {bus_display}"]
        if uptime_text:
            parts.append(f"{RU['banner_uptime']}: {uptime_text}")
        alerts = self._state.alerts()
        if alerts:
            parts.append(f"{RU['banner_alerts']}: {len(alerts)}")
        self._subtitle_var.set("  •  ".join(parts))

    @staticmethod
    def _format_uptime(value: object) -> str:
        if isinstance(value, (int, float)) and value >= 0:
            total = int(value)
            hours, rem = divmod(total, 3600)
            minutes, seconds = divmod(rem, 60)
            if hours:
                return f"{hours}ч {minutes:02d}м"
            if minutes:
                return f"{minutes}м {seconds:02d}с"
            return f"{seconds}с"
        return ""


def run_dashboard(
    repository_path,
    *,
    status_provider: Callable[[], dict] | None = None,
    refresh_interval_ms: int = 1000,
    signal_limit: int = 25,
    headless: bool = False,
    title: str = "Scalp System Dashboard",
    background: bool = False,
    config_path=None,
    restart_callback: Callable[[], tuple[bool, str]] | None = None,
    token_status_provider: Callable[[], Dict[str, bool]] | None = None,
    token_writer: Callable[[Optional[str], Optional[str]], bool] | None = None,
    instrument_replace_callback: Callable[[str, str], tuple[bool, str]] | None = None,
    sandbox_forward_callback: Callable[[], tuple[bool, str]] | None = None,
    backtest_callback: Callable[[], tuple[bool, str]] | None = None,
    training_callback: Callable[[], tuple[bool, str]] | None = None,
    production_callback: Callable[[], tuple[bool, str]] | None = None,
    bus_address: tuple[str, int] | None = None,
    events_endpoint: str | None = None,
    status_endpoint: str | None = None,
):
    """Entry point compatible with orchestrator expectations."""

    host, port = bus_address if bus_address else ("127.0.0.1", 8787)

    app = DashboardApp(
        status_endpoint=status_endpoint,
        status_provider=status_provider,
        bus_host=host,
        bus_port=port,
        refresh_interval_ms=refresh_interval_ms,
        headless=headless,
        title=title,
        restart_callback=restart_callback,
        token_writer=token_writer,
        token_status_provider=token_status_provider,
        instrument_replace_callback=instrument_replace_callback,
        sandbox_forward_callback=sandbox_forward_callback,
        backtest_callback=backtest_callback,
        training_callback=training_callback,
        production_callback=production_callback,
        events_endpoint=events_endpoint,
    )

    if headless or not background:
        app.run()
        return None

    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    return thread


__all__ = ["DashboardApp", "run_dashboard", "ScreenContext"]
