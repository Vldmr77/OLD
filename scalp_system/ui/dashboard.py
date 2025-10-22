"""Tkinter-based dashboard visualising scalping system telemetry."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..storage.repository import SQLiteRepository

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - tkinter availability depends on environment
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - headless environments
    tk = None  # type: ignore
    ttk = None  # type: ignore


@dataclass
class ModuleStatus:
    """Represents the current state of a subsystem."""

    name: str
    state: str
    detail: str = ""
    severity: str = "normal"  # one of {normal, warning, error}


@dataclass
class DashboardStatus:
    """Aggregated runtime status for the dashboard."""

    modules: list[ModuleStatus] = field(default_factory=list)
    processes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ensemble: dict[str, object] = field(default_factory=dict)


@dataclass
class DashboardSnapshot:
    """Rendered snapshot of repository data and runtime status."""

    summary: dict
    signals: list[dict]
    status: DashboardStatus


class DashboardUI:
    """Interactive Tkinter dashboard that refreshes from the repository."""

    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        status_provider: Optional[Callable[[], DashboardStatus]] = None,
        refresh_interval_ms: int = 1000,
        signal_limit: int = 25,
        headless: bool = False,
        title: str = "Scalp System Dashboard",
    ) -> None:
        self._repository = repository
        self._status_provider = status_provider
        self._refresh_interval_ms = max(int(refresh_interval_ms), 250)
        self._signal_limit = max(int(signal_limit), 1)
        self._title = title
        self._latest_snapshot: Optional[DashboardSnapshot] = None
        self._headless = headless or tk is None
        self._closed = False
        self._status_error: Optional[str] = None
        self._root = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start the Tk event loop (or execute one refresh when headless)."""

        if self._headless:
            self.refresh_once()
            return

        self._ensure_root()
        if self._headless:
            # ``_ensure_root`` may switch to headless if Tk initialisation fails.
            self.refresh_once()
            return

        self.refresh_once()
        self._schedule_refresh()
        assert self._root is not None
        try:
            self._root.mainloop()
        finally:
            self._closed = True

    def refresh_once(self) -> DashboardSnapshot:
        """Fetch the latest repository data and update widgets."""

        summary = self._repository.summary()
        signals = self._repository.fetch_signals(limit=self._signal_limit)
        status = self._collect_status()
        snapshot = DashboardSnapshot(summary=summary, signals=signals, status=status)
        self._latest_snapshot = snapshot
        if not self._headless:
            self._update_widgets(snapshot)
        return snapshot

    @property
    def latest_snapshot(self) -> Optional[DashboardSnapshot]:
        return self._latest_snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_status(self) -> DashboardStatus:
        try:
            if self._status_provider is None:
                return DashboardStatus()
            payload = self._status_provider()
            if payload is None:
                return DashboardStatus()
            if not isinstance(payload, DashboardStatus):
                raise TypeError(
                    "status_provider must return DashboardStatus, got "
                    f"{type(payload)!r}"
                )
            return payload
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Dashboard status provider failed: %s", exc)
            status = DashboardStatus()
            status.errors.append(f"Status provider error: {exc}")
            self._status_error = str(exc)
            return status

    def _schedule_refresh(self) -> None:
        if self._headless or self._closed:
            return
        assert self._root is not None
        self._root.after(self._refresh_interval_ms, self._tick)

    def _tick(self) -> None:
        if self._closed:
            return
        self.refresh_once()
        self._schedule_refresh()

    def _build_layout(self) -> None:
        assert tk is not None and ttk is not None
        style = ttk.Style()  # type: ignore[call-arg]
        try:
            style.theme_use("clam")
        except Exception:  # pragma: no cover - fall back to default theme
            pass
        style.configure("Dashboard.TFrame", background="#0f172a")
        style.configure(
            "Dashboard.TLabel",
            background="#0f172a",
            foreground="#e2e8f0",
            font=("Segoe UI", 11),
        )
        style.configure(
            "Dashboard.Title.TLabel",
            background="#0f172a",
            foreground="#38bdf8",
            font=("Segoe UI", 20, "bold"),
        )
        style.configure(
            "Dashboard.Badge.TLabel",
            background="#1e293b",
            foreground="#22c55e",
            font=("Segoe UI", 12, "bold"),
            padding=(8, 4),
        )
        style.configure(
            "Dashboard.Treeview",
            background="#1e293b",
            fieldbackground="#1e293b",
            foreground="#e2e8f0",
            rowheight=24,
        )
        style.map(
            "Dashboard.Treeview",
            background=[("selected", "#2563eb")],
            foreground=[("selected", "#f8fafc")],
        )
        style.configure("Dashboard.Treeview.Heading", background="#1d4ed8", foreground="#f8fafc")

        root = self._root
        assert root is not None

        container = ttk.Frame(root, style="Dashboard.TFrame")  # type: ignore[arg-type]
        container.pack(fill="both", expand=True, padx=16, pady=16)

        header = ttk.Frame(container, style="Dashboard.TFrame")  # type: ignore[arg-type]
        header.pack(fill="x")
        self._title_label = ttk.Label(
            header,
            text=self._title,
            style="Dashboard.Title.TLabel",
        )
        self._title_label.pack(anchor="w")
        self._summary_label = ttk.Label(
            header,
            text="",
            style="Dashboard.TLabel",
        )
        self._summary_label.pack(anchor="w", pady=(4, 12))

        top_split = ttk.Frame(container, style="Dashboard.TFrame")  # type: ignore[arg-type]
        top_split.pack(fill="x")

        modules_frame = ttk.Frame(top_split, style="Dashboard.TFrame")  # type: ignore[arg-type]
        modules_frame.pack(side="left", fill="both", expand=True)
        modules_label = ttk.Label(
            modules_frame,
            text="Subsystem status",
            style="Dashboard.TLabel",
        )
        modules_label.pack(anchor="w")
        columns = ("module", "state", "detail")
        self._module_tree = ttk.Treeview(
            modules_frame,
            columns=columns,
            show="headings",
            style="Dashboard.Treeview",
        )
        self._module_tree.heading("module", text="Module")
        self._module_tree.heading("state", text="State")
        self._module_tree.heading("detail", text="Detail")
        self._module_tree.column("module", width=180, anchor="w")
        self._module_tree.column("state", width=140, anchor="center")
        self._module_tree.column("detail", width=320, anchor="w")
        self._module_tree.pack(fill="both", expand=True, pady=(4, 0))
        self._module_tree.tag_configure("warning", foreground="#facc15")
        self._module_tree.tag_configure("error", foreground="#f87171")

        right_frame = ttk.Frame(top_split, style="Dashboard.TFrame")  # type: ignore[arg-type]
        right_frame.pack(side="left", fill="both", expand=True, padx=(16, 0))

        processes_label = ttk.Label(
            right_frame,
            text="Background processes",
            style="Dashboard.TLabel",
        )
        processes_label.pack(anchor="w")
        self._process_list = tk.Listbox(
            right_frame,
            height=6,
            bg="#1e293b",
            fg="#e2e8f0",
            highlightthickness=0,
            selectbackground="#2563eb",
            selectforeground="#f8fafc",
            activestyle="none",
        )
        self._process_list.pack(fill="both", expand=True, pady=(4, 12))

        errors_label = ttk.Label(
            right_frame,
            text="Active alerts",
            style="Dashboard.TLabel",
        )
        errors_label.pack(anchor="w")
        self._error_text = tk.Text(
            right_frame,
            height=6,
            bg="#1e293b",
            fg="#f87171",
            highlightthickness=0,
            relief="flat",
            wrap="word",
        )
        self._error_text.pack(fill="both", expand=True, pady=(4, 0))
        self._error_text.configure(state="disabled")

        bottom_frame = ttk.Frame(container, style="Dashboard.TFrame")  # type: ignore[arg-type]
        bottom_frame.pack(fill="both", expand=True, pady=(16, 0))

        signals_frame = ttk.Frame(bottom_frame, style="Dashboard.TFrame")  # type: ignore[arg-type]
        signals_frame.pack(side="left", fill="both", expand=True)
        signals_label = ttk.Label(
            signals_frame,
            text="Recent signals",
            style="Dashboard.TLabel",
        )
        signals_label.pack(anchor="w")
        self._signals_tree = ttk.Treeview(
            signals_frame,
            columns=("figi", "direction", "confidence", "timestamp"),
            show="headings",
            style="Dashboard.Treeview",
        )
        for column, width in (
            ("figi", 160),
            ("direction", 100),
            ("confidence", 120),
            ("timestamp", 200),
        ):
            self._signals_tree.heading(column, text=column.title())
            self._signals_tree.column(column, width=width, anchor="center" if column != "timestamp" else "w")
        self._signals_tree.pack(fill="both", expand=True, pady=(4, 0))

        ensemble_frame = ttk.Frame(bottom_frame, style="Dashboard.TFrame")  # type: ignore[arg-type]
        ensemble_frame.pack(side="left", fill="both", expand=True, padx=(16, 0))
        ensemble_label = ttk.Label(
            ensemble_frame,
            text="ML ensemble overview",
            style="Dashboard.TLabel",
        )
        ensemble_label.pack(anchor="w")
        self._ensemble_tree = ttk.Treeview(
            ensemble_frame,
            columns=("scope", "model", "weight"),
            show="headings",
            style="Dashboard.Treeview",
        )
        self._ensemble_tree.heading("scope", text="Scope")
        self._ensemble_tree.heading("model", text="Model")
        self._ensemble_tree.heading("weight", text="Weight")
        self._ensemble_tree.column("scope", width=120, anchor="center")
        self._ensemble_tree.column("model", width=160, anchor="w")
        self._ensemble_tree.column("weight", width=120, anchor="center")
        self._ensemble_tree.pack(fill="both", expand=True, pady=(4, 0))

    def _update_widgets(self, snapshot: DashboardSnapshot) -> None:
        assert tk is not None and ttk is not None and self._root is not None
        total = snapshot.summary.get("total_signals", 0)
        latest = snapshot.summary.get("latest")
        latest_line = "No signals captured yet."
        if latest:
            latest_line = (
                f"Latest {latest.get('figi')} • direction {latest.get('direction')} "
                f"• confidence {latest.get('confidence', 0):.2f} at "
                f"{self._format_timestamp(latest.get('timestamp'))}"
            )
        self._summary_label.config(
            text=f"Signals stored: {total} — {latest_line}"
        )
        self._update_module_tree(snapshot.status.modules)
        self._update_process_list(snapshot.status.processes)
        self._update_errors(snapshot.status.errors)
        self._update_signals(snapshot.signals)
        self._update_ensemble(snapshot.status.ensemble)

    def _update_module_tree(self, modules: Iterable[ModuleStatus]) -> None:
        self._module_tree.delete(*self._module_tree.get_children())
        for module in modules:
            tags = ()
            if module.severity in {"warning", "error"}:
                tags = (module.severity,)
            self._module_tree.insert(
                "",
                "end",
                values=(module.name, f"{module.state}", module.detail),
                tags=tags,
            )

    def _update_process_list(self, processes: Iterable[str]) -> None:
        self._process_list.delete(0, tk.END)
        for process in processes:
            self._process_list.insert(tk.END, process)

    def _update_errors(self, errors: Iterable[str]) -> None:
        self._error_text.configure(state="normal")
        self._error_text.delete("1.0", tk.END)
        for error in errors:
            self._error_text.insert(tk.END, f"• {error}\n")
        if not errors:
            self._error_text.insert(tk.END, "No active alerts.")
        self._error_text.configure(state="disabled")

    def _update_signals(self, signals: Iterable[dict]) -> None:
        self._signals_tree.delete(*self._signals_tree.get_children())
        for signal in signals:
            timestamp = self._format_timestamp(signal.get("timestamp"))
            direction = signal.get("direction")
            if direction == 1:
                direction = "BUY"
            elif direction == -1:
                direction = "SELL"
            self._signals_tree.insert(
                "",
                "end",
                values=(
                    signal.get("figi"),
                    direction,
                    f"{float(signal.get('confidence', 0.0)):.2f}",
                    timestamp,
                ),
            )

    def _update_ensemble(self, ensemble: dict[str, object]) -> None:
        self._ensemble_tree.delete(*self._ensemble_tree.get_children())
        if not ensemble:
            return
        base = ensemble.get("base_weights", {})
        for model, weight in base.items():
            self._ensemble_tree.insert(
                "",
                "end",
                values=("base", model, f"{float(weight):.2f}"),
            )
        classes = ensemble.get("class_weights", {})
        for label, weights in classes.items():
            for model, weight in weights.items():
                self._ensemble_tree.insert(
                    "",
                    "end",
                    values=(label, model, f"{float(weight):.2f}"),
                )
        device = ensemble.get("device_mode")
        throttle = ensemble.get("throttle_delay")
        model_dir = ensemble.get("model_dir")
        footer = []
        if device:
            footer.append(f"device={device}")
        if throttle is not None:
            footer.append(f"throttle={float(throttle):.3f}s")
        if model_dir:
            footer.append(f"models={model_dir}")
        if footer:
            self._ensemble_tree.insert(
                "",
                "end",
                values=("info", " ".join(footer), ""),
            )

    def _format_timestamp(self, value) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:  # pragma: no cover - safety guard
                return str(value)
        return str(value)

    def _on_close(self) -> None:
        self._closed = True
        if self._root is not None:
            self._root.quit()

    def _ensure_root(self) -> None:
        """Initialise the Tk root in the current thread if required."""

        if self._headless or self._root is not None:
            return
        assert tk is not None and ttk is not None  # for type-checkers
        try:
            root = tk.Tk()  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - headless fallback
            LOGGER.warning("Falling back to headless dashboard: %s", exc)
            self._headless = True
            return
        self._root = root
        self._root.title(self._title)
        self._root.configure(bg="#0f172a")
        self._build_layout()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)


def run_dashboard(
    repository_path: Path,
    *,
    status_provider: Optional[Callable[[], DashboardStatus]] = None,
    refresh_interval_ms: int = 1000,
    signal_limit: int = 25,
    background: bool = False,
    headless: bool = False,
    title: str = "Scalp System Dashboard",
) -> Optional[threading.Thread]:
    """Launch the Tkinter dashboard.

    When ``background`` is True a daemon thread running the Tk loop is
    returned. Otherwise this function blocks until the window is closed.
    """

    def _run_ui() -> None:
        repository = SQLiteRepository(repository_path)
        ui = DashboardUI(
            repository,
            status_provider=status_provider,
            refresh_interval_ms=refresh_interval_ms,
            signal_limit=signal_limit,
            headless=headless,
            title=title,
        )
        ui.run()

    if background:
        thread = threading.Thread(target=_run_ui, daemon=True)
        thread.start()
        return thread

    _run_ui()
    return None


__all__ = [
    "DashboardUI",
    "DashboardStatus",
    "DashboardSnapshot",
    "ModuleStatus",
    "run_dashboard",
]
