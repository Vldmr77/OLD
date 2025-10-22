"""Tkinter-based dashboard visualising scalping system telemetry."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..config.token_prompt import store_tokens, token_status
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
    status: str = "running"  # one of {running, inactive, error}


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
        config_path: Optional[Path] = None,
        restart_callback: Optional[Callable[[], None]] = None,
        token_status_provider: Optional[Callable[[], dict[str, bool]]] = None,
        token_writer: Optional[Callable[[Optional[str], Optional[str]], bool]] = None,
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
        self._config_path = Path(config_path).expanduser() if config_path else None
        if token_status_provider is None and self._config_path is not None:
            token_status_provider = lambda: token_status(self._config_path)
        self._token_status_provider = token_status_provider
        if token_writer is None and self._config_path is not None:
            token_writer = lambda sandbox, production: store_tokens(
                self._config_path, sandbox=sandbox, production=production
            )
        self._token_writer = token_writer
        self._restart_callback = restart_callback
        self._module_badge_frame = None
        self._sandbox_entry = None
        self._production_entry = None
        self._sandbox_status_label = None
        self._production_status_label = None
        self._token_feedback_label = None
        self._token_status: dict[str, bool] = {"sandbox": False, "production": False}

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
            "Dashboard.ModuleBadgeRunning.TLabel",
            background="#14532d",
            foreground="#bbf7d0",
            font=("Segoe UI", 11, "bold"),
            padding=(10, 6),
        )
        style.configure(
            "Dashboard.ModuleBadgeInactive.TLabel",
            background="#334155",
            foreground="#cbd5f5",
            font=("Segoe UI", 11, "bold"),
            padding=(10, 6),
        )
        style.configure(
            "Dashboard.ModuleBadgeError.TLabel",
            background="#7f1d1d",
            foreground="#fecaca",
            font=("Segoe UI", 11, "bold"),
            padding=(10, 6),
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

        self._module_badge_frame = ttk.Frame(
            container, style="Dashboard.TFrame"
        )  # type: ignore[arg-type]
        self._module_badge_frame.pack(fill="x", pady=(0, 12))

        token_frame = ttk.Frame(
            container, style="Dashboard.TFrame"
        )  # type: ignore[arg-type]
        token_frame.pack(fill="x", pady=(0, 16))
        token_frame.columnconfigure(1, weight=1)
        tokens_label = ttk.Label(
            token_frame,
            text="Tinkoff API tokens",
            style="Dashboard.TLabel",
        )
        tokens_label.grid(row=0, column=0, sticky="w")

        ttk.Label(
            token_frame,
            text="Sandbox",
            style="Dashboard.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self._sandbox_entry = ttk.Entry(token_frame, show="•")
        self._sandbox_entry.grid(row=1, column=1, sticky="we", padx=(8, 8), pady=(6, 0))
        self._bind_copy_paste(self._sandbox_entry)
        self._sandbox_status_label = ttk.Label(
            token_frame,
            text="Sandbox: unknown",
            style="Dashboard.TLabel",
        )
        self._sandbox_status_label.grid(row=1, column=2, sticky="w", pady=(6, 0))

        ttk.Label(
            token_frame,
            text="Production",
            style="Dashboard.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))
        self._production_entry = ttk.Entry(token_frame, show="•")
        self._production_entry.grid(
            row=2, column=1, sticky="we", padx=(8, 8), pady=(6, 0)
        )
        self._bind_copy_paste(self._production_entry)
        self._production_status_label = ttk.Label(
            token_frame,
            text="Production: unknown",
            style="Dashboard.TLabel",
        )
        self._production_status_label.grid(row=2, column=2, sticky="w", pady=(6, 0))

        button_frame = ttk.Frame(
            token_frame, style="Dashboard.TFrame"
        )  # type: ignore[arg-type]
        button_frame.grid(row=0, column=3, rowspan=3, sticky="e", padx=(16, 0))

        apply_button = ttk.Button(
            button_frame,
            text="Save tokens",
            command=self._apply_tokens,
        )
        apply_button.pack(anchor="e", pady=(6, 4))

        restart_button = ttk.Button(
            button_frame,
            text="Restart system",
            command=self._restart_system,
        )
        if self._restart_callback is None:
            restart_button.state(["disabled"])
        restart_button.pack(anchor="e")

        self._token_feedback_label = ttk.Label(
            token_frame,
            text="",
            style="Dashboard.TLabel",
        )
        self._token_feedback_label.grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(6, 0)
        )

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
        self._module_tree.tag_configure("running", foreground="#22c55e")
        self._module_tree.tag_configure("inactive", foreground="#94a3b8")
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
        self._update_module_badges(snapshot.status.modules)
        self._update_module_tree(snapshot.status.modules)
        self._update_process_list(snapshot.status.processes)
        self._update_errors(snapshot.status.errors)
        self._update_signals(snapshot.signals)
        self._update_ensemble(snapshot.status.ensemble)
        self._refresh_token_status()

    def _update_module_tree(self, modules: Iterable[ModuleStatus]) -> None:
        self._module_tree.delete(*self._module_tree.get_children())
        for module in modules:
            if module.severity == "error" or module.status == "error":
                tags = ("error",)
            elif module.severity == "warning":
                tags = ("warning",)
            elif module.status == "inactive":
                tags = ("inactive",)
            else:
                tags = ("running",)
            self._module_tree.insert(
                "",
                "end",
                values=(module.name, f"{module.state}", module.detail),
                tags=tags,
            )

    def _update_module_badges(self, modules: Iterable[ModuleStatus]) -> None:
        if self._module_badge_frame is None:
            return
        for child in self._module_badge_frame.winfo_children():
            child.destroy()
        for module in modules:
            style_name = self._badge_style(module)
            label = ttk.Label(
                self._module_badge_frame,
                text=module.name,
                style=style_name,
            )
            label.pack(side="left", padx=(0, 8), pady=(0, 4))

    def _badge_style(self, module: ModuleStatus) -> str:
        if module.severity == "error" or module.status == "error":
            return "Dashboard.ModuleBadgeError.TLabel"
        if module.status == "inactive":
            return "Dashboard.ModuleBadgeInactive.TLabel"
        return "Dashboard.ModuleBadgeRunning.TLabel"

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

    def _refresh_token_status(self) -> None:
        if self._sandbox_status_label is None or self._production_status_label is None:
            return
        if self._token_status_provider is None:
            self._sandbox_status_label.config(
                text="Sandbox: unavailable", foreground="#facc15"
            )
            self._production_status_label.config(
                text="Production: unavailable", foreground="#facc15"
            )
            return
        try:
            status = self._token_status_provider() or {}
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Failed to read token status: %s", exc)
            self._set_feedback(f"Token status error: {exc}", "#f87171")
            self._sandbox_status_label.config(
                text="Sandbox: error", foreground="#f87171"
            )
            self._production_status_label.config(
                text="Production: error", foreground="#f87171"
            )
            return
        sandbox = bool(status.get("sandbox"))
        production = bool(status.get("production"))
        self._token_status = {"sandbox": sandbox, "production": production}
        self._sandbox_status_label.config(
            text=("Sandbox: configured" if sandbox else "Sandbox: missing"),
            foreground="#22c55e" if sandbox else "#f87171",
        )
        self._production_status_label.config(
            text=("Production: configured" if production else "Production: missing"),
            foreground="#22c55e" if production else "#f87171",
        )

    def _apply_tokens(self) -> None:
        if self._token_writer is None:
            self._set_feedback("Token updates are unavailable without a config path.", "#facc15")
            return
        if self._sandbox_entry is None or self._production_entry is None:
            return
        sandbox_raw = self._sandbox_entry.get().strip()
        production_raw = self._production_entry.get().strip()
        if not sandbox_raw and not production_raw:
            self._set_feedback("Enter at least one token to save.", "#facc15")
            return
        try:
            updated = self._token_writer(
                sandbox_raw or None,
                production_raw or None,
            )
        except Exception as exc:  # pragma: no cover - file system guard
            LOGGER.exception("Failed to store tokens: %s", exc)
            self._set_feedback(f"Failed to store tokens: {exc}", "#f87171")
            return
        finally:
            self._sandbox_entry.delete(0, tk.END)
            self._production_entry.delete(0, tk.END)
        if updated:
            self._set_feedback("Tokens updated.", "#22c55e")
        else:
            self._set_feedback("No changes detected.", "#facc15")
        self._refresh_token_status()

    def _restart_system(self) -> None:
        if self._restart_callback is None:
            self._set_feedback("Restart is unavailable in this mode.", "#facc15")
            return
        try:
            self._restart_callback()
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Failed to request orchestrator restart: %s", exc)
            self._set_feedback(f"Failed to request restart: {exc}", "#f87171")
            return
        self._set_feedback("Restart requested.", "#22c55e")

    def _set_feedback(self, message: str, colour: str = "#e2e8f0") -> None:
        if self._token_feedback_label is None:
            return
        self._token_feedback_label.config(text=message, foreground=colour)

    def _bind_copy_paste(self, widget) -> None:
        if tk is None:
            return

        def _copy(event):
            try:
                text = widget.selection_get()
            except tk.TclError:
                return "break"
            widget.clipboard_clear()
            widget.clipboard_append(text)
            return "break"

        def _paste(event):
            try:
                text = widget.clipboard_get()
            except tk.TclError:
                return "break"
            widget.delete(0, tk.END)
            widget.insert(tk.INSERT, text)
            return "break"

        for sequence in ("<Control-c>", "<Control-C>", "<Command-c>", "<Command-C>"):
            widget.bind(sequence, _copy)
        for sequence in ("<Control-v>", "<Control-V>", "<Command-v>", "<Command-V>"):
            widget.bind(sequence, _paste)


def run_dashboard(
    repository_path: Path,
    *,
    status_provider: Optional[Callable[[], DashboardStatus]] = None,
    refresh_interval_ms: int = 1000,
    signal_limit: int = 25,
    background: bool = False,
    headless: bool = False,
    title: str = "Scalp System Dashboard",
    config_path: Optional[Path] = None,
    restart_callback: Optional[Callable[[], None]] = None,
    token_status_provider: Optional[Callable[[], dict[str, bool]]] = None,
    token_writer: Optional[Callable[[Optional[str], Optional[str]], bool]] = None,
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
            config_path=config_path,
            restart_callback=restart_callback,
            token_status_provider=token_status_provider,
            token_writer=token_writer,
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
