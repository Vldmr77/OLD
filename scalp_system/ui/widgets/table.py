"""Reusable table widget with sortable columns."""
from __future__ import annotations

from typing import Iterable, Mapping

try:  # pragma: no cover - tkinter is optional in headless envs
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    ttk = None  # type: ignore

from ..theme import tag_colours

NUMERIC_SUFFIXES = ("qty", "latency", "eta", "weight", "pnl", "price", "score", "size")


class SimpleTable(ttk.Frame):  # type: ignore[misc]
    """Treeview wrapper with convenience update helpers."""

    def __init__(self, master: tk.Misc, columns: Iterable[str], *, height: int = 8) -> None:  # type: ignore[valid-type]
        super().__init__(master)
        self._columns = list(columns)
        style_name = "Dashboard.Treeview"
        self._tree = ttk.Treeview(
            self,
            columns=self._columns,
            show="headings",
            height=height,
            style=style_name,
        )
        for column in self._columns:
            self._tree.heading(column, text=column.title(), command=lambda c=column: self._sort(c, False))
            anchor = "e" if column.lower().endswith(NUMERIC_SUFFIXES) else "w"
            self._tree.column(column, width=140, anchor=anchor, stretch=True)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        colours = tag_colours()
        self._tree.tag_configure("odd", background=colours["odd"])
        self._tree.tag_configure("even", background=colours["even"])
        self._tree.tag_configure("error", background=colours["error"], foreground="white")
        self._tree.tag_configure("warning", background=colours["warning"], foreground="black")
        self._tree.tag_configure("ok", background=colours["ok"], foreground="black")

    # ------------------------------------------------------------------
    def update_rows(self, rows: Iterable[Mapping[str, object]]) -> None:
        self._tree.delete(*self._tree.get_children())
        for index, row in enumerate(rows):
            values = [row.get(col, "") for col in self._columns]
            severity = str(
                row.get("state")
                or row.get("status")
                or row.get("severity")
                or ""
            ).lower()
            tags = ["odd" if index % 2 else "even"]
            if any(key in severity for key in ("error", "fail", "halt", "critical")):
                tags.append("error")
            elif any(key in severity for key in ("warn", "slow", "degraded")):
                tags.append("warning")
            elif severity in {"ready", "active", "ok", "success"}:
                tags.append("ok")
            self._tree.insert("", "end", values=values, tags=tags)

    # ------------------------------------------------------------------
    def _sort(self, column: str, reverse: bool) -> None:
        data = [(self._tree.set(child, column), child) for child in self._tree.get_children("")]
        data.sort(reverse=reverse)
        for index, (_, child) in enumerate(data):
            self._tree.move(child, "", index)
        self._tree.heading(column, command=lambda: self._sort(column, not reverse))


__all__ = ["SimpleTable"]
