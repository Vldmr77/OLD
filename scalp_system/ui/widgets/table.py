"""Reusable table widget with sortable columns."""
from __future__ import annotations

from typing import Iterable, Mapping

try:  # pragma: no cover - tkinter is optional in headless envs
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    ttk = None  # type: ignore


class SimpleTable(ttk.Frame):  # type: ignore[misc]
    """Treeview wrapper with convenience update helpers."""

    def __init__(self, master: tk.Misc, columns: Iterable[str], *, height: int = 8) -> None:  # type: ignore[valid-type]
        super().__init__(master)
        self._columns = list(columns)
        self._tree = ttk.Treeview(self, columns=self._columns, show="headings", height=height)
        for column in self._columns:
            self._tree.heading(column, text=column.title(), command=lambda c=column: self._sort(c, False))
            self._tree.column(column, width=120, anchor="center")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    # ------------------------------------------------------------------
    def update_rows(self, rows: Iterable[Mapping[str, object]]) -> None:
        self._tree.delete(*self._tree.get_children())
        for row in rows:
            values = [row.get(col, "") for col in self._columns]
            self._tree.insert("", "end", values=values)

    # ------------------------------------------------------------------
    def _sort(self, column: str, reverse: bool) -> None:
        data = [(self._tree.set(child, column), child) for child in self._tree.get_children("")]
        data.sort(reverse=reverse)
        for index, (_, child) in enumerate(data):
            self._tree.move(child, "", index)
        self._tree.heading(column, command=lambda: self._sort(column, not reverse))


__all__ = ["SimpleTable"]
