"""Lightweight tail view for runtime logs."""
from __future__ import annotations

import collections
from typing import Iterable

try:  # pragma: no cover - tkinter optional
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    ttk = None  # type: ignore


class LogTail(ttk.Frame):  # type: ignore[misc]
    """Simple scrollable buffer for log lines."""

    def __init__(self, master: tk.Misc, *, max_lines: int = 500) -> None:  # type: ignore[valid-type]
        super().__init__(master)
        self._max_lines = max_lines
        self._buffer = collections.deque(maxlen=max_lines)
        self._text = tk.Text(self, height=15, state="disabled", wrap="none", background="#111", foreground="#eee")
        y_scroll = ttk.Scrollbar(self, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=y_scroll.set)
        self._text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def extend(self, lines: Iterable[str]) -> None:
        for line in lines:
            self._buffer.append(line.rstrip())
        self._text.configure(state="normal")
        self._text.delete("1.0", tk.END)
        self._text.insert(tk.END, "\n".join(self._buffer))
        self._text.see(tk.END)
        self._text.configure(state="disabled")


__all__ = ["LogTail"]
