"""Graph visualisation for module statuses."""
from __future__ import annotations

try:  # pragma: no cover - tkinter not available in headless tests
    import tkinter as tk  # type: ignore
except Exception:  # pragma: no cover
    tk = None  # type: ignore

from textwrap import shorten

from ..theme import PALETTE

STATE_COLOURS = {
    "ready": PALETTE.accent_alt,
    "running": PALETTE.accent_alt,
    "active": PALETTE.accent_alt,
    "paused": PALETTE.surface_alt,
    "idle": PALETTE.surface_alt,
    "error": PALETTE.error,
    "failed": PALETTE.error,
    "warning": PALETTE.warning,
    "unknown": "#4b5563",
}


class StatusGraph(tk.Canvas):  # type: ignore[misc]
    """Canvas drawing subsystem blocks and connectors."""

    COLUMN_SPACING = 220
    ROW_SPACING = 120
    NODE_WIDTH = 190
    NODE_HEIGHT = 72
    MARGIN_X = 140
    MARGIN_Y = 40

    GRID = {
        "sources": (0, 1, "Источники данных"),
        "data": (1, 1, "Data Engine"),
        "features": (2, 1, "Feature Pipeline"),
        "ml": (3, 1, "ML Engine"),
        "bus": (4, 1, "Шина событий"),
        "risk": (5, 0, "Risk Engine"),
        "execution": (5, 1, "Execution"),
        "adapter": (5, 2, "Tinkoff Adapter"),
        "storage": (3, 2, "Хранилище"),
        "heartbeat": (2, 0, "Мониторинг"),
    }

    CONNECTIONS = (
        ("sources", "data"),
        ("data", "features"),
        ("features", "ml"),
        ("ml", "bus"),
        ("bus", "risk"),
        ("bus", "execution"),
        ("execution", "adapter"),
        ("adapter", "storage"),
        ("heartbeat", "data"),
        ("heartbeat", "ml"),
        ("heartbeat", "risk"),
        ("heartbeat", "execution"),
    )

    def __init__(self, master: tk.Misc) -> None:  # type: ignore[valid-type]
        self._node_geometry = self._compute_geometry()
        width = int(max(x2 for x1, y1, x2, y2, _ in self._node_geometry.values()) + self.MARGIN_X)
        height = int(max(y2 for x1, y1, x2, y2, _ in self._node_geometry.values()) + self.MARGIN_Y)
        super().__init__(master, width=width, height=height, highlightthickness=0)
        self.configure(bg=PALETTE.surface, bd=0)
        self._node_items: dict[str, int] = {}
        self._label_items: dict[str, int] = {}
        self._draw_static()

    # ------------------------------------------------------------------
    def _compute_geometry(self) -> dict[str, tuple[float, float, float, float, str]]:
        layout: dict[str, tuple[float, float, float, float, str]] = {}
        for key, (column, row, label) in self.GRID.items():
            cx = self.MARGIN_X + column * self.COLUMN_SPACING
            cy = self.MARGIN_Y + row * self.ROW_SPACING
            half_w = self.NODE_WIDTH / 2
            half_h = self.NODE_HEIGHT / 2
            layout[key] = (cx - half_w, cy - half_h, cx + half_w, cy + half_h, label)
        return layout

    # ------------------------------------------------------------------
    def _draw_static(self) -> None:
        self.delete("all")
        for start, end in self.CONNECTIONS:
            sx1, sy1, sx2, sy2, _ = self._node_geometry[start]
            ex1, ey1, ex2, ey2, _ = self._node_geometry[end]
            self.create_line(
                (sx2, (sy1 + sy2) / 2),
                (ex1, (ey1 + ey2) / 2),
                fill=PALETTE.outline,
                width=2,
                smooth=True,
                splinesteps=12,
                arrow=tk.LAST,
                arrowshape=(14, 16, 6),
                capstyle=tk.ROUND,
            )
        for key, (x1, y1, x2, y2, label) in self._node_geometry.items():
            rect = self.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=STATE_COLOURS["idle"],
                outline=PALETTE.outline,
                width=2,
            )
            text = self.create_text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                text=label,
                fill=PALETTE.text,
                font=("Segoe UI", 11),
                width=self.NODE_WIDTH - 24,
            )
            self._node_items[key] = rect
            self._label_items[key] = text

    # ------------------------------------------------------------------
    def update_modules(self, modules: dict[str, dict[str, str]]) -> None:
        """Update colours and tooltips based on module states."""

        for key, rect_id in self._node_items.items():
            module = modules.get(key)
            state = (module or {}).get("state", "unknown").lower()
            colour = STATE_COLOURS.get(state, STATE_COLOURS["unknown"])
            detail = (module or {}).get("detail", "")
            self.itemconfigure(rect_id, fill=colour)
            label_id = self._label_items[key]
            base_label = self._node_geometry[key][4]
            if detail:
                detail_text = shorten(str(detail), width=30, placeholder="…")
                text = f"{base_label}\n{detail_text}"
            else:
                text = base_label
            self.itemconfigure(label_id, text=text)


__all__ = ["StatusGraph"]
