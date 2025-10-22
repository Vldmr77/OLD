"""Graph visualisation for module statuses."""
from __future__ import annotations

try:  # pragma: no cover - tkinter not available in headless tests
    import tkinter as tk  # type: ignore
except Exception:  # pragma: no cover
    tk = None  # type: ignore

STATE_COLOURS = {
    "ready": "#3CB371",
    "running": "#3CB371",
    "active": "#3CB371",
    "paused": "#D3D3D3",
    "idle": "#D3D3D3",
    "error": "#CD5C5C",
    "failed": "#CD5C5C",
    "warning": "#F0E68C",
    "unknown": "#B0B0B0",
}


class StatusGraph(tk.Canvas):  # type: ignore[misc]
    """Canvas drawing subsystem blocks and connectors."""

    WIDTH = 980
    HEIGHT = 360

    NODES = {
        "sources": (60, 60, 200, 120, "Источники данных"),
        "data": (60, 140, 200, 200, "Data Engine"),
        "features": (240, 140, 380, 200, "Feature Pipeline"),
        "ml": (420, 140, 560, 200, "ML Engine"),
        "bus": (600, 140, 760, 200, "Шина событий"),
        "risk": (780, 60, 920, 120, "Risk Engine"),
        "execution": (780, 140, 920, 200, "Execution"),
        "adapter": (780, 220, 920, 280, "Tinkoff Adapter"),
        "storage": (420, 240, 560, 300, "Хранилище"),
        "heartbeat": (240, 40, 380, 100, "Мониторинг"),
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
        super().__init__(master, width=self.WIDTH, height=self.HEIGHT, highlightthickness=0)
        self.configure(bg="#1E1E1E")
        self._node_items: dict[str, int] = {}
        self._label_items: dict[str, int] = {}
        self._draw_static()

    # ------------------------------------------------------------------
    def _draw_static(self) -> None:
        self.delete("all")
        for start, end in self.CONNECTIONS:
            sx1, sy1, sx2, sy2, _ = self.NODES[start]
            ex1, ey1, ex2, ey2, _ = self.NODES[end]
            self.create_line((sx2, (sy1 + sy2) / 2), (ex1, (ey1 + ey2) / 2), fill="#808080", width=2)
        for key, (x1, y1, x2, y2, label) in self.NODES.items():
            rect = self.create_rectangle(x1, y1, x2, y2, fill=STATE_COLOURS["idle"], outline="")
            text = self.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=label, fill="white", font=("Arial", 11))
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
            base_label = self.NODES[key][4]
            if detail:
                text = f"{base_label}\n{detail}"
            else:
                text = base_label
            self.itemconfigure(label_id, text=text)


__all__ = ["StatusGraph"]
