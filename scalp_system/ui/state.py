"""Shared dashboard state with convenience selectors."""
from __future__ import annotations

import copy
import threading
from typing import Any, Iterable


class State:
    """Thread-safe container for the latest `/status` payload."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def update_from_status(self, payload: dict[str, Any] | None) -> None:
        with self._lock:
            self._data = copy.deepcopy(payload or {})

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    # ------------------------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        parts = path.split(".") if path else []
        with self._lock:
            node: Any = self._data
            for part in parts:
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return copy.deepcopy(node)

    # ------------------------------------------------------------------
    def modules(self) -> list[dict[str, Any]]:
        value = self.get("modules", [])
        return value if isinstance(value, list) else []

    # ------------------------------------------------------------------
    def alerts(self) -> list[dict[str, Any]]:
        value = self.get("alerts", [])
        return value if isinstance(value, list) else []

    # ------------------------------------------------------------------
    def signals(self) -> list[dict[str, Any]]:
        value = self.get("signals", [])
        return value if isinstance(value, list) else []

    # ------------------------------------------------------------------
    def orders(self) -> list[dict[str, Any]]:
        value = self.get("orders", [])
        return value if isinstance(value, list) else []

    # ------------------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        value = self.get("metrics", {})
        return value if isinstance(value, dict) else {}

    # ------------------------------------------------------------------
    def data_metrics(self) -> dict[str, Any]:
        value = self.get("data", {})
        return value if isinstance(value, dict) else {}

    # ------------------------------------------------------------------
    def iter_modules(self, names: Iterable[str]) -> list[dict[str, Any]]:
        lookup = {m.get("name"): m for m in self.modules()}
        return [lookup.get(name, {"name": name, "state": "unknown"}) for name in names]


__all__ = ["State"]
