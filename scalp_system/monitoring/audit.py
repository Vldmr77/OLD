"""Audit logging utilities in W3C Extended Log format."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_FIELDS: Sequence[str] = ("date", "time", "c-category", "c-event", "s-detail")


@dataclass
class AuditLogger:
    """Append-only audit logger.

    Logs events in the W3C Extended Log format with a configurable list
    of fields. By default the format matches ``date time c-category
    c-event s-detail``.
    """

    path: Path
    fields: Sequence[str] = DEFAULT_FIELDS

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8") as handle:
                handle.write("#Version: 1.0\n")
                handle.write(f"#Fields: {' '.join(self.fields)}\n")

    def log(self, category: str, event: str, detail: str) -> None:
        """Append a new record to the audit log."""

        timestamp = datetime.utcnow()
        safe_detail = detail.replace(" ", "_")
        line = f"{timestamp:%Y-%m-%d} {timestamp:%H:%M:%S} {category} {event} {safe_detail}\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def iter_lines(self) -> Iterable[str]:
        with self.path.open("r", encoding="utf-8") as handle:
            yield from handle


__all__ = ["AuditLogger", "DEFAULT_FIELDS"]
