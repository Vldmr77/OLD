"""Manual override guard for emergency trading halts."""
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..config.base import ManualOverrideConfig


@dataclass
class ManualOverrideStatus:
    """Represents the current manual override state."""

    halted: bool
    reason: Optional[str]
    expires_at: Optional[datetime]


class ManualOverrideGuard:
    """Checks for manual intervention signals via a flag file."""

    def __init__(self, config: ManualOverrideConfig) -> None:
        self._config = config
        self._flag_path: Path = config.flag_path
        self._config.ensure()

    @property
    def poll_interval(self) -> float:
        return float(self._config.poll_interval_seconds)

    def status(self) -> ManualOverrideStatus:
        if not self._config.enabled:
            return ManualOverrideStatus(False, None, None)
        if not self._flag_path.exists():
            return ManualOverrideStatus(False, None, None)
        reason = self._read_reason()
        expires_at = self._expiry()
        if expires_at and datetime.utcnow() >= expires_at:
            with suppress(FileNotFoundError):
                self._flag_path.unlink()
            return ManualOverrideStatus(False, None, None)
        return ManualOverrideStatus(True, reason, expires_at)

    def clear(self) -> None:
        with suppress(FileNotFoundError):
            self._flag_path.unlink()

    def activate(self, reason: Optional[str] = None) -> None:
        """Create or update the manual override flag file."""

        if not self._config.enabled:
            return
        message = (reason or "manual override").strip() or "manual override"
        self._flag_path.parent.mkdir(parents=True, exist_ok=True)
        self._flag_path.write_text(message, encoding="utf-8")

    def _read_reason(self) -> Optional[str]:
        try:
            content = self._flag_path.read_text(encoding="utf-8").strip()
        except OSError:
            return "manual override"
        return content or "manual override"

    def _expiry(self) -> Optional[datetime]:
        if self._config.auto_resume_minutes is None:
            return None
        try:
            mtime = datetime.utcfromtimestamp(self._flag_path.stat().st_mtime)
        except OSError:
            return None
        return mtime + timedelta(minutes=self._config.auto_resume_minutes)


__all__ = ["ManualOverrideGuard", "ManualOverrideStatus"]
