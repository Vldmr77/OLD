"""Trading session guard to enforce schedule windows."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional

from ..config.base import SessionScheduleConfig


@dataclass
class SessionState:
    """Represents the evaluated schedule state."""

    active: bool
    reason: Optional[str] = None
    next_transition: Optional[datetime] = None
    changed: bool = False


class SessionGuard:
    """Tracks schedule adherence and emits state transitions."""

    def __init__(self, config: SessionScheduleConfig) -> None:
        self._config = config
        self._last_active: Optional[bool] = None
        self._last_reason: Optional[str] = None

    def evaluate(self, now: Optional[datetime] = None) -> SessionState:
        now = now or datetime.utcnow()
        active, reason, transition = self._compute_state(now)
        changed = (
            self._last_active is None
            or active != self._last_active
            or reason != self._last_reason
        )
        self._last_active = active
        self._last_reason = reason
        return SessionState(
            active=active,
            reason=reason,
            next_transition=transition,
            changed=changed,
        )

    def _compute_state(self, now: datetime) -> tuple[bool, Optional[str], Optional[datetime]]:
        if not self._config.enabled:
            return True, None, None

        allowed = self._config.allowed_weekdays
        if allowed and now.weekday() not in allowed:
            next_day = self._next_allowed_date(now, include_today=False)
            next_open = self._build_datetime(next_day, self._config.start_time)
            if next_open:
                next_open -= timedelta(minutes=self._config.pre_open_minutes)
            return False, "weekday_blocked", next_open

        start_time = _parse_time(self._config.start_time)
        end_time = _parse_time(self._config.end_time)
        pre_open = timedelta(minutes=self._config.pre_open_minutes)
        post_close = timedelta(minutes=self._config.post_close_minutes)

        open_dt = datetime.combine(now.date(), start_time) - pre_open
        close_dt = datetime.combine(now.date(), end_time) + post_close

        if close_dt <= open_dt:
            if now >= open_dt:
                close_dt += timedelta(days=1)
            else:
                open_dt -= timedelta(days=1)

        if now < open_dt:
            return False, "pre_open", open_dt

        if now > close_dt:
            next_day = self._next_allowed_date(now, include_today=False)
            next_open = self._build_datetime(next_day, self._config.start_time)
            if next_open:
                next_open -= pre_open
            return False, "post_close", next_open

        return True, None, close_dt

    def _next_allowed_date(
        self, now: datetime, *, include_today: bool
    ) -> Optional[datetime.date]:
        allowed = self._config.allowed_weekdays
        if not allowed:
            return None
        start_offset = 0 if include_today else 1
        for offset in range(start_offset, 8):
            candidate = now.date() + timedelta(days=offset)
            if candidate.weekday() in allowed:
                return candidate
        return None

    def _build_datetime(
        self, target_date: Optional[datetime.date], time_str: str
    ) -> Optional[datetime]:
        if target_date is None:
            return None
        target_time = _parse_time(time_str)
        return datetime.combine(target_date, target_time)


def _parse_time(value: str) -> time:
    hour, minute = (int(part) for part in value.split(":", 1))
    hour = max(0, min(23, hour))
    minute = max(0, min(59, minute))
    return time(hour=hour, minute=minute)


__all__ = ["SessionGuard", "SessionState"]
