"""Connectivity failover coordinator for switching between network paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from ..config.base import ConnectivityConfig


@dataclass(frozen=True)
class ConnectivityEvent:
    kind: Literal["failover", "recovery"]
    channel: str
    reason: Optional[str] = None


class ConnectivityFailover:
    """Tracks connectivity health and activates LTE failover when needed."""

    def __init__(self, config: ConnectivityConfig) -> None:
        self._config = config
        self._active_channel = config.primary_label
        self._failure_count = 0
        self._success_streak = 0
        self._last_event: Optional[ConnectivityEvent] = None
        self._last_failure_at: Optional[datetime] = None

    @property
    def active_channel(self) -> str:
        return self._active_channel

    def record_failure(self, reason: str) -> tuple[Optional[ConnectivityEvent], float]:
        """Register a connectivity failure and maybe trigger LTE failover."""

        self._failure_count += 1
        self._last_failure_at = datetime.utcnow()
        delay = 0.0
        event: Optional[ConnectivityEvent] = None
        if self._active_channel == self._config.primary_label:
            if self._failure_count >= self._config.failure_threshold:
                self._active_channel = self._config.backup_label
                self._success_streak = 0
                event = ConnectivityEvent("failover", self._active_channel, reason)
                delay = self._config.failover_latency_ms / 1000.0
        else:
            delay = self._config.failover_latency_ms / 1000.0
        self._last_event = event
        return event, delay

    def record_success(self) -> Optional[ConnectivityEvent]:
        """Record successful packets and maybe restore the primary channel."""

        self._failure_count = 0
        if self._active_channel == self._config.backup_label:
            self._success_streak += 1
            if self._success_streak >= self._config.recovery_message_count:
                self._active_channel = self._config.primary_label
                self._success_streak = 0
                event = ConnectivityEvent("recovery", self._active_channel)
                self._last_event = event
                return event
        else:
            self._success_streak = 0
        return None

    def snapshot(self) -> dict:
        return {
            "active_channel": self._active_channel,
            "failure_count": self._failure_count,
            "success_streak": self._success_streak,
            "last_event": self._last_event.kind if self._last_event else None,
            "last_failure_at": self._last_failure_at.isoformat()
            if self._last_failure_at
            else None,
        }


__all__ = ["ConnectivityFailover", "ConnectivityEvent"]

