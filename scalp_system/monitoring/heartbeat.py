"""Heartbeat monitoring for market data streams."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable, Optional


@dataclass
class HeartbeatStatus:
    """Represents a heartbeat miss condition."""

    missed_intervals: int
    last_beat: Optional[datetime]
    elapsed_seconds: float
    last_context: Optional[str] = None
    last_failure_reason: Optional[str] = None


class HeartbeatMonitor:
    """Detects prolonged pauses in upstream market data."""

    def __init__(
        self,
        *,
        enabled: bool,
        interval_seconds: float,
        miss_threshold: int,
        clock: Callable[[], datetime] = datetime.utcnow,
    ) -> None:
        self.enabled = enabled
        self.interval_seconds = max(float(interval_seconds), 0.1)
        self.miss_threshold = max(int(miss_threshold), 1)
        self._clock = clock
        self._last_beat: datetime = self._clock()
        self._last_context: Optional[str] = None
        self._last_failure_reason: Optional[str] = None
        self._misses = 0

    async def monitor(
        self,
        callback: Callable[[HeartbeatStatus], Awaitable[None]],
        poll_interval: Optional[float] = None,
    ) -> None:
        """Periodically evaluates the heartbeat window."""

        poll = poll_interval if poll_interval and poll_interval > 0 else self.interval_seconds
        try:
            while True:
                await asyncio.sleep(poll)
                if not self.enabled:
                    continue
                now = self._clock()
                elapsed = (now - self._last_beat).total_seconds()
                if elapsed < self.interval_seconds:
                    self._misses = 0
                    continue
                self._misses += 1
                if self._misses < self.miss_threshold:
                    continue
                status = HeartbeatStatus(
                    missed_intervals=self._misses,
                    last_beat=self._last_beat,
                    elapsed_seconds=elapsed,
                    last_context=self._last_context,
                    last_failure_reason=self._last_failure_reason,
                )
                await callback(status)
                self._misses = 0
                self._last_failure_reason = None
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            raise

    def record_beat(self, context: Optional[str] = None) -> None:
        """Registers an incoming message as a healthy heartbeat."""

        self._last_beat = self._clock()
        self._last_context = context
        self._misses = 0
        self._last_failure_reason = None

    def record_failure(self, reason: str) -> None:
        """Stores the most recent failure reason for diagnostics."""

        self._last_failure_reason = reason

    def diagnostics(self) -> dict:
        """Expose internal heartbeat counters for observability."""

        return {
            "enabled": self.enabled,
            "interval_seconds": self.interval_seconds,
            "miss_threshold": self.miss_threshold,
            "last_beat": self._last_beat,
            "misses": self._misses,
            "last_context": self._last_context,
            "last_failure_reason": self._last_failure_reason,
        }


__all__ = ["HeartbeatMonitor", "HeartbeatStatus"]
