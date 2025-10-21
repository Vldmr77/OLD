"""Latency monitoring utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class LatencyAlert:
    stage: str
    latency_ms: float
    threshold_ms: float
    severity: str
    consecutive: int


class LatencyMonitor:
    """Tracks stage latencies and emits alerts on breaches."""

    def __init__(self, thresholds: Dict[str, float], *, violation_limit: int = 3) -> None:
        self._thresholds = thresholds
        self._violation_limit = max(1, violation_limit)
        self._violations: Dict[str, int] = {}

    def observe(self, stage: str, latency_ms: float) -> Optional[LatencyAlert]:
        threshold = self._thresholds.get(stage)
        if threshold is None:
            return None
        if latency_ms <= threshold:
            self._violations.pop(stage, None)
            return None
        count = self._violations.get(stage, 0) + 1
        self._violations[stage] = count
        severity = "critical" if count >= self._violation_limit else "warn"
        if severity == "critical":
            self._violations[stage] = 0
        return LatencyAlert(
            stage=stage,
            latency_ms=latency_ms,
            threshold_ms=threshold,
            severity=severity,
            consecutive=count,
        )


__all__ = ["LatencyMonitor", "LatencyAlert"]
