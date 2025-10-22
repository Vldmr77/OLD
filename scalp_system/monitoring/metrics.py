"""In-memory metrics registry for observability hooks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MetricsRegistry:
    signal_confidence: Dict[str, float] = field(default_factory=dict)
    risk_var: Dict[str, float] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    liquidity_score: float = 1.0

    def record_signal(self, figi: str, confidence: float) -> None:
        self.signal_confidence[figi] = confidence

    def record_risk(self, figi: str, var_value: float) -> None:
        self.risk_var[figi] = var_value

    def record_latency(self, stage: str, latency_ms: float) -> None:
        self.latency_ms[stage] = latency_ms

    def record_liquidity(self, score: float) -> None:
        self.liquidity_score = max(0.0, min(1.0, float(score)))


__all__ = ["MetricsRegistry"]
