from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, Optional, Sequence

from ..config.base import FallbackConfig
from ..data.models import OrderBook
from .engine import MLSignal


@dataclass(slots=True)
class FallbackDecision:
    signal: MLSignal
    reason: str
    imbalance: float
    volatility: float


class FallbackSignalGenerator:
    """Generate heuristic signals when the ML stack is unavailable."""

    def __init__(self, config: FallbackConfig) -> None:
        self._config = config
        self._recent: Deque[datetime] = deque()
        self._last_rejection: Optional[str] = None

    @property
    def last_rejection(self) -> Optional[str]:
        return self._last_rejection

    def remaining_allowance(self) -> int:
        now = datetime.utcnow()
        self._prune(now)
        return max(0, self._config.max_signals_per_hour - len(self._recent))

    def generate(
        self, window: Sequence[OrderBook], *, reason: str
    ) -> Optional[FallbackDecision]:
        self._last_rejection = None
        if not self._config.enabled:
            self._last_rejection = "disabled"
            return None
        if not window:
            self._last_rejection = "empty_window"
            return None
        now = datetime.utcnow()
        self._prune(now)
        if len(self._recent) >= self._config.max_signals_per_hour:
            self._last_rejection = "quota_exceeded"
            return None
        latest = window[-1]
        imbalance = self._imbalance(latest)
        volatility = self._volatility(window)
        if volatility < self._config.volatility_threshold:
            self._last_rejection = "low_volatility"
            return None
        threshold = self._config.imbalance_threshold
        direction: Optional[int] = None
        suffix = "neutral"
        if imbalance >= threshold:
            direction = 1
            suffix = "long"
        elif imbalance <= -threshold:
            direction = -1
            suffix = "short"
        if direction is None:
            self._last_rejection = "low_imbalance"
            return None
        signal = MLSignal(
            figi=latest.figi,
            direction=direction,
            confidence=self._config.confidence,
            source="fallback",
        )
        self._recent.append(now)
        return FallbackDecision(
            signal=signal,
            reason=f"{reason}:{suffix}",
            imbalance=imbalance,
            volatility=volatility,
        )

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=1)
        while self._recent and self._recent[0] < cutoff:
            self._recent.popleft()

    @staticmethod
    def _imbalance(book: OrderBook) -> float:
        bids = book.bids[:10]
        asks = book.asks[:10]
        bid_volume = sum(level.quantity for level in bids)
        ask_volume = sum(level.quantity for level in asks)
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        return (bid_volume - ask_volume) / total

    @staticmethod
    def _volatility(window: Sequence[OrderBook]) -> float:
        if len(window) < 2:
            return 0.0
        mids = [book.mid_price() for book in window]
        deltas = [abs(curr - prev) for prev, curr in zip(mids[:-1], mids[1:])]
        if not deltas or mids[-1] == 0:
            return 0.0
        return sum(deltas) / len(deltas) / abs(mids[-1])


__all__ = ["FallbackSignalGenerator", "FallbackDecision"]

