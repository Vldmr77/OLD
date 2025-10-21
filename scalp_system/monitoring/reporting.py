"""Performance reporting utilities."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional

from ..risk.engine import RiskEngine
from ..storage.repository import SQLiteRepository


def _average_confidence(records: Iterable[dict]) -> float:
    values = [abs(record["confidence"]) for record in records]
    if not values:
        return 0.0
    return float(mean(values))


class PerformanceReporter:
    """Aggregates trading activity into periodic JSONL reports."""

    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        risk_engine: RiskEngine,
        report_path: Path,
        interval_minutes: int = 60,
        max_history_days: int = 14,
        enabled: bool = True,
        notifications=None,
    ) -> None:
        self._repository = repository
        self._risk_engine = risk_engine
        self._report_path = report_path
        self._interval = max(1, int(interval_minutes))
        self._enabled = enabled
        self._max_history = max(1, int(max_history_days))
        self._notifications = notifications
        self._last_emitted: Optional[datetime] = None
        self._report_path.parent.mkdir(parents=True, exist_ok=True)

    async def maybe_emit(self, now: Optional[datetime] = None) -> Optional[dict]:
        if not self._enabled:
            return None
        timestamp = now or datetime.now(timezone.utc)
        if self._last_emitted and (timestamp - self._last_emitted).total_seconds() < self._interval * 60:
            return None
        report = self._build_report(timestamp)
        self._persist(report)
        self._last_emitted = timestamp
        if self._notifications and hasattr(self._notifications, "notify_performance_summary"):
            summary = report["summary"]
            await self._notifications.notify_performance_summary(
                realized_pnl=summary["realized_pnl"],
                signal_count=summary["signal_count"],
                avg_confidence=summary["avg_confidence"],
                halted=summary["halted"],
            )
        return report

    def _build_report(self, timestamp: datetime) -> dict:
        cutoff = timestamp - timedelta(days=self._max_history)
        signals = self._repository.fetch_signals(since=cutoff)
        avg_confidence = _average_confidence(signals)
        risk = self._risk_engine.performance_snapshot()
        summary = {
            "timestamp": timestamp.isoformat(),
            "signal_count": len(signals),
            "avg_confidence": avg_confidence,
            "realized_pnl": risk["realized_pnl"],
            "consecutive_losses": risk["consecutive_losses"],
            "daily_loss_triggered": risk["daily_loss_triggered"],
            "halted": risk["halt_trading"],
        }
        breakdown = {}
        for record in signals:
            stats = breakdown.setdefault(record["figi"], {"count": 0, "avg_confidence": 0.0})
            stats["count"] += 1
            stats["avg_confidence"] += abs(record["confidence"])
        for figi, stats in breakdown.items():
            stats["avg_confidence"] = stats["avg_confidence"] / max(stats["count"], 1)
        return {
            "summary": summary,
            "per_instrument": breakdown,
        }

    def _persist(self, payload: dict) -> None:
        self._prune_history(payload["summary"]["timestamp"])
        with self._report_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _prune_history(self, timestamp_str: str) -> None:
        if not self._report_path.exists():
            return
        cutoff = datetime.fromisoformat(timestamp_str) - timedelta(days=self._max_history)
        lines = self._report_path.read_text(encoding="utf-8").splitlines()
        kept = []
        for line in lines:
            try:
                data = json.loads(line)
                ts = datetime.fromisoformat(data["summary"]["timestamp"])
            except Exception:
                continue
            if ts >= cutoff:
                kept.append(line)
        self._report_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


__all__ = ["PerformanceReporter"]
