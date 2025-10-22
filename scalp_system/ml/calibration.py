"""Calibration queue management for model retraining triggers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CalibrationRequest:
    reason: str
    triggered_at: datetime
    metadata: Dict[str, Any]


class CalibrationCoordinator:
    """Persists calibration requests with optional deduplication."""

    def __init__(self, *, queue_path: Path, dedupe_ttl_seconds: int = 900) -> None:
        self._queue_path = queue_path
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)
        self._dedupe_ttl = timedelta(seconds=dedupe_ttl_seconds)
        self._recent: Dict[str, datetime] = {}

    def enqueue(self, *, reason: str, metadata: Dict[str, Any], dedupe_key: Optional[str] = None) -> bool:
        now = datetime.utcnow()
        if dedupe_key:
            last = self._recent.get(dedupe_key)
            if last and now - last < self._dedupe_ttl:
                return False
            self._recent[dedupe_key] = now
        request = CalibrationRequest(reason=reason, triggered_at=now, metadata=metadata)
        payload = {
            "reason": request.reason,
            "triggered_at": request.triggered_at.isoformat(timespec="seconds"),
            "metadata": request.metadata,
        }
        with self._queue_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True


__all__ = ["CalibrationCoordinator", "CalibrationRequest"]
