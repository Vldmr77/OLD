"""Storage abstractions backed by sqlite."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional


class SQLiteRepository:
    def __init__(
        self,
        path: Path,
        *,
        cache_min_interval: float = 1.0,
        cache_max_interval: float = 5.0,
        cache_target_buffer: int = 50,
    ) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fallback_path = self._path.with_suffix(".fallback.jsonl")
        self._cache_path = self._path.with_suffix(".cache.jsonl")
        self._cache_buffer: List[dict] = []
        self._last_cache_flush = time.monotonic()
        self._min_cache_interval = max(0.1, cache_min_interval)
        self._max_cache_interval = max(self._min_cache_interval, cache_max_interval)
        self._target_buffer = max(1, cache_target_buffer)
        self._cache_interval = self._max_cache_interval
        self._last_record_ts: Optional[float] = None
        self._arrival_ema: Optional[float] = None
        self._initialise()

    def _initialise(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    figi TEXT NOT NULL,
                    direction INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        try:
            yield conn
        finally:
            conn.close()

    def persist_signal(self, figi: str, direction: int, confidence: float) -> None:
        record = {
            "figi": figi,
            "direction": direction,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with self._connection() as conn:
                conn.execute(
                    "INSERT INTO signals(figi, direction, confidence) VALUES (?, ?, ?)",
                    (figi, direction, confidence),
                )
                conn.commit()
        except sqlite3.Error:
            self._fallback_write(record)
            return
        self._record_cache(record)

    def fetch_signals(
        self,
        *,
        limit: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> List[dict]:
        query = "SELECT figi, direction, confidence, created_at FROM signals"
        params: List[object] = []
        if since is not None:
            query += " WHERE datetime(created_at) >= datetime(?)"
            params.append(since.isoformat())
        query += " ORDER BY datetime(created_at) DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        results = []
        for figi, direction, confidence, created_at in rows:
            timestamp = self._parse_timestamp(created_at)
            results.append(
                {
                    "figi": str(figi),
                    "direction": int(direction),
                    "confidence": float(confidence),
                    "timestamp": timestamp,
                }
            )
        return results

    def _record_cache(self, record: dict) -> None:
        self._cache_buffer.append(record)
        now = time.monotonic()
        self._adjust_cache_interval(now)
        if now - self._last_cache_flush >= self._cache_interval:
            self._flush_cache()
            self._last_cache_flush = now

    def _flush_cache(self) -> None:
        if not self._cache_buffer:
            return
        with self._cache_path.open("a", encoding="utf-8") as handle:
            for item in self._cache_buffer:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._cache_buffer.clear()

    def _fallback_write(self, record: dict) -> None:
        with self._fallback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _adjust_cache_interval(self, now: float) -> None:
        """Adapt the cache flush cadence to the observed throughput."""

        if self._last_record_ts is not None:
            delta = max(now - self._last_record_ts, 1e-6)
            alpha = 0.2
            if self._arrival_ema is None:
                self._arrival_ema = delta
            else:
                self._arrival_ema = (1 - alpha) * self._arrival_ema + alpha * delta
        self._last_record_ts = now

        if self._arrival_ema and self._arrival_ema > 0:
            throughput = 1.0 / self._arrival_ema
            suggested = self._target_buffer / throughput
        else:
            suggested = self._max_cache_interval

        buffer_ratio = max(1.0, len(self._cache_buffer) / self._target_buffer)
        target_interval = suggested / buffer_ratio
        target_interval = max(self._min_cache_interval, min(self._max_cache_interval, target_interval))
        self._cache_interval = target_interval

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        if isinstance(value, datetime):
            return value.replace(tzinfo=value.tzinfo or timezone.utc)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
            except ValueError:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        return datetime.now(timezone.utc)


__all__ = ["SQLiteRepository"]
