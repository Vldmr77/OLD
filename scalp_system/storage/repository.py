"""Storage abstractions backed by sqlite."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List


class SQLiteRepository:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fallback_path = self._path.with_suffix(".fallback.jsonl")
        self._cache_path = self._path.with_suffix(".cache.jsonl")
        self._cache_buffer: List[dict] = []
        self._last_cache_flush = time.monotonic()
        self._cache_interval = 30.0
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

    def _record_cache(self, record: dict) -> None:
        self._cache_buffer.append(record)
        now = time.monotonic()
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


__all__ = ["SQLiteRepository"]
