"""Storage abstractions backed by sqlite."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class SQLiteRepository:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
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
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO signals(figi, direction, confidence) VALUES (?, ?, ?)",
                (figi, direction, confidence),
            )
            conn.commit()


__all__ = ["SQLiteRepository"]
