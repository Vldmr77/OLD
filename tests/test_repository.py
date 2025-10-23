import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scalp_system.storage.repository import SQLiteRepository


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_repository_caches_state(tmp_path, monkeypatch):
    repo = SQLiteRepository(tmp_path / "signals.db")
    repo._last_cache_flush -= 31  # force immediate flush
    repo.persist_signal("FIGI", 1, 0.9)
    cache_path = tmp_path / "signals.cache.jsonl"
    entries = read_jsonl(cache_path)
    assert entries
    assert entries[0]["figi"] == "FIGI"


def test_repository_fallback_on_error(tmp_path, monkeypatch):
    repo = SQLiteRepository(tmp_path / "signals.db")

    def broken_connect(*args, **kwargs):
        raise sqlite3.Error("boom")

    monkeypatch.setattr(sqlite3, "connect", broken_connect)
    repo.persist_signal("FIGI", -1, 0.5)
    fallback_path = tmp_path / "signals.fallback.jsonl"
    entries = read_jsonl(fallback_path)
    assert entries
    assert entries[0]["direction"] == -1


def test_repository_dynamic_interval_adapts(tmp_path, monkeypatch):
    clock = {"value": 0.0}

    def fake_monotonic() -> float:
        clock["value"] += 0.1
        return clock["value"]

    monkeypatch.setattr("scalp_system.storage.repository.time.monotonic", fake_monotonic)
    repo = SQLiteRepository(
        tmp_path / "signals.db",
        cache_min_interval=1.0,
        cache_max_interval=5.0,
        cache_target_buffer=5,
    )
    initial_interval = repo._cache_interval
    for index in range(15):
        repo.persist_signal(f"FIGI{index}", 1, 0.5)
    assert repo._cache_interval <= initial_interval
    assert repo._cache_interval >= repo._min_cache_interval
    assert repo._cache_interval <= repo._max_cache_interval


def test_fetch_signals_returns_recent(tmp_path):
    repo = SQLiteRepository(tmp_path / "signals.db")
    repo.persist_signal("AAA", 1, 0.9)
    repo.persist_signal("BBB", -1, 0.4)
    records = repo.fetch_signals()
    assert len(records) == 2
    assert records[0]["timestamp"].tzinfo is timezone.utc
    assert {record["figi"] for record in records} == {"AAA", "BBB"}


def test_fetch_signals_filters_by_since(tmp_path):
    repo = SQLiteRepository(tmp_path / "signals.db")
    repo.persist_signal("AAA", 1, 0.9)
    with sqlite3.connect(tmp_path / "signals.db") as conn:
        conn.execute("UPDATE signals SET created_at = datetime('now', '-2 day') WHERE figi = ?", ("AAA",))
        conn.commit()
    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    repo.persist_signal("BBB", -1, 0.4)
    recent = repo.fetch_signals(since=cutoff)
    assert len(recent) == 1
    assert recent[0]["figi"] == "BBB"
