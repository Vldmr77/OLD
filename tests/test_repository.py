import json
import sqlite3
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
