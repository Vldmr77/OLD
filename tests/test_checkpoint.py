from pathlib import Path

from scalp_system.storage.checkpoint import CheckpointManager


def test_checkpoint_roundtrip(tmp_path: Path):
    manager = CheckpointManager(tmp_path / "state.json")
    state = {"foo": "bar", "count": 3}
    manager.persist(state)
    loaded = manager.load()
    assert loaded == state


def test_checkpoint_handles_missing_file(tmp_path: Path):
    manager = CheckpointManager(tmp_path / "missing.json")
    assert manager.load() is None
