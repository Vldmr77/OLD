import asyncio
from datetime import datetime, timedelta

from scalp_system.config.base import DisasterRecoveryConfig
from scalp_system.storage.disaster_recovery import DisasterRecoveryManager


def test_disaster_recovery_creates_snapshot(tmp_path):
    base = tmp_path / "runtime"
    base.mkdir()
    (base / "signals.db").write_text("content", encoding="utf-8")

    config = DisasterRecoveryConfig(
        enabled=True,
        replication_path=tmp_path / "backups",
        include_patterns=["signals.db"],
        max_snapshots=3,
        interval_minutes=0,
    )
    config.ensure()

    manager = DisasterRecoveryManager(base, config)

    created = asyncio.run(manager.maybe_replicate())

    assert created is True
    snapshots = list(config.replication_path.iterdir())
    assert len(snapshots) == 1
    backup_file = next(snapshots[0].iterdir())
    assert backup_file.name == "signals.db"
    assert backup_file.read_text(encoding="utf-8") == "content"


def test_disaster_recovery_prunes_old_snapshots(tmp_path):
    base = tmp_path / "runtime"
    base.mkdir()
    (base / "signals.db").write_text("content", encoding="utf-8")

    config = DisasterRecoveryConfig(
        enabled=True,
        replication_path=tmp_path / "backups",
        include_patterns=["signals.db"],
        max_snapshots=2,
        interval_minutes=0,
    )
    config.ensure()

    timestamps = [
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 1, 1, 0, 5, 0),
        datetime(2024, 1, 1, 0, 10, 0),
    ]
    iterator = iter(timestamps)
    manager = DisasterRecoveryManager(base, config, time_provider=lambda: next(iterator))

    for _ in range(3):
        asyncio.run(manager.maybe_replicate())

    snapshots = sorted(config.replication_path.iterdir())
    assert len(snapshots) == 2
    names = [path.name for path in snapshots]
    assert names == ["20240101000500", "20240101001000"]
