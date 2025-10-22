"""Disaster recovery helpers for runtime artefacts."""
from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from ..config.base import DisasterRecoveryConfig
from ..monitoring.notifications import NotificationDispatcher

LOGGER = logging.getLogger(__name__)


class DisasterRecoveryManager:
    """Creates periodic snapshots of runtime artefacts for failover."""

    def __init__(
        self,
        base_path: Path,
        config: DisasterRecoveryConfig,
        notifications: NotificationDispatcher | None = None,
        time_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._base_path = base_path
        self._config = config
        self._notifications = notifications
        self._now = time_provider or datetime.utcnow
        self._last_run: Optional[datetime] = None

    async def maybe_replicate(self) -> bool:
        """Replicate state if the interval has elapsed and files exist."""

        if not self._config.enabled:
            return False
        now = self._now()
        if self._should_skip(now):
            return False
        snapshot_dir = await asyncio.to_thread(self._create_snapshot, now)
        if snapshot_dir is None:
            return False
        self._last_run = now
        await self._notify(snapshot_dir)
        return True

    def _should_skip(self, now: datetime) -> bool:
        if self._last_run is None:
            return False
        if self._config.interval_minutes <= 0:
            return False
        return now - self._last_run < timedelta(minutes=self._config.interval_minutes)

    def _create_snapshot(self, now: datetime) -> Optional[Path]:
        timestamp = now.strftime("%Y%m%d%H%M%S")
        target_dir = self._config.replication_path / timestamp
        if target_dir.exists():
            target_dir = self._config.replication_path / f"{timestamp}_{now.microsecond}"
        target_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for pattern in self._config.include_patterns:
            copied += self._copy_pattern(pattern, target_dir)
        if copied == 0:
            shutil.rmtree(target_dir, ignore_errors=True)
            return None
        self._prune_old_snapshots()
        return target_dir

    def _copy_pattern(self, pattern: str, target_dir: Path) -> int:
        count = 0
        for source in self._base_path.glob(pattern):
            if source.is_file():
                shutil.copy2(source, target_dir / source.name)
                count += 1
        return count

    def _prune_old_snapshots(self) -> None:
        snapshots = sorted(
            (path for path in self._config.replication_path.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        excess = len(snapshots) - self._config.max_snapshots
        for index in range(max(0, excess)):
            shutil.rmtree(snapshots[index], ignore_errors=True)

    async def _notify(self, snapshot_dir: Path) -> None:
        if not self._notifications:
            return
        size_bytes = await asyncio.to_thread(self._calculate_size, snapshot_dir)
        try:
            await self._notifications.notify_backup_created(snapshot_dir, size_bytes)
        except Exception:  # pragma: no cover - defensive fallback
            LOGGER.exception("Failed to send disaster recovery notification")

    def _calculate_size(self, snapshot_dir: Path) -> int:
        size = 0
        for path in snapshot_dir.rglob("*"):
            if path.is_file():
                size += path.stat().st_size
        return size


__all__ = ["DisasterRecoveryManager"]
