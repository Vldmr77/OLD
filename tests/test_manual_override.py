from __future__ import annotations

import os
from datetime import datetime, timedelta

from scalp_system.config.base import ManualOverrideConfig
from scalp_system.control.manual_override import ManualOverrideGuard


def test_manual_override_inactive_without_flag(tmp_path):
    config = ManualOverrideConfig(enabled=True, flag_path=tmp_path / "flag")
    guard = ManualOverrideGuard(config)

    status = guard.status()

    assert status.halted is False
    assert status.reason is None


def test_manual_override_reads_reason(tmp_path):
    flag = tmp_path / "flag"
    flag.write_text("maintenance window", encoding="utf-8")
    config = ManualOverrideConfig(enabled=True, flag_path=flag)
    guard = ManualOverrideGuard(config)

    status = guard.status()

    assert status.halted is True
    assert status.reason == "maintenance window"


def test_manual_override_auto_resume(tmp_path):
    flag = tmp_path / "flag"
    flag.write_text("halt", encoding="utf-8")
    past = datetime.utcnow() - timedelta(minutes=5)
    os.utime(flag, (past.timestamp(), past.timestamp()))
    config = ManualOverrideConfig(
        enabled=True,
        flag_path=flag,
        auto_resume_minutes=1,
    )
    guard = ManualOverrideGuard(config)

    status = guard.status()

    assert status.halted is False
    assert status.reason is None
    assert not flag.exists()


def test_manual_override_poll_interval_positive(tmp_path):
    config = ManualOverrideConfig(
        enabled=True,
        flag_path=tmp_path / "flag",
        poll_interval_seconds=0,
    )
    guard = ManualOverrideGuard(config)

    assert guard.poll_interval > 0


def test_manual_override_activate_writes_flag(tmp_path):
    config = ManualOverrideConfig(enabled=True, flag_path=tmp_path / "flag")
    guard = ManualOverrideGuard(config)

    guard.activate("paused")

    status = guard.status()
    assert status.halted is True
    assert config.flag_path.read_text(encoding="utf-8").strip() == "paused"
