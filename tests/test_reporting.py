import asyncio
import json
from datetime import datetime, timedelta, timezone

from scalp_system.monitoring.reporting import PerformanceReporter
from scalp_system.storage.repository import SQLiteRepository


class DummyRiskEngine:
    def __init__(self, realized_pnl: float = 0.0, halted: bool = False) -> None:
        self.realized_pnl = realized_pnl
        self.halted = halted
        self.losses = 0

    def performance_snapshot(self) -> dict:
        return {
            "realized_pnl": self.realized_pnl,
            "consecutive_losses": self.losses,
            "daily_loss_triggered": False,
            "halt_trading": self.halted,
        }


class StubNotifications:
    def __init__(self) -> None:
        self.calls = []

    async def notify_performance_summary(self, **payload) -> bool:
        self.calls.append(payload)
        return True


def test_performance_reporter_writes_report(tmp_path):
    repo = SQLiteRepository(tmp_path / "signals.db")
    repo.persist_signal("AAA", 1, 0.9)
    repo.persist_signal("BBB", -1, 0.4)
    notifications = StubNotifications()
    reporter = PerformanceReporter(
        repository=repo,
        risk_engine=DummyRiskEngine(realized_pnl=123.45),
        report_path=tmp_path / "reports" / "perf.jsonl",
        interval_minutes=1,
        max_history_days=1,
        notifications=notifications,
    )
    now = datetime.now(timezone.utc)
    report = asyncio.run(reporter.maybe_emit(now))

    assert report is not None
    assert report["summary"]["signal_count"] == 2
    assert notifications.calls
    payload = notifications.calls[0]
    assert payload["realized_pnl"] == 123.45
    assert payload["halted"] is False

    # second call within interval should not emit
    assert asyncio.run(reporter.maybe_emit(now)) is None

    saved = (tmp_path / "reports" / "perf.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(saved) == 1


def test_performance_reporter_prunes_history(tmp_path):
    repo = SQLiteRepository(tmp_path / "signals.db")
    notifications = StubNotifications()
    reporter = PerformanceReporter(
        repository=repo,
        risk_engine=DummyRiskEngine(),
        report_path=tmp_path / "reports" / "perf.jsonl",
        interval_minutes=1,
        max_history_days=1,
        notifications=notifications,
    )

    old_entry = {
        "summary": {
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
            "signal_count": 0,
            "avg_confidence": 0.0,
            "realized_pnl": 0.0,
            "consecutive_losses": 0,
            "daily_loss_triggered": False,
            "halted": False,
        },
        "per_instrument": {},
    }
    reporter._report_path.parent.mkdir(parents=True, exist_ok=True)
    reporter._report_path.write_text(json.dumps(old_entry) + "\n", encoding="utf-8")

    repo.persist_signal("AAA", 1, 0.5)
    asyncio.run(reporter.maybe_emit(datetime.now(timezone.utc)))

    saved = reporter._report_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(saved) == 1
    latest = json.loads(saved[0])
    assert latest["summary"]["signal_count"] == 1
