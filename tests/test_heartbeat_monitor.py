import asyncio
from contextlib import suppress

from scalp_system.monitoring.heartbeat import HeartbeatMonitor


def test_heartbeat_monitor_triggers_after_misses():
    events = []

    async def runner() -> None:
        monitor = HeartbeatMonitor(
            enabled=True,
            interval_seconds=0.1,
            miss_threshold=2,
        )

        async def callback(status):
            events.append(status)

        task = asyncio.create_task(monitor.monitor(callback, poll_interval=0.05))
        await asyncio.sleep(0.12)
        monitor.record_beat("FIGI")
        await asyncio.sleep(0.35)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    asyncio.run(runner())

    assert events, "Expected heartbeat miss to be reported"
    status = events[0]
    assert status.last_context == "FIGI"
    assert status.missed_intervals >= 1
    assert status.last_beat is not None


def test_heartbeat_monitor_disabled_no_events():
    events = []

    async def runner() -> None:
        monitor = HeartbeatMonitor(
            enabled=False,
            interval_seconds=0.1,
            miss_threshold=1,
        )

        async def callback(status):
            events.append(status)

        task = asyncio.create_task(monitor.monitor(callback, poll_interval=0.05))
        await asyncio.sleep(0.12)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    asyncio.run(runner())

    assert events == []
