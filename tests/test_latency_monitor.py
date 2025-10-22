from scalp_system.monitoring.latency import LatencyMonitor


def test_latency_monitor_emits_warn_and_critical():
    monitor = LatencyMonitor({"features": 10.0}, violation_limit=2)

    first = monitor.observe("features", 11.0)
    assert first is not None
    assert first.severity == "warn"

    second = monitor.observe("features", 12.5)
    assert second is not None
    assert second.severity == "critical"
    assert second.consecutive == 2

    reset = monitor.observe("features", 9.0)
    assert reset is None


def test_latency_monitor_ignores_unknown_stage():
    monitor = LatencyMonitor({"ml": 20.0})
    assert monitor.observe("features", 30.0) is None
