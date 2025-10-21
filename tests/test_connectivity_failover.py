from scalp_system.config.base import ConnectivityConfig
from scalp_system.network import ConnectivityFailover


def test_failover_switches_and_recovers():
    config = ConnectivityConfig(
        failure_threshold=1,
        recovery_message_count=2,
        failover_latency_ms=150,
    )
    failover = ConnectivityFailover(config)

    event, delay = failover.record_failure("timeout")
    assert event is not None
    assert event.kind == "failover"
    assert event.channel == config.backup_label
    assert delay == config.failover_latency_ms / 1000.0

    assert failover.active_channel == config.backup_label
    assert failover.record_success() is None
    recovery = failover.record_success()
    assert recovery is not None
    assert recovery.kind == "recovery"
    assert failover.active_channel == config.primary_label

    snapshot = failover.snapshot()
    assert snapshot["active_channel"] == config.primary_label
