from scalp_system.monitoring.resource import ResourceMonitor, ResourceSnapshot


def test_resource_monitor_recommends_actions():
    monitor = ResourceMonitor(
        cpu_threshold=80.0,
        memory_threshold=75.0,
        gpu_threshold=85.0,
        sampler=lambda: ResourceSnapshot(cpu_percent=95.0, memory_percent=80.0, gpu_memory_percent=90.0),
    )
    snapshot, actions = monitor.check_thresholds()
    assert snapshot.cpu_percent == 95.0
    assert set(actions) == {"disable_gui", "trim_caches", "downgrade_lstm"}


def test_resource_monitor_handles_missing_gpu():
    monitor = ResourceMonitor(
        cpu_threshold=90.0,
        memory_threshold=90.0,
        gpu_threshold=90.0,
        sampler=lambda: ResourceSnapshot(cpu_percent=10.0, memory_percent=20.0, gpu_memory_percent=None),
    )
    snapshot, actions = monitor.check_thresholds()
    assert actions == []
    assert snapshot.gpu_memory_percent is None
