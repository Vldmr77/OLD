from scalp_system.ml.calibration import CalibrationCoordinator


def test_calibration_coordinator_dedup(tmp_path):
    queue = tmp_path / "calibration.jsonl"
    coordinator = CalibrationCoordinator(queue_path=queue, dedupe_ttl_seconds=3600)
    first = coordinator.enqueue(reason="drift", metadata={"figi": "FIGI"}, dedupe_key="drift:FIGI")
    second = coordinator.enqueue(reason="drift", metadata={"figi": "FIGI"}, dedupe_key="drift:FIGI")
    assert first is True
    assert second is False
    contents = queue.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
