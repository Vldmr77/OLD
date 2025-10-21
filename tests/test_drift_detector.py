from pathlib import Path

from scalp_system.monitoring.drift import DriftDetector


def test_drift_detector_logs(tmp_path: Path):
    detector = DriftDetector(threshold=0.01, history_path=tmp_path / "drift.log")
    report = detector.evaluate([1, 1, 1], [1.1, 1.1, 1.1])
    assert report.mean_diff >= 0
    assert (tmp_path / "drift.log").exists()
