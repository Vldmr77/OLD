from pathlib import Path

from scalp_system.monitoring.drift import DriftDetector


def test_drift_detector_logs_json(tmp_path: Path):
    detector = DriftDetector(threshold=0.01, history_dir=tmp_path)
    report = detector.evaluate([1, 1, 1], [1.5, 1.5, 1.5])
    assert report.calibrate is True
    log_files = list(tmp_path.glob("drift_metrics_*.jsonl"))
    assert len(log_files) == 1
    content = log_files[0].read_text(encoding="utf-8").strip()
    assert "\"calibrate\": true" in content
