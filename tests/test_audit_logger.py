from pathlib import Path

from scalp_system.monitoring.audit import AuditLogger


def test_audit_logger_creates_header(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.log"
    logger = AuditLogger(log_path)
    logger.log("ORDER", "EXECUTED", "figi=TEST;confidence=0.9")
    contents = log_path.read_text(encoding="utf-8").splitlines()
    assert contents[0].startswith("#Version: 1.0")
    assert contents[1].startswith("#Fields:")
    assert any("ORDER" in line and "EXECUTED" in line for line in contents)


def test_iter_lines_reads_entries(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.log"
    logger = AuditLogger(log_path)
    logger.log("RISK", "DATA_DRIFT", "figi=XYZ")
    lines = list(logger.iter_lines())
    assert len(lines) == 3  # header + fields + record
