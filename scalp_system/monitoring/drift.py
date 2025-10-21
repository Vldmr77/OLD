"""Data drift detection utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path
from statistics import pstdev
from typing import Iterable, List, Literal


@dataclass
class DriftReport:
    p_value: float
    mean_diff: float
    sigma: float
    severity: Literal["none", "warning", "critical"]
    triggered: bool
    calibrate: bool
    alert: bool


def _ks_2samp(data1: List[float], data2: List[float]) -> tuple[float, float]:
    data1 = sorted(data1)
    data2 = sorted(data2)
    n1 = len(data1)
    n2 = len(data2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    i1 = i2 = 0
    d = 0.0
    c1 = c2 = 0.0
    while i1 < n1 and i2 < n2:
        if data1[i1] <= data2[i2]:
            c1 = (i1 + 1) / n1
            i1 += 1
        else:
            c2 = (i2 + 1) / n2
            i2 += 1
        d = max(d, abs(c1 - c2))
    en = sqrt(n1 * n2 / (n1 + n2))
    p_value = 2 * sum((-1) ** (k - 1) * (2 * en * d) ** (2 * k) for k in range(1, 5))
    p_value = max(0.0, min(1.0, 1 - p_value))
    return d, p_value


class DriftDetector:
"""Detects distribution drift and persists daily JSON metrics."""

    def __init__(self, *, threshold: float, history_dir: Path) -> None:
        self._threshold = threshold
        self._history_dir = history_dir
        self._history_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, reference: Iterable[float], current: Iterable[float]) -> DriftReport:
        ref = list(reference)
        cur = list(current)
        if not ref or not cur:
            return DriftReport(
                p_value=1.0,
                mean_diff=0.0,
                sigma=0.0,
                severity="none",
                triggered=False,
                calibrate=False,
                alert=False,
            )
        _, p_value = _ks_2samp(ref, cur)
        mean_diff = abs(sum(ref) / len(ref) - sum(cur) / len(cur))
        sigma = pstdev(ref) if len(ref) > 1 else 0.0
        two_sigma = 2 * sigma
        three_sigma = 3 * sigma
        calibrate_threshold = max(self._threshold, two_sigma)
        alert_threshold = max(self._threshold * 1.5, three_sigma)
        calibrate = p_value < 0.05 and mean_diff > calibrate_threshold
        alert = mean_diff > alert_threshold
        if alert:
            calibrate = True
        severity: Literal["none", "warning", "critical"]
        if alert:
            severity = "critical"
        elif calibrate:
            severity = "warning"
        else:
            severity = "none"
        report = DriftReport(
            p_value=p_value,
            mean_diff=mean_diff,
            sigma=sigma,
            severity=severity,
            triggered=severity != "none",
            calibrate=calibrate,
            alert=alert,
        )
        self._log_metrics(report)
        return report

    def _log_metrics(self, report: DriftReport) -> None:
        filename = f"drift_metrics_{datetime.utcnow():%Y%m%d}.jsonl"
        path = self._history_dir / filename
        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "p_value": report.p_value,
            "mean_diff": report.mean_diff,
            "sigma": report.sigma,
            "severity": report.severity,
            "calibrate": report.calibrate,
            "alert": report.alert,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["DriftDetector", "DriftReport"]
