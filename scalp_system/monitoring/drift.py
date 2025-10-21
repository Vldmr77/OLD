"""Data drift detection utilities."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable, List


@dataclass
class DriftReport:
    p_value: float
    mean_diff: float
    triggered: bool


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
    def __init__(self, *, threshold: float, history_path: Path) -> None:
        self._threshold = threshold
        self._history_path = history_path
        self._history_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(self, reference: Iterable[float], current: Iterable[float]) -> DriftReport:
        ref = list(reference)
        cur = list(current)
        if not ref or not cur:
            return DriftReport(p_value=1.0, mean_diff=0.0, triggered=False)
        _, p_value = _ks_2samp(ref, cur)
        mean_diff = abs(sum(ref) / len(ref) - sum(cur) / len(cur))
        triggered = p_value < 0.05 and mean_diff > self._threshold
        self._log_metrics(p_value, mean_diff)
        return DriftReport(p_value=p_value, mean_diff=mean_diff, triggered=triggered)

    def _log_metrics(self, p_value: float, mean_diff: float) -> None:
        with self._history_path.open("a", encoding="utf-8") as f:
            f.write(f"p_value={p_value:.5f},mean_diff={mean_diff:.5f}\n")


__all__ = ["DriftDetector", "DriftReport"]
