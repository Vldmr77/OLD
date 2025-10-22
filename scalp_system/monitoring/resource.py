"""Resource monitoring helpers."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

try:  # pragma: no cover - psutil may be missing in minimal envs
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


@dataclass
class ResourceSnapshot:
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: Optional[float] = None


class ResourceMonitor:
    """Monitor system resource usage and recommend mitigations."""

    def __init__(
        self,
        *,
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        gpu_threshold: float = 90.0,
        sampler: Optional[Callable[[], ResourceSnapshot]] = None,
    ) -> None:
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.gpu_threshold = gpu_threshold
        self._sampler = sampler or self._sample

    def snapshot(self) -> ResourceSnapshot:
        return self._sampler()

    def check_thresholds(self) -> Tuple[ResourceSnapshot, List[str]]:
        snapshot = self.snapshot()
        actions: List[str] = []
        if snapshot.cpu_percent >= self.cpu_threshold:
            actions.append("disable_gui")
        if snapshot.memory_percent >= self.memory_threshold:
            actions.append("trim_caches")
        if snapshot.gpu_memory_percent is not None and snapshot.gpu_memory_percent >= self.gpu_threshold:
            actions.append("downgrade_lstm")
        return snapshot, actions

    def _sample(self) -> ResourceSnapshot:
        cpu = self._cpu_percent()
        memory = self._memory_percent()
        gpu = self._gpu_memory_percent()
        return ResourceSnapshot(cpu_percent=cpu, memory_percent=memory, gpu_memory_percent=gpu)

    def _cpu_percent(self) -> float:
        if psutil is None:  # pragma: no cover - fallback path
            return 0.0
        return float(psutil.cpu_percent(interval=0.0))

    def _memory_percent(self) -> float:
        if psutil is None:  # pragma: no cover - fallback path
            return 0.0
        return float(psutil.virtual_memory().percent)

    def _gpu_memory_percent(self) -> Optional[float]:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=0.5,
            )
        except Exception:  # pragma: no cover - GPUs are optional
            return None
        parts = output.strip().splitlines()
        if not parts:
            return None
        used_str, total_str = parts[0].split(",")
        try:
            used = float(used_str.strip())
            total = float(total_str.strip())
        except ValueError:  # pragma: no cover - unexpected output
            return None
        if total == 0:
            return None
        return (used / total) * 100.0


__all__ = ["ResourceMonitor", "ResourceSnapshot"]
