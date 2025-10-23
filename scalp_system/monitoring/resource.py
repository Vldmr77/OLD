"""Resource monitoring helpers."""
from __future__ import annotations

import subprocess
import time
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
    cpu_temp_c: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    network_mbps: Optional[float] = None


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
        self._net_prev_bytes: Optional[int] = None
        self._net_prev_time: Optional[float] = None

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
        cpu_temp = self._cpu_temperature()
        gpu_temp = self._gpu_temperature()
        network = self._network_mbps()
        return ResourceSnapshot(
            cpu_percent=cpu,
            memory_percent=memory,
            gpu_memory_percent=gpu,
            cpu_temp_c=cpu_temp,
            gpu_temp_c=gpu_temp,
            network_mbps=network,
        )

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

    def _gpu_temperature(self) -> Optional[float]:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=0.5,
            )
        except Exception:  # pragma: no cover - GPU sensors optional
            return None
        value = output.strip().splitlines()
        if not value:
            return None
        try:
            return float(value[0])
        except ValueError:  # pragma: no cover - unexpected output
            return None

    def _cpu_temperature(self) -> Optional[float]:
        if psutil is None or not hasattr(psutil, "sensors_temperatures"):
            return None  # pragma: no cover - optional sensors
        try:
            temps = psutil.sensors_temperatures(fahrenheit=False)
        except Exception:  # pragma: no cover - sensors optional
            return None
        if not temps:
            return None
        preferred = ("coretemp", "cpu-thermal", "cpu_thermal", "acpitz")
        readings: list[float] = []
        for key in preferred:
            entries = temps.get(key)
            if entries:
                readings = [float(t.current) for t in entries if getattr(t, "current", None) is not None]
                if readings:
                    break
        if not readings:
            readings = [
                float(entry.current)
                for sensors in temps.values()
                for entry in sensors
                if getattr(entry, "current", None) is not None
            ]
        if not readings:
            return None
        return sum(readings) / len(readings)

    def _network_mbps(self) -> Optional[float]:
        if psutil is None or not hasattr(psutil, "net_io_counters"):
            return None  # pragma: no cover - optional psutil
        counters = psutil.net_io_counters()
        total_bytes = getattr(counters, "bytes_sent", 0) + getattr(counters, "bytes_recv", 0)
        now = time.monotonic()
        if self._net_prev_bytes is None or self._net_prev_time is None:
            self._net_prev_bytes = int(total_bytes)
            self._net_prev_time = now
            return None
        delta_bytes = max(0, int(total_bytes) - self._net_prev_bytes)
        delta_time = max(1e-6, now - self._net_prev_time)
        self._net_prev_bytes = int(total_bytes)
        self._net_prev_time = now
        # Convert to megabits per second
        return (delta_bytes * 8.0) / (delta_time * 1_000_000.0)


__all__ = ["ResourceMonitor", "ResourceSnapshot"]
