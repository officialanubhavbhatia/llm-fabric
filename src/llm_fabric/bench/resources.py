"""Host resource samples for a benchmark report.

CPU and RSS come from this process. GPU is probed and left unavailable when
nothing is using one — which is every run against the mock provider. Queue
depth is observed only when a `HealthTracker` is handed in; the HTTP harness
does not invent one from connection count.
"""

from __future__ import annotations

import resource
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from llm_fabric.router.health import HealthTracker


@dataclass(frozen=True, slots=True)
class ResourceSample:
    cpu_user_s: float
    cpu_system_s: float
    rss_bytes: int | None
    gpu: dict[str, Any] | None
    queue_depth: int | None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cpu_user_s": round(self.cpu_user_s, 4),
            "cpu_system_s": round(self.cpu_system_s, 4),
            "rss_bytes": self.rss_bytes,
            "gpu": self.gpu,
            "queue_depth": self.queue_depth,
            "note": self.note,
        }


def sample_process(*, health: HealthTracker | None = None) -> ResourceSample:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    rss = _rss_bytes(usage.ru_maxrss)
    queue: int | None = None
    if health is not None:
        queue = sum(snap.queue_depth for snap in health.all_snapshots().values())
    return ResourceSample(
        cpu_user_s=usage.ru_utime + children.ru_utime,
        cpu_system_s=usage.ru_stime + children.ru_stime,
        rss_bytes=rss,
        gpu=probe_gpu(),
        queue_depth=queue,
        note=(
            "CPU is this process plus waited-for children. RSS is this "
            "process only. GPU is unavailable unless a probe finds an active "
            "device. Queue depth is HealthTracker in-flight counts when a "
            "tracker is provided, otherwise absent."
        ),
    )


def _rss_bytes(raw: int) -> int | None:
    """`ru_maxrss` is kilobytes on Linux and bytes on macOS."""
    if raw <= 0:
        return None
    import sys

    if sys.platform == "darwin":
        return raw
    return raw * 1024


def probe_gpu() -> dict[str, Any] | None:
    """Return a GPU reading, or `None` when no device is observable.

    `nvidia-smi` is the only probe. Apple Silicon has a GPU; this process does
    not use it for inference, and claiming Metal utilisation would be a guess.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [
                binary,
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    line = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        return {
            "name": parts[0],
            "utilization_percent": float(parts[1]),
            "memory_used_mib": float(parts[2]),
            "memory_total_mib": float(parts[3]),
        }
    except ValueError:
        return None


def elapsed_cpu(before: ResourceSample, after: ResourceSample) -> dict[str, float]:
    wall_cpu = (after.cpu_user_s - before.cpu_user_s) + (after.cpu_system_s - before.cpu_system_s)
    return {
        "cpu_user_s": round(after.cpu_user_s - before.cpu_user_s, 4),
        "cpu_system_s": round(after.cpu_system_s - before.cpu_system_s, 4),
        "cpu_s": round(wall_cpu, 4),
    }
