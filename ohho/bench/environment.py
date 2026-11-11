"""Host, versions, and optional GPU probe for an ohho.bench/v1 result.

Imports of torch, onnxruntime, numpy, and tensorrt are optional. A missing
module omits that version. ``nvidia-smi`` is optional. GPU utilisation and
power are null when the probe does not return a number. The typed gauges
``accelerator_util_mean`` and ``power_w_mean`` stay absent in that case,
because ohho.bench/v1 defines them as non-null numbers.
"""

from __future__ import annotations

import importlib
import platform
import re
import socket
import subprocess
import sys
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ohho.targets.schema import TargetProfile

_SHA = re.compile(r"[0-9a-fA-F]{7,40}\Z")
_MISSING = {"", "n/a", "[n/a]", "not supported"}


@dataclass(frozen=True)
class GpuProbe:
    """One sample from ``nvidia-smi``, or all nulls when it is unavailable."""

    util_percent: float | None = None
    power_w: float | None = None
    driver: str | None = None


@dataclass
class MemorySnapshot:
    """Peak memory taken at the end of a run."""

    rss_mb: float | None = None
    python_alloc_mb: float | None = None

    def as_telemetry(self) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if self.rss_mb is not None:
            fields["memory_mb_max"] = self.rss_mb
            fields["memory_source"] = "process_peak_rss"
            fields["peak_memory_mb"] = self.rss_mb
        elif self.python_alloc_mb is not None:
            fields["memory_mb_max"] = self.python_alloc_mb
            fields["memory_source"] = "tracemalloc"
            fields["peak_memory_mb"] = self.python_alloc_mb
        else:
            fields["peak_memory_mb"] = None
        if self.python_alloc_mb is not None:
            fields["python_alloc_mb_max"] = self.python_alloc_mb
        return fields


class MemoryTrace:
    """Tracemalloc plus the process peak RSS.

    ``start`` is a no-op when the process is already tracing. ``close`` stops
    tracing only if this object started it. ``snapshot`` can be called before
    ``close``.
    """

    def __init__(self) -> None:
        self._started = False

    def start(self) -> None:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started = True

    def snapshot(self) -> MemorySnapshot:
        python_alloc = None
        if tracemalloc.is_tracing():
            _current, peak = tracemalloc.get_traced_memory()
            python_alloc = peak / (1024 * 1024)
        return MemorySnapshot(rss_mb=process_peak_mb(), python_alloc_mb=python_alloc)

    def close(self) -> None:
        if self._started and tracemalloc.is_tracing():
            tracemalloc.stop()
        self._started = False


def process_peak_mb() -> float | None:
    """Peak resident memory of this process, in MiB, or None if it cannot be read."""
    try:
        if sys.platform == "win32":
            return _windows_peak_mb()
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return peak / (1024 * 1024)
        return peak / 1024.0
    except Exception:
        return None


def _windows_peak_mb() -> float | None:
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return None
    return counters.PeakWorkingSetSize / (1024 * 1024)


def host_block(profile: TargetProfile) -> dict[str, str]:
    """Host identity for ``run.host``. Empty probes become ``unknown``."""
    cpu = platform.processor().strip() or platform.machine().strip() or "unknown"
    return {
        "name": socket.gethostname().strip() or "unknown",
        "platform": platform.system().strip().lower() or "unknown",
        "arch": platform.machine().strip() or "unknown",
        "cpu": cpu,
        "accelerator": profile.accelerator or "none",
    }


def git_sha(root: Path | None = None) -> str:
    """HEAD commit, or ``unknown`` when git is missing or this tree is not a repo."""
    repo = root if root is not None else _repo_root()
    if repo is None:
        return "unknown"
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    sha = completed.stdout.strip()
    if _SHA.fullmatch(sha):
        return sha.lower()
    return "unknown"


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return None


def probe_nvidia() -> GpuProbe:
    """Read the first GPU from ``nvidia-smi``. Any failure is an empty probe."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return GpuProbe()
    if completed.returncode != 0:
        return GpuProbe()
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return GpuProbe()
    parts = [part.strip() for part in lines[0].split(",")]
    driver = _text(parts[0]) if parts else None
    util = _number(parts[1]) if len(parts) > 1 else None
    power = _number(parts[2]) if len(parts) > 2 else None
    return GpuProbe(util_percent=util, power_w=power, driver=driver)


def _text(value: str) -> str | None:
    if value.strip().lower() in _MISSING:
        return None
    return value.strip()


def _number(value: str) -> float | None:
    if value.strip().lower() in _MISSING:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if number < 0:
        return None
    return number


def version_map() -> dict[str, str]:
    """python and ohho, plus optional component versions that import cleanly."""
    import ohho

    versions: dict[str, str] = {
        "python": platform.python_version(),
        "ohho": ohho.__version__,
    }
    for name in ("onnxruntime", "numpy", "torch", "tensorrt"):
        found = _module_version(name)
        if found is not None:
            versions[name] = found
    cuda = _cuda_version()
    if cuda is not None:
        versions["cuda"] = cuda
    return versions


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    value = getattr(module, "__version__", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _cuda_version() -> str | None:
    try:
        import torch
    except Exception:
        return None
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    if isinstance(cuda, str) and cuda.strip():
        return cuda.strip()
    return None


def collect_environment(
    profile: TargetProfile,
    *,
    timing_source: str,
    execution_provider: str,
    execution_providers: tuple[str, ...] = (),
    timing_note: str = "",
    gpu: GpuProbe | None = None,
) -> tuple[dict[str, Any], GpuProbe]:
    """Build ``environment`` and return the GPU probe used for telemetry."""
    probed = gpu if gpu is not None else probe_nvidia()
    versions = version_map()
    if probed.driver is not None:
        versions["nvidia_driver"] = probed.driver
    environment: dict[str, Any] = {
        "versions": versions,
        "target_profile": profile.id,
        "target_summary": profile.summary,
        "git_sha": git_sha(),
        "timing_source": timing_source,
        "execution_provider": execution_provider,
    }
    if execution_providers:
        environment["execution_providers"] = list(execution_providers)
    if timing_note:
        environment["timing_note"] = timing_note
    if probed.driver is not None:
        environment["driver"] = probed.driver
    return environment, probed


def gpu_telemetry(gpu: GpuProbe) -> dict[str, Any]:
    """Utilisation and power. Null when the probe has no number.

    Typed ohho.bench/v1 gauges are set only for real numbers. Changing those
    gauges to accept null would be an ohho.bench/v2 change.
    """
    fields: dict[str, Any] = {
        "gpu_util_percent": gpu.util_percent,
        "gpu_power_w": gpu.power_w,
    }
    if gpu.util_percent is not None:
        fields["accelerator_util_mean"] = gpu.util_percent
    if gpu.power_w is not None:
        fields["power_w_mean"] = gpu.power_w
    return fields
