"""NVML device power, memory, temperature, clocks, and throttle reasons.

The management library is loaded with ctypes when a benchmark probes this
source. ``pynvml`` is not a dependency. A missing library, a failed init, or
a machine with no device leaves the source unavailable and does not invent
a reading. Tests pass a backend and never load the library.

Only the first device is read. Power is that device's draw. Throttle flags
are the slowdown and power-cap bits. An idle clock is not reported as
throttling.
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable, Mapping
from typing import Protocol

from ohho.bench.sensors.base import (
    DOMAIN_DEVICE,
    SensorSample,
    SensorSource,
    SourceStatus,
)

_GRAPHICS = 0
_SM = 1
_MEMORY = 2
_VIDEO = 3
_TEMP_GPU = 0
_SUCCESS = 0

# Slowdown and power-cap bits from nvmlClocksThrottleReason*.
_THROTTLE_BITS: tuple[tuple[int, str], ...] = (
    (0x0000000000000004, "sw power cap"),
    (0x0000000000000008, "hw slowdown"),
    (0x0000000000000020, "sw thermal slowdown"),
    (0x0000000000000040, "hw thermal slowdown"),
    (0x0000000000000080, "hw power brake"),
)

_missing: OSError | None = None
_library: ctypes.CDLL | None = None


class NvmlBackend(Protocol):
    """The handful of reads the source needs. Numbers are already in SI units."""

    def power_w(self) -> float | None:
        """Device power in watts, or None when the call fails."""

    def memory_used_mb(self) -> float | None:
        """Used device memory in mebibytes, or None."""

    def temperature_c(self) -> float | None:
        """Device temperature in Celsius, or None."""

    def clocks_mhz(self) -> Mapping[str, float]:
        """Named clocks in MHz. An empty mapping means none were read."""

    def throttled(self) -> tuple[bool, str | None]:
        """Whether a slowdown bit is set, and a short detail."""

    def close(self) -> None:
        """Release the library session."""


class NvmlSource(SensorSource):
    """First-device NVML source. Pass ``backend`` to avoid loading a library."""

    source_id = "nvml"

    def __init__(
        self,
        backend: NvmlBackend | None = None,
        loader: Callable[[], NvmlBackend] | None = None,
    ) -> None:
        self._backend = backend
        self._loader = loader or load_nvml

    def probe(self) -> SourceStatus:
        if self._backend is None:
            try:
                self._backend = self._loader()
            except Exception as exc:
                text = str(exc).strip() or "NVML library not found"
                return SourceStatus(self.source_id, False, reason=text)
        return SourceStatus(
            self.source_id,
            True,
            power_domain=DOMAIN_DEVICE,
            observes_throttle=True,
        )

    def read(self) -> SensorSample:
        backend = self._backend
        if backend is None:
            return SensorSample(self.source_id)
        try:
            power = _nonnegative(backend.power_w())
            memory = _nonnegative(backend.memory_used_mb())
            temperature = _finite(backend.temperature_c())
            clocks = _clocks(backend.clocks_mhz())
            throttled, detail = backend.throttled()
        except Exception:
            return SensorSample(self.source_id)
        text = detail.strip() if isinstance(detail, str) else None
        return SensorSample(
            self.source_id,
            power_w=power,
            device_memory_mb=memory,
            temperature_c=temperature,
            throttled=bool(throttled),
            throttle_detail=text or None,
            clocks_mhz=clocks,
        )

    def close(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is None:
            return
        try:
            backend.close()
        except Exception:
            return


def load_nvml() -> NvmlBackend:
    """Open the NVML shared library and initialise it.

    A missing library is remembered for this process so later probes do not
    search again. A library that opens and then fails init is not cached.
    """
    if _missing is not None:
        raise _missing
    return _CtypesNvml(_open_library())


def _library_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("nvml.dll",)
    if sys.platform == "darwin":
        return ("libnvidia-ml.dylib",)
    return ("libnvidia-ml.so.1", "libnvidia-ml.so")


def _open_library() -> ctypes.CDLL:
    global _library, _missing
    if _library is not None:
        return _library
    if _missing is not None:
        raise _missing
    for name in _library_names():
        try:
            _library = ctypes.CDLL(name)
            return _library
        except OSError:
            continue
    _missing = OSError("NVML library not found")
    raise _missing


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _CtypesNvml:
    """ctypes session for device 0. Field failures become None, not guesses."""

    def __init__(self, library: ctypes.CDLL) -> None:
        self._lib = library
        self._ready = False
        self._handle = ctypes.c_void_p()
        self._bind()
        rc = self._init()
        if rc != _SUCCESS:
            raise OSError(f"NVML init failed ({rc})")
        self._ready = True
        count = ctypes.c_uint()
        if self._lib.nvmlDeviceGetCount(ctypes.byref(count)) != _SUCCESS:
            self.close()
            raise OSError("NVML device count failed")
        if count.value < 1:
            self.close()
            raise OSError("NVML reported no devices")
        if (
            self._lib.nvmlDeviceGetHandleByIndex(0, ctypes.byref(self._handle))
            != _SUCCESS
        ):
            self.close()
            raise OSError("NVML device handle failed")

    def power_w(self) -> float | None:
        if not self._ready:
            return None
        milli = ctypes.c_uint()
        if self._lib.nvmlDeviceGetPowerUsage(self._handle, ctypes.byref(milli)) != 0:
            return None
        return milli.value / 1000.0

    def memory_used_mb(self) -> float | None:
        if not self._ready:
            return None
        info = _NvmlMemory()
        if self._lib.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(info)) != 0:
            return None
        return info.used / (1024.0 * 1024.0)

    def temperature_c(self) -> float | None:
        if not self._ready:
            return None
        temp = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetTemperature(
            self._handle, _TEMP_GPU, ctypes.byref(temp)
        )
        if rc != _SUCCESS:
            return None
        return float(temp.value)

    def clocks_mhz(self) -> dict[str, float]:
        if not self._ready:
            return {}
        clocks: dict[str, float] = {}
        for name, kind in (
            ("graphics", _GRAPHICS),
            ("sm", _SM),
            ("memory", _MEMORY),
            ("video", _VIDEO),
        ):
            value = ctypes.c_uint()
            rc = self._lib.nvmlDeviceGetClockInfo(
                self._handle, kind, ctypes.byref(value)
            )
            if rc == _SUCCESS:
                clocks[name] = float(value.value)
        return clocks

    def throttled(self) -> tuple[bool, str | None]:
        if not self._ready:
            return False, None
        reasons = ctypes.c_ulonglong()
        rc = self._lib.nvmlDeviceGetCurrentClocksThrottleReasons(
            self._handle, ctypes.byref(reasons)
        )
        if rc != _SUCCESS:
            return False, None
        labels = [label for bit, label in _THROTTLE_BITS if reasons.value & bit]
        if not labels:
            return False, None
        return True, ", ".join(labels)

    def close(self) -> None:
        if not self._ready:
            return
        self._ready = False
        try:
            self._lib.nvmlShutdown()
        except Exception:
            return

    def _bind(self) -> None:
        lib = self._lib
        self._init = getattr(lib, "nvmlInit_v2", None) or getattr(lib, "nvmlInit", None)
        if self._init is None:
            raise OSError("NVML library has no init function")
        self._init.restype = ctypes.c_int
        shutdown = getattr(lib, "nvmlShutdown", None)
        if shutdown is None:
            raise OSError("NVML library has no shutdown function")
        shutdown.restype = ctypes.c_int
        self._bind_call(
            "nvmlDeviceGetCount", ctypes.c_int, ctypes.POINTER(ctypes.c_uint)
        )
        self._bind_call(
            "nvmlDeviceGetHandleByIndex",
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self._bind_call(
            "nvmlDeviceGetPowerUsage",
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        )
        self._bind_call(
            "nvmlDeviceGetMemoryInfo",
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(_NvmlMemory),
        )
        self._bind_call(
            "nvmlDeviceGetTemperature",
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint),
        )
        self._bind_call(
            "nvmlDeviceGetClockInfo",
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint),
        )
        self._bind_call(
            "nvmlDeviceGetCurrentClocksThrottleReasons",
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulonglong),
        )

    def _bind_call(self, name: str, restype: object, *argtypes: object) -> None:
        function = getattr(self._lib, name, None)
        if function is None:
            raise OSError(f"NVML library is missing {name}")
        function.restype = restype
        function.argtypes = argtypes


def _nonnegative(value: object) -> float | None:
    number = _finite(value)
    if number is None or number < 0:
        return None
    return number


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def source() -> NvmlSource:
    """Adapter entry used by discovery."""
    return NvmlSource()


def _clocks(values: object) -> dict[str, float]:
    if not isinstance(values, Mapping):
        return {}
    clocks: dict[str, float] = {}
    for name, mhz in values.items():
        if not isinstance(name, str) or not name.strip():
            continue
        number = _nonnegative(mhz)
        if number is None:
            continue
        clocks[name] = number
    return clocks
