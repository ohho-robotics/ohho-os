"""Board telemetry from tegrastats text, with jtop as a fallback reader.

``parse_tegrastats_line`` is the pure parser. Power is the instantaneous
input rail (``VDD_IN``, otherwise ``POM_5V_IN``) in watts. Other rails are
not summed, because they sit inside that input. Host memory is the RAM used
figure. Throttling is true when the line contains the word ``throttled`` or a
``name curMHz/maxMHz`` pair whose current clock is below the max. The live
reader appends ``THROTTLED`` when a throttle-status file says so.

A missing ``tegrastats`` binary and a missing jtop install leave the source
unavailable. No power value is filled in from another rail.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ohho.bench.sensors.base import (
    DOMAIN_BOARD,
    SensorSample,
    SensorSource,
    SourceStatus,
)

_POWER_MW = re.compile(r"\b([A-Z][A-Z0-9_]*)\s+(\d+(?:\.\d+)?)mW/(\d+(?:\.\d+)?)mW")
_POWER_LEGACY = re.compile(
    r"\b(POM_[A-Z0-9_]+|VDD_[A-Z0-9_]+)\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)(?!mW)"
)
_RAM = re.compile(r"\bRAM\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)MB\b")
_TEMP = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)@(-?\d+(?:\.\d+)?)C\b")
_CPU = re.compile(r"\bCPU\s+\[([^\]]*)\]")
_CPU_CORE = re.compile(r"(\d+(?:\.\d+)?)%@(\d+(?:\.\d+)?)")
_FREQ = re.compile(r"\b([A-Z][A-Z0-9_]*)_FREQ\s+(\d+(?:\.\d+)?)%(?:@(\d+(?:\.\d+)?))?")
_THROTTLED = re.compile(r"(?i)\bthrottled\b")
_CLOCK_CAP = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)\s+(\d+(?:\.\d+)?)MHz/(\d+(?:\.\d+)?)MHz\b"
)
_INPUT_RAILS = ("VDD_IN", "POM_5V_IN")
_THROTTLE_FILES = (
    Path("/sys/kernel/debug/bpmp/debug/soctherm/throttle/status"),
    Path("/sys/devices/platform/tegra-throttle/throttle"),
)
_QUIET_THROTTLE = frozenset({"", "0", "ok", "none", "disabled", "false"})


def parse_tegrastats_line(line: str) -> SensorSample:
    """Parse one tegrastats line into power, memory, temperature, and clocks.

    ``THROTTLED`` on the line, or a current clock below its paired max, sets
    ``throttled``. A line with no input rail has ``power_w`` None.
    """
    text = line.strip()
    rails: dict[str, float] = {}
    for match in _POWER_MW.finditer(text):
        rails[match.group(1)] = float(match.group(2)) / 1000.0
    for match in _POWER_LEGACY.finditer(text):
        rails.setdefault(match.group(1), float(match.group(2)) / 1000.0)
    power = None
    for name in _INPUT_RAILS:
        if name in rails:
            power = rails[name]
            break
    host = None
    ram = _RAM.search(text)
    if ram:
        host = float(ram.group(1))
    temperature = None
    for match in _TEMP.finditer(text):
        value = float(match.group(2))
        temperature = value if temperature is None else max(temperature, value)
    clocks: dict[str, float] = {}
    cpu = _CPU.search(text)
    if cpu:
        for index, core in enumerate(_CPU_CORE.finditer(cpu.group(1))):
            clocks[f"cpu{index}"] = float(core.group(2))
    for match in _FREQ.finditer(text):
        if match.group(3) is None:
            continue
        clocks[match.group(1).lower()] = float(match.group(3))
    caps: list[str] = []
    for match in _CLOCK_CAP.finditer(text):
        current = float(match.group(2))
        limit = float(match.group(3))
        clocks.setdefault(match.group(1).lower(), current)
        if current < limit:
            caps.append(f"{match.group(1).lower()} {current:g}/{limit:g} MHz")
    word = _THROTTLED.search(text) is not None
    details: list[str] = []
    if word:
        details.append("throttled")
    details.extend(caps)
    return SensorSample(
        "tegrastats",
        power_w=power,
        host_memory_mb=host,
        temperature_c=temperature,
        throttled=word or bool(caps),
        throttle_detail=", ".join(details) or None,
        clocks_mhz=clocks,
    )


def sample_from_jtop(board: Any) -> SensorSample:
    """Read one jtop board object. Unknown shapes leave the field None.

    Rail ``power`` values are milliwatts, matching the tegrastats text.
    """
    power = _jtop_power_w(getattr(board, "power", None))
    host = _jtop_ram_mb(getattr(board, "stats", None))
    temperature = _jtop_temperature_c(board)
    throttled, detail = _jtop_throttled(board)
    return SensorSample(
        "tegrastats",
        power_w=power,
        host_memory_mb=host,
        temperature_c=temperature,
        throttled=throttled,
        throttle_detail=detail,
    )


class TegraStatsSource(SensorSource):
    """tegrastats when the binary is on PATH, otherwise jtop when it imports."""

    source_id = "tegrastats"

    def __init__(
        self,
        *,
        which: Callable[[str], str | None] | None = None,
        jtop_loader: Callable[[], Any] | None = None,
        lines: Callable[[], str | None] | None = None,
        throttle_reader: Callable[[], str | None] | None = None,
    ) -> None:
        self._which = which or shutil.which
        self._jtop_loader = jtop_loader or _load_jtop
        self._lines = lines
        self._throttle_reader = throttle_reader
        self._binary: str | None = None
        self._mode: str | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._tail: threading.Thread | None = None
        self._latest = ""
        self._jetson: Any = None

    def probe(self) -> SourceStatus:
        if self._lines is not None:
            self._mode = "lines"
            return SourceStatus(
                self.source_id,
                True,
                power_domain=DOMAIN_BOARD,
                observes_throttle=True,
            )
        found = self._which("tegrastats")
        if found:
            self._binary = found
            self._mode = "binary"
            return SourceStatus(
                self.source_id,
                True,
                power_domain=DOMAIN_BOARD,
                observes_throttle=True,
            )
        try:
            self._jtop_loader()
        except Exception as exc:
            text = str(exc).strip()
            if not text or text == "jtop is not installed":
                reason = "tegrastats was not found and jtop is not installed"
            else:
                reason = f"tegrastats was not found and jtop is not installed ({text})"
            return SourceStatus(self.source_id, False, reason=reason)
        self._mode = "jtop"
        return SourceStatus(
            self.source_id,
            True,
            power_domain=DOMAIN_BOARD,
            observes_throttle=True,
        )

    def start(self) -> None:
        if self._mode == "binary":
            self._start_binary()
        elif self._mode == "jtop":
            self._start_jtop()

    def read(self) -> SensorSample:
        if self._mode == "jtop":
            if self._jetson is None:
                return SensorSample(self.source_id)
            try:
                return sample_from_jtop(self._jetson)
            except Exception:
                return SensorSample(self.source_id)
        text = self._next_line()
        if not text:
            return SensorSample(self.source_id)
        if self._throttle_active() and _THROTTLED.search(text) is None:
            text = text.rstrip() + " THROTTLED"
        return parse_tegrastats_line(text)

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
        tail = self._tail
        self._tail = None
        if tail is not None and tail.is_alive():
            tail.join(timeout=1)
        self._stop_jtop()

    def _start_binary(self) -> None:
        if self._proc is not None or not self._binary:
            return
        self._latest = ""
        self._proc = subprocess.Popen(
            [self._binary, "--interval", "100"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._tail = threading.Thread(
            target=self._pump,
            name="ohho-bench-tegrastats",
            daemon=True,
        )
        self._tail.start()

    def _pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._latest = line

    def _start_jtop(self) -> None:
        factory = self._jtop_loader()
        board = factory()
        start = getattr(board, "start", None)
        if callable(start):
            start()
        ok = getattr(board, "ok", None)
        if callable(ok) and not ok():
            self._jetson = board
            self._stop_jtop()
            raise OSError("jtop is not running")
        self._jetson = board

    def _stop_jtop(self) -> None:
        board = self._jetson
        self._jetson = None
        if board is None:
            return
        for name in ("close", "stop"):
            method = getattr(board, name, None)
            if not callable(method):
                continue
            try:
                method()
            except Exception:
                return
            return

    def _next_line(self) -> str | None:
        if self._lines is not None:
            return self._lines()
        latest = self._latest.strip()
        return latest or None

    def _throttle_active(self) -> bool:
        if self._lines is not None and self._throttle_reader is None:
            return False
        reader = self._throttle_reader or _default_throttle_text
        try:
            text = reader()
        except Exception:
            return False
        if not isinstance(text, str):
            return False
        return _throttle_text_active(text)


def source() -> TegraStatsSource:
    """Adapter entry used by discovery."""
    return TegraStatsSource()


def _load_jtop() -> Any:
    try:
        from jtop import jtop as factory
    except Exception as exc:
        raise ImportError("jtop is not installed") from exc
    return factory


def _default_throttle_text() -> str | None:
    for path in _THROTTLE_FILES:
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return None


def _throttle_text_active(text: str) -> bool:
    stripped = text.strip().lower()
    if stripped in _QUIET_THROTTLE:
        return False
    if "throttl" in stripped:
        return True
    return stripped.isdigit() and int(stripped) > 0


def _jtop_power_w(power: Any) -> float | None:
    if not isinstance(power, dict):
        return None
    for key in ("tot", "VDD_IN", "POM_5V_IN"):
        watts = _mw(power.get(key))
        if watts is not None:
            return watts
    rail = power.get("rail")
    if isinstance(rail, dict):
        for key in ("VDD_IN", "POM_5V_IN"):
            watts = _mw(rail.get(key))
            if watts is not None:
                return watts
    return None


def _mw(rail: Any) -> float | None:
    if not isinstance(rail, dict):
        return None
    for name in ("power", "instant", "cur"):
        number = _nonnegative(rail.get(name))
        if number is not None:
            return number / 1000.0
    return None


def _jtop_ram_mb(stats: Any) -> float | None:
    if not isinstance(stats, dict):
        return None
    for key in ("RAM", "RAM used"):
        value = stats.get(key)
        number = _nonnegative(value)
        if number is not None:
            return number
        if isinstance(value, dict):
            for inner in ("use", "used", "cur"):
                number = _nonnegative(value.get(inner))
                if number is not None:
                    return number
    return None


def _jtop_temperature_c(board: Any) -> float | None:
    raw = getattr(board, "temperature", None)
    peak = _max_temp(raw) if isinstance(raw, dict) else None
    if peak is not None:
        return peak
    stats = getattr(board, "stats", None)
    if not isinstance(stats, dict):
        return None
    for key, value in stats.items():
        if not isinstance(key, str) or not key.lower().startswith("temp"):
            continue
        number = _finite(value)
        if number is None:
            continue
        peak = number if peak is None else max(peak, number)
    return peak


def _max_temp(raw: dict[str, Any]) -> float | None:
    peak: float | None = None
    for value in raw.values():
        number = _finite(value)
        if isinstance(value, dict):
            for inner in ("temp", "temperature", "c"):
                number = _finite(value.get(inner))
                if number is not None:
                    break
        if number is None:
            continue
        peak = number if peak is None else max(peak, number)
    return peak


def _jtop_throttled(board: Any) -> tuple[bool, str | None]:
    for name in ("throttled", "throttle"):
        value = getattr(board, name, None)
        if value is True:
            return True, "throttled"
        if isinstance(value, str) and "throttl" in value.lower():
            return True, value.strip()
    return False, None


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
