"""Temperature, clocks, and the throttled bitfield from vcgencmd.

This source does not report watts. Voltage is not turned into power.
``get_throttled`` current bits 0–3 (under-voltage, frequency cap, throttled,
soft temperature limit) set ``throttled``. Bits 16–19 record events since
boot and are not treated as a throttle during this run. A missing binary
leaves the source unavailable.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable, Mapping

from ohho.bench.sensors.base import SensorSample, SensorSource, SourceStatus

_TEMP = re.compile(r"temp=(-?\d+(?:\.\d+)?)'C")
_FLAGS = re.compile(r"throttled=(0x[0-9a-fA-F]+|\d+)")
_FREQ = re.compile(r"frequency\(\d+\)=(\d+)")
_CURRENT_BITS: tuple[tuple[int, str], ...] = (
    (0, "under-voltage"),
    (1, "arm frequency capped"),
    (2, "throttled"),
    (3, "soft temperature limit"),
)
_QUERIES: tuple[tuple[str, ...], ...] = (
    ("measure_temp",),
    ("get_throttled",),
    ("measure_clock", "arm"),
    ("measure_clock", "core"),
)


def parse_vcgencmd(readings: Mapping[str, str]) -> SensorSample:
    """Parse command output. ``power_w`` is always None."""
    temperature: float | None = None
    flags: int | None = None
    clocks: dict[str, float] = {}
    for key, text in readings.items():
        if not isinstance(text, str) or not text.strip():
            continue
        temp_match = _TEMP.search(text)
        if temp_match:
            temperature = float(temp_match.group(1))
        flag_match = _FLAGS.search(text)
        if flag_match:
            raw = flag_match.group(1)
            flags = int(raw, 16) if raw.lower().startswith("0x") else int(raw, 10)
        freq_match = _FREQ.search(text)
        if freq_match and isinstance(key, str) and key.strip():
            clocks[key.split()[-1]] = int(freq_match.group(1)) / 1_000_000.0
    details: list[str] = []
    if flags is not None:
        for bit, label in _CURRENT_BITS:
            if flags & (1 << bit):
                details.append(label)
    return SensorSample(
        "vcgencmd",
        temperature_c=temperature,
        throttled=bool(details),
        throttle_detail=", ".join(details) or None,
        clocks_mhz=clocks,
    )


class VcgencmdSource(SensorSource):
    """vcgencmd queries. ``runner`` returns the output map and skips the binary."""

    source_id = "vcgencmd"

    def __init__(
        self,
        *,
        which: Callable[[str], str | None] | None = None,
        runner: Callable[[], Mapping[str, str]] | None = None,
    ) -> None:
        self._which = which or shutil.which
        self._runner = runner
        self._binary: str | None = None

    def probe(self) -> SourceStatus:
        if self._runner is not None:
            return SourceStatus(self.source_id, True, observes_throttle=True)
        found = self._which("vcgencmd")
        if not found:
            return SourceStatus(
                self.source_id,
                False,
                reason="vcgencmd was not found",
            )
        self._binary = found
        return SourceStatus(self.source_id, True, observes_throttle=True)

    def read(self) -> SensorSample:
        if self._runner is not None:
            try:
                readings = self._runner()
            except Exception:
                return SensorSample(self.source_id)
            if not isinstance(readings, Mapping):
                return SensorSample(self.source_id)
            return parse_vcgencmd(readings)
        binary = self._binary
        if not binary:
            return SensorSample(self.source_id)
        readings: dict[str, str] = {}
        for args in _QUERIES:
            key = args[-1] if args[0] == "measure_clock" else args[0]
            readings[key] = _query(binary, args)
        return parse_vcgencmd(readings)


def source() -> VcgencmdSource:
    """Adapter entry used by discovery."""
    return VcgencmdSource()


def _query(binary: str, args: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()
