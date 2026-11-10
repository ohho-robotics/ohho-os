"""CPU package power from the powercap energy counter.

Power is the difference between two energy readings divided by the elapsed
time. The first reading has no power yet. Package zones are the directories
whose ``name`` file starts with ``package``. Child zones are not added, so a
core counter is not double-counted on top of the package. A missing or
unreadable counter leaves the source unavailable.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from ohho.bench.sensors.base import (
    DOMAIN_PACKAGE,
    SensorSample,
    SensorSource,
    SourceStatus,
)

_DEFAULT_ROOT = Path("/sys/class/powercap")


class RaplSource(SensorSource):
    """Package energy counter. ``root`` overrides the powercap directory."""

    source_id = "rapl"

    def __init__(
        self,
        root: Path | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._root = root if root is not None else _DEFAULT_ROOT
        self._clock = clock or time.monotonic
        self._zones: list[Path] = []
        self._last: tuple[float, int] | None = None

    def probe(self) -> SourceStatus:
        zones = _package_zones(self._root)
        if not zones:
            return SourceStatus(
                self.source_id,
                False,
                reason="RAPL package energy file not found",
            )
        for zone in zones:
            try:
                _read_counter(zone / "energy_uj")
            except OSError as exc:
                detail = exc.strerror or str(exc)
                return SourceStatus(
                    self.source_id,
                    False,
                    reason=f"RAPL energy file is not readable: {detail}",
                )
        self._zones = zones
        self._last = None
        return SourceStatus(self.source_id, True, power_domain=DOMAIN_PACKAGE)

    def read(self) -> SensorSample:
        if not self._zones:
            return SensorSample(self.source_id)
        try:
            energy, span = _read_zones(self._zones)
        except OSError:
            return SensorSample(self.source_id)
        now = float(self._clock())
        previous = self._last
        self._last = (now, energy)
        if previous is None:
            return SensorSample(self.source_id)
        prev_t, prev_e = previous
        dt = now - prev_t
        if dt <= 0:
            return SensorSample(self.source_id)
        delta = _energy_delta(prev_e, energy, span)
        if delta is None:
            return SensorSample(self.source_id)
        return SensorSample(self.source_id, power_w=(delta / 1_000_000.0) / dt)

    def stop(self) -> None:
        self._last = None


def source() -> RaplSource:
    """Adapter entry used by discovery."""
    return RaplSource()


def _package_zones(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    found: list[Path] = []
    try:
        children = sorted(root.iterdir())
    except OSError:
        return []
    for child in children:
        name_file = child / "name"
        energy = child / "energy_uj"
        if not name_file.is_file() or not energy.is_file():
            continue
        try:
            label = name_file.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if label.startswith("package"):
            found.append(child)
    return found


def _read_zones(zones: list[Path]) -> tuple[int, int | None]:
    energy = 0
    span = 0
    saw_span = False
    for zone in zones:
        energy += _read_counter(zone / "energy_uj")
        span_path = zone / "max_energy_range_uj"
        if span_path.is_file():
            span += _read_counter(span_path)
            saw_span = True
    return energy, span if saw_span else None


def _read_counter(path: Path) -> int:
    text = path.read_text(encoding="utf-8").strip()
    return int(text)


def _energy_delta(previous: int, current: int, span: int | None) -> int | None:
    if current >= previous:
        return current - previous
    if span is None or span <= previous:
        return None
    return (span - previous) + current
