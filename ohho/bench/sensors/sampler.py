"""Sample optional sensors on a thread at 10 Hz during the measured window.

The first sample is taken on that thread before the window is returned, and
another is taken when the window closes. Missing quantities stay null.
Energy per inference is the headline mean power in watts times the mean
latency in milliseconds, which is millijoules. Board input power is preferred
over accelerator power, which is preferred over CPU package power, so those
domains are not added together.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ohho.bench.sensors.base import (
    DOMAIN_BOARD,
    DOMAIN_DEVICE,
    DOMAIN_PACKAGE,
    SensorSample,
    SensorSource,
    SourceStatus,
)

SAMPLE_HZ = 10.0

_DOMAIN_RANK = {DOMAIN_BOARD: 0, DOMAIN_DEVICE: 1, DOMAIN_PACKAGE: 2}
_START_TIMEOUT_S = 2.0
_JOIN_TIMEOUT_S = 2.0


@dataclass
class WindowSummary:
    """Samples collected on the sampler thread for one measured window."""

    sample_hz: float
    sample_count: int
    samples: list[tuple[float, SensorSample]] = field(default_factory=list)
    statuses: list[SourceStatus] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_block(
        self,
        *,
        mean_latency_ms: float,
        host_fallback_mb: float | None = None,
        host_fallback_source: str | None = None,
    ) -> dict[str, Any]:
        """JSON object stored on ``telemetry.sensors``.

        ``host_fallback_mb`` is used only when no source reported host memory.
        It is a measured process figure, passed in by the harness, not a guess.
        """
        powers = _headline_power(self.samples, self.statuses)
        power_mean = None if powers is None else sum(powers) / len(powers)
        power_peak = None if powers is None else max(powers)
        energy = _energy_mj(power_mean, mean_latency_ms)
        device = _peak(self.samples, "device_memory_mb")
        host = _peak(self.samples, "host_memory_mb")
        host_source = _host_source(self.samples) if host is not None else None
        if host is None and _finite_nonnegative(host_fallback_mb):
            host = float(host_fallback_mb)
            host_source = host_fallback_source
        temperature = _peak(self.samples, "temperature_c", allow_negative=True)
        clocks = _clocks(self.samples)
        events = _events(self.samples)
        known_throttle = any(
            status.available and status.observes_throttle for status in self.statuses
        )
        block: dict[str, Any] = {
            "sample_hz": self.sample_hz,
            "sample_count": self.sample_count,
            "power_w_mean": power_mean,
            "power_w_peak": power_peak,
            "energy_per_inference_mj": energy,
            "device_memory_mb_peak": device,
            "host_memory_mb_peak": host,
            "host_memory_source": host_source,
            "temperature_c_max": temperature,
            "throttling_events": events,
            "throttling_flagged": bool(events),
            "clocks_mhz": clocks or None,
            "sources": [_source_row(status) for status in self.statuses],
        }
        block["reason"] = _reason(block, self.statuses, self.notes, known_throttle)
        return block


class SensorSampler:
    """Own the sources for one benchmark and open a window per measured batch."""

    def __init__(
        self,
        sources: Sequence[SensorSource] | None = None,
        *,
        hz: float = SAMPLE_HZ,
        sleeper: Callable[[float], bool] | None = None,
    ) -> None:
        if sources is None:
            from ohho.bench.sensors.registry import discover_sources

            sources = discover_sources()
        if isinstance(hz, bool) or not isinstance(hz, (int, float)):
            raise ValueError("sample rate must be a positive number")
        if not math.isfinite(float(hz)) or float(hz) <= 0:
            raise ValueError("sample rate must be a positive number")
        self._sources = list(sources)
        self._hz = float(hz)
        self._sleeper = sleeper
        self._statuses: list[SourceStatus] | None = None

    @property
    def hz(self) -> float:
        return self._hz

    def probe(self) -> list[SourceStatus]:
        """Probe each source once. A probe that raises becomes unavailable."""
        if self._statuses is None:
            found: list[SourceStatus] = []
            for source in self._sources:
                found.append(_safe_probe(source))
            self._statuses = found
        return list(self._statuses)

    def open(self) -> SampleWindow:
        """Start the sampler thread and block until the first sample returns."""
        statuses = self.probe()
        pairs: list[tuple[SensorSource, SourceStatus]] = []
        updated: list[SourceStatus] = []
        for source, status in zip(self._sources, statuses, strict=True):
            if not status.available:
                updated.append(status)
                continue
            try:
                source.start()
            except Exception as exc:
                message = _exc_text(exc)
                updated.append(
                    SourceStatus(
                        status.id,
                        False,
                        reason=message,
                    )
                )
                continue
            pairs.append((source, status))
            updated.append(status)
        window = SampleWindow(pairs, self._hz, self._sleeper, updated)
        window.start()
        return window

    def close(self) -> None:
        """Release every source. Safe to call more than once."""
        for source in self._sources:
            try:
                source.close()
            except Exception:
                continue


class SampleWindow:
    """One measured window. Samples are taken on ``ohho-bench-sensors``."""

    def __init__(
        self,
        pairs: Sequence[tuple[SensorSource, SourceStatus]],
        hz: float,
        sleeper: Callable[[float], bool] | None,
        statuses: Sequence[SourceStatus],
    ) -> None:
        self._pairs = list(pairs)
        self._hz = hz
        self._sleeper = sleeper
        self._statuses = list(statuses)
        self._stop = threading.Event()
        self._started = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[tuple[float, SensorSample]] = []
        self._notes: list[str] = []
        self._ticks = 0
        self._t0 = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            name="ohho-bench-sensors",
            daemon=True,
        )

    def start(self) -> None:
        """Start the thread and wait until it has taken the first sample."""
        self._thread.start()
        if not self._started.wait(timeout=_START_TIMEOUT_S):
            self._remember("sampler thread did not start")

    def close(self) -> WindowSummary:
        """Stop the thread, take the final sample, and stop the live sources."""
        self._stop.set()
        self._thread.join(timeout=_JOIN_TIMEOUT_S)
        if self._thread.is_alive():
            self._remember("sampler thread did not stop")
        for source, _status in self._pairs:
            try:
                source.stop()
            except Exception as exc:
                self._remember(f"{source.source_id}: {_exc_text(exc)}")
        with self._lock:
            return WindowSummary(
                sample_hz=self._hz,
                sample_count=self._ticks,
                samples=list(self._samples),
                statuses=list(self._statuses),
                notes=list(self._notes),
            )

    def _run(self) -> None:
        try:
            self._take()
            self._started.set()
            while True:
                if self._wait() or self._stop.is_set():
                    break
                self._take()
            self._take()
        except Exception as exc:
            self._remember(f"sampler failed: {_exc_text(exc)}")
            self._started.set()

    def _wait(self) -> bool:
        if self._sleeper is not None:
            return bool(self._sleeper(1.0 / self._hz))
        return self._stop.wait(1.0 / self._hz)

    def _take(self) -> None:
        now = time.monotonic() - self._t0
        batch: list[tuple[float, SensorSample]] = []
        for source, _status in self._pairs:
            try:
                sample = source.read()
            except Exception as exc:
                self._remember(f"{source.source_id}: {_exc_text(exc)}")
                continue
            if isinstance(sample, SensorSample):
                batch.append((now, sample))
        with self._lock:
            self._ticks += 1
            self._samples.extend(batch)

    def _remember(self, note: str) -> None:
        text = note.strip()
        if not text:
            return
        with self._lock:
            if text not in self._notes:
                self._notes.append(text)


def _safe_probe(source: SensorSource) -> SourceStatus:
    try:
        status = source.probe()
    except Exception as exc:
        return SourceStatus(
            str(getattr(source, "source_id", "sensor") or "sensor"),
            False,
            reason=f"probe failed: {_exc_text(exc)}",
        )
    if isinstance(status, SourceStatus):
        return status
    return SourceStatus(
        str(getattr(source, "source_id", "sensor") or "sensor"),
        False,
        reason="probe did not return a source status",
    )


def _exc_text(exc: BaseException) -> str:
    text = str(exc).strip()
    if text:
        return text
    return type(exc).__name__


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _finite_nonnegative(value: object) -> bool:
    return _finite(value) and float(value) >= 0


def _energy_mj(power_mean: float | None, mean_latency_ms: object) -> float | None:
    if power_mean is None or not _finite_nonnegative(mean_latency_ms):
        return None
    return power_mean * float(mean_latency_ms)


def _headline_power(
    samples: Sequence[tuple[float, SensorSample]],
    statuses: Sequence[SourceStatus],
) -> list[float] | None:
    by_source: dict[str, list[float]] = {}
    for _t_s, sample in samples:
        if not _finite_nonnegative(sample.power_w):
            continue
        by_source.setdefault(sample.source_id, []).append(float(sample.power_w))
    ranked: list[tuple[int, str, list[float]]] = []
    for status in statuses:
        values = by_source.get(status.id)
        if not values:
            continue
        rank = _DOMAIN_RANK.get(status.power_domain, 9)
        ranked.append((rank, status.id, values))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2]


def _peak(
    samples: Sequence[tuple[float, SensorSample]],
    name: str,
    *,
    allow_negative: bool = False,
) -> float | None:
    peak: float | None = None
    for _t_s, sample in samples:
        value = getattr(sample, name)
        if allow_negative:
            if not _finite(value):
                continue
        elif not _finite_nonnegative(value):
            continue
        number = float(value)
        peak = number if peak is None else max(peak, number)
    return peak


def _host_source(samples: Sequence[tuple[float, SensorSample]]) -> str | None:
    best: tuple[float, str] | None = None
    for _t_s, sample in samples:
        if not _finite_nonnegative(sample.host_memory_mb):
            continue
        number = float(sample.host_memory_mb)
        if best is None or number > best[0]:
            best = (number, sample.source_id)
    if best is None:
        return None
    return best[1]


def _clocks(samples: Sequence[tuple[float, SensorSample]]) -> dict[str, float]:
    found: dict[str, float] = {}
    for _t_s, sample in samples:
        clocks: Mapping[str, float] = sample.clocks_mhz
        for name, mhz in clocks.items():
            if not isinstance(name, str) or not name.strip():
                continue
            if not _finite_nonnegative(mhz):
                continue
            key = f"{sample.source_id}.{name}"
            found[key] = max(found.get(key, 0.0), float(mhz))
    return found


def _events(samples: Sequence[tuple[float, SensorSample]]) -> list[dict[str, Any]]:
    active: dict[str, bool] = {}
    events: list[dict[str, Any]] = []
    for t_s, sample in samples:
        now = bool(sample.throttled)
        if now and not active.get(sample.source_id, False):
            detail = sample.throttle_detail or "throttled"
            events.append(
                {
                    "source": sample.source_id,
                    "t_s": t_s,
                    "detail": str(detail),
                }
            )
        active[sample.source_id] = now
    return events


def _source_row(status: SourceStatus) -> dict[str, Any]:
    return {
        "id": status.id,
        "available": status.available,
        "reason": status.reason,
        "power_domain": status.power_domain,
    }


def _reason(
    block: Mapping[str, Any],
    statuses: Sequence[SourceStatus],
    notes: Sequence[str],
    throttling_known: bool,
) -> str | None:
    missing: list[str] = []
    if block["power_w_mean"] is None:
        missing.append("power")
    if block["device_memory_mb_peak"] is None:
        missing.append("device memory")
    if block["host_memory_mb_peak"] is None:
        missing.append("host memory")
    if block["temperature_c_max"] is None:
        missing.append("temperature")
    if block["clocks_mhz"] is None:
        missing.append("clocks")
    if not throttling_known:
        missing.append("throttling")
    if not missing and not notes:
        return None
    parts: list[str] = []
    if missing:
        parts.append("no sensor reported " + ", ".join(missing))
    parts.extend(notes)
    text = "; ".join(parts)
    unavailable = [
        f"{status.id}: {status.reason}"
        for status in statuses
        if not status.available and status.reason
    ]
    if unavailable:
        text = f"{text} ({'; '.join(unavailable)})"
    return text
