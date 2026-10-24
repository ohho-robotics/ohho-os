"""Drift-free timer for one periodic loop.

The primitive lives here, in ``ohho``, because the native executor and the
transport readers call it. ``ohho`` does not import ``ohho_runtime``.
``ohho_runtime.timing`` re-exports this type and turns a snapshot into the
report a supervised service publishes.

:meth:`LoopTimer.begin` and :meth:`LoopTimer.end` write fixed rings
(``array.array``). They do not build lists, dicts, or snapshots.
:meth:`LoopTimer.stats` copies the window; that is the cold path.
"""

from __future__ import annotations

import array
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_WINDOW = 256

__all__ = ["DEFAULT_WINDOW", "LoopStats", "LoopTimer"]


def _index_at_or_after(origin: float, period: float, now: float) -> int:
    """Smallest ``k >= 0`` such that ``origin + k * period >= now``."""
    ratio = (now - origin) / period
    if ratio <= 0.0:
        return 0
    k = math.ceil(ratio - 1e-9)
    if k < 0:
        return 0
    return k


def _percentile(values: list[float], percent: float) -> float:
    """Nearest-rank percentile. ``values`` is not sorted."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil((percent / 100.0) * len(ordered))
    if rank < 1:
        rank = 1
    if rank > len(ordered):
        rank = len(ordered)
    return ordered[rank - 1]


def _population_stddev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((value - mean) ** 2 for value in values) / n
    return math.sqrt(var)


def _copy_filled(buf: array.array, filled: int, window: int) -> list[float]:
    if filled <= 0:
        return []
    if filled < window:
        return buf[:filled].tolist()
    return buf.tolist()


@dataclass(frozen=True, slots=True)
class LoopStats:
    """One read of a :class:`LoopTimer` window.

    ``period_s`` is the declared period. ``p50_s``, ``p95_s``, and ``p99_s``
    are cycle-time percentiles over the rolling window (nearest rank).
    ``lateness_s`` is the latest finish minus that cycle's deadline; a
    positive value missed the deadline. ``jitter_s`` is the population
    standard deviation of inter-start intervals in the window. Period
    percentiles describe those same intervals.
    """

    period_s: float
    samples: int
    deadline_misses: int
    cycle_time_s: float
    lateness_s: float
    jitter_s: float
    p50_s: float
    p95_s: float
    p99_s: float
    lateness_p50_s: float
    lateness_p95_s: float
    lateness_p99_s: float
    observed_period_s: float
    period_p50_s: float
    period_p95_s: float
    period_p99_s: float


class LoopTimer:
    """Monotonic timer for one periodic loop.

    Call :meth:`start`, then :meth:`begin` / :meth:`end` around the body.
    :meth:`end` returns how long to sleep so the next start lands on
    ``origin + n * period``. Sleeping that long keeps a run of ticks inside
    one period of the ideal grid. A body that finishes after its deadline
    counts a miss for every release that has already passed, and the next
    sleep is only the remainder to the following grid point.

    ``wait=False`` records the same sample and then re-anchors the grid at
    now. A reader that still has buffered data uses that so it can drain
    the buffer instead of holding the declared period.

    The clock must be monotonic. The default is :func:`time.monotonic`.
    """

    __slots__ = (
        "_period",
        "_clock",
        "_window",
        "_lock",
        "_cycles",
        "_lates",
        "_gaps",
        "_pos",
        "_filled",
        "_gap_pos",
        "_gap_filled",
        "_samples",
        "_misses",
        "_origin",
        "_n",
        "_cycle_start",
        "_last_begin",
        "_have_begin",
        "_in_cycle",
        "_started",
        "_last_cycle",
        "_last_late",
        "_last_gap",
        "_sched_start",
    )

    def __init__(
        self,
        period_s: float,
        *,
        clock: Callable[[], float] | None = None,
        window: int = DEFAULT_WINDOW,
    ) -> None:
        if isinstance(period_s, bool) or not isinstance(period_s, (int, float)):
            raise ValueError("period_s must be a number > 0")
        if not math.isfinite(period_s) or period_s <= 0.0:
            raise ValueError("period_s must be finite and > 0")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be an int >= 1")
        self._period = float(period_s)
        self._clock: Callable[[], float] = time.monotonic if clock is None else clock
        self._window = window
        self._lock = threading.Lock()
        self._cycles: array.array = array.array("d", [0.0]) * window
        self._lates: array.array = array.array("d", [0.0]) * window
        self._gaps: array.array = array.array("d", [0.0]) * window
        self._pos = 0
        self._filled = 0
        self._gap_pos = 0
        self._gap_filled = 0
        self._samples = 0
        self._misses = 0
        self._origin = 0.0
        self._n = 0
        self._cycle_start = 0.0
        self._last_begin = 0.0
        self._have_begin = False
        self._in_cycle = False
        self._started = False
        self._last_cycle = 0.0
        self._last_late = 0.0
        self._last_gap = 0.0
        self._sched_start = 0.0

    @property
    def period_s(self) -> float:
        """Declared period, in seconds."""
        return self._period

    @property
    def next_deadline(self) -> float:
        """Absolute time of the next scheduled start, on the same clock."""
        with self._lock:
            return self._origin + self._n * self._period

    def start(self) -> None:
        """Anchor the grid at the current clock.

        Samples and the miss counter are kept. A later :meth:`start` drops
        the in-progress cycle and the previous inter-start gap so time spent
        stopped is not a miss and does not inflate jitter.
        """
        now = self._clock()
        with self._lock:
            self._origin = now
            self._n = 0
            self._started = True
            self._in_cycle = False
            self._have_begin = False

    def begin(self) -> None:
        """Mark the start of one cycle. Pair with :meth:`end`."""
        now = self._clock()
        with self._lock:
            if not self._started:
                self._origin = now
                self._n = 0
                self._started = True
                self._have_begin = False
            if self._in_cycle:
                raise RuntimeError("LoopTimer.begin() called twice")
            if self._have_begin:
                gap = now - self._last_begin
                gpos = self._gap_pos
                self._gaps[gpos] = gap
                self._gap_pos = (gpos + 1) % self._window
                if self._gap_filled < self._window:
                    self._gap_filled += 1
                self._last_gap = gap
            self._last_begin = now
            self._have_begin = True
            self._cycle_start = now
            self._sched_start = self._origin + self._n * self._period
            self._in_cycle = True

    def end(self, *, wait: bool = True) -> float:
        """Record the cycle and return the sleep that hits the next deadline.

        The return value is ``>= 0``. Zero means the deadline has already
        passed; the caller should run the next cycle immediately. With
        ``wait=False`` the return is always zero and the grid is re-anchored
        at the finish time.
        """
        now = self._clock()
        with self._lock:
            if not self._in_cycle:
                raise RuntimeError("LoopTimer.end() called before begin()")
            cycle = now - self._cycle_start
            deadline = self._sched_start + self._period
            lateness = now - deadline
            self._store(cycle, lateness)
            self._in_cycle = False
            if lateness > 0.0:
                k = _index_at_or_after(self._origin, self._period, now)
                if k < self._n + 1:
                    k = self._n + 1
                self._misses += k - self._n
                self._n = k
            else:
                self._n += 1
            if not wait:
                self._origin = now
                self._n = 0
                return 0.0
            next_at = self._origin + self._n * self._period
        delay = next_at - self._clock()
        if delay <= 0.0:
            return 0.0
        return delay

    def run(self, callback: Callable[[], None], *, wait: bool = True) -> float:
        """:meth:`begin`, call ``callback``, :meth:`end`.

        The callback's exception propagates after the cycle is recorded.
        """
        self.begin()
        try:
            callback()
        finally:
            delay = self.end(wait=wait)
        return delay

    def stats(self) -> LoopStats:
        """Copy the rolling window into a snapshot."""
        with self._lock:
            cycles = _copy_filled(self._cycles, self._filled, self._window)
            lates = _copy_filled(self._lates, self._filled, self._window)
            gaps = _copy_filled(self._gaps, self._gap_filled, self._window)
            period = self._period
            samples = self._samples
            misses = self._misses
            last_cycle = self._last_cycle
            last_late = self._last_late
            last_gap = self._last_gap
        return LoopStats(
            period_s=period,
            samples=samples,
            deadline_misses=misses,
            cycle_time_s=last_cycle,
            lateness_s=last_late,
            jitter_s=_population_stddev(gaps),
            p50_s=_percentile(cycles, 50.0),
            p95_s=_percentile(cycles, 95.0),
            p99_s=_percentile(cycles, 99.0),
            lateness_p50_s=_percentile(lates, 50.0),
            lateness_p95_s=_percentile(lates, 95.0),
            lateness_p99_s=_percentile(lates, 99.0),
            observed_period_s=last_gap,
            period_p50_s=_percentile(gaps, 50.0),
            period_p95_s=_percentile(gaps, 95.0),
            period_p99_s=_percentile(gaps, 99.0),
        )

    def _store(self, cycle: float, lateness: float) -> None:
        pos = self._pos
        self._cycles[pos] = cycle
        self._lates[pos] = lateness
        self._pos = (pos + 1) % self._window
        if self._filled < self._window:
            self._filled += 1
        self._samples += 1
        self._last_cycle = cycle
        self._last_late = lateness
