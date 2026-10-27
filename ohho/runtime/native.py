"""The native (no-ROS) runtime — a lightweight threaded scheduler.

Pure standard library: a background thread fires registered timers. This is the
default backend and runs anywhere Python runs. The thread sleeps until the
next :class:`~ohho.loop_timer.LoopTimer` deadline, and each registered timer
stays on the phase captured when it was created.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

from ..loop_timer import LoopStats, LoopTimer
from .base import Runtime, TimerHandle


@dataclass(eq=False)
class _Timer:
    period: float
    cb: Callable[[], None]
    origin: float
    index: int
    handle: TimerHandle


def _consume_deadline(
    origin: float, index: int, period: float, now: float
) -> int | None:
    """Advance a phase-locked deadline. ``None`` means it is not due yet.

    The returned index is the next slot strictly after ``now``, still on
    ``origin + k * period``. Missed slots are skipped rather than fired in a
    burst, and a late poll does not replace the phase with ``now + period``.
    """
    scheduled = origin + index * period
    if now < scheduled:
        return None
    late = now - scheduled
    advance = 1
    if late > 0.0:
        advance += int(late / period)
    return index + advance


class NativeRuntime(Runtime):
    name = "native"

    def __init__(
        self,
        tick_s: float = 0.005,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__()
        self._clock = time.monotonic if clock is None else clock
        self._timers: list[_Timer] = []
        self._tick = tick_s
        self._loop = LoopTimer(tick_s, clock=self._clock)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def now(self) -> float:
        return self._clock()

    def loop_stats(self) -> LoopStats:
        """Timing of the scheduler thread.

        Cycle time, lateness, jitter, percentiles, and deadline misses for
        the tick loop. Registered timers share this thread; their deadlines
        stay on the phase captured in :meth:`create_timer`.
        """
        return self._loop.stats()

    def create_timer(
        self, period_s: float, callback: Callable[[], None]
    ) -> TimerHandle:
        handle = TimerHandle()
        with self._lock:
            self._timers.append(_Timer(period_s, callback, self.now(), 1, handle))
        return handle

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._spin, name="ohho-native-runtime", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
            self._thread = None

    def _dispatch(self, now: float) -> None:
        with self._lock:
            timers = list(self._timers)
        for tm in timers:
            if tm.handle.cancelled:
                with self._lock:
                    self._timers = [item for item in self._timers if item is not tm]
                continue
            if tm.period > 0.0:
                nxt = _consume_deadline(tm.origin, tm.index, tm.period, now)
                if nxt is None:
                    continue
                tm.index = nxt
            try:
                tm.cb()
            except Exception:
                pass

    def _spin(self) -> None:
        self._loop.start()
        while not self._stop.is_set():
            self._loop.begin()
            try:
                self._dispatch(self.now())
            finally:
                delay = self._loop.end()
            if self._stop.is_set():
                return
            if delay > 0.0 and self._stop.wait(delay):
                return
