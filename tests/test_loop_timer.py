"""LoopTimer deadlines, rolling stats, and the loops that publish them."""

from __future__ import annotations

import math
import time
import unittest

from ohho.adapters.feetech import FeetechTransport
from ohho.robots.unitree.dds import UnitreeDdsTransport
from ohho.robots.omnibot.yahboom import YahboomTransport
from ohho.loop_timer import LoopTimer
from ohho.registry import get_spec
from ohho.runtime.native import NativeRuntime


class _Clock:
    """Monotonic fake. Tests advance it; it never reads the wall clock."""

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, dt: float) -> None:
        if dt < 0.0:
            raise ValueError("clock cannot go backwards")
        self._now += dt

    def set(self, now: float) -> None:
        if now < self._now:
            raise ValueError("clock cannot go backwards")
        self._now = now


class TestLoopTimer(unittest.TestCase):
    def test_rejects_bad_period_and_window(self) -> None:
        for period in (0, -1, float("nan"), float("inf"), True, "0.1"):
            with self.assertRaises(ValueError):
                LoopTimer(period)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            LoopTimer(0.01, window=0)
        with self.assertRaises(ValueError):
            LoopTimer(0.01, window=True)  # type: ignore[arg-type]

    def test_empty_stats(self) -> None:
        stats = LoopTimer(0.02).stats()
        self.assertAlmostEqual(stats.period_s, 0.02)
        self.assertEqual(stats.samples, 0)
        self.assertEqual(stats.deadline_misses, 0)
        self.assertEqual(stats.cycle_time_s, 0.0)
        self.assertEqual(stats.lateness_s, 0.0)
        self.assertEqual(stats.jitter_s, 0.0)
        self.assertEqual(stats.p50_s, 0.0)
        self.assertEqual(stats.p99_s, 0.0)

    def test_end_before_begin_and_double_begin(self) -> None:
        timer = LoopTimer(0.01, clock=_Clock())
        with self.assertRaises(RuntimeError):
            timer.end()
        timer.begin()
        with self.assertRaises(RuntimeError):
            timer.begin()

    def test_exact_period_is_not_a_miss(self) -> None:
        clock = _Clock()
        timer = LoopTimer(0.01, clock=clock)
        timer.start()
        for _ in range(3):
            timer.begin()
            clock.advance(0.01)
            self.assertEqual(timer.end(), 0.0)
        self.assertEqual(timer.stats().deadline_misses, 0)
        self.assertAlmostEqual(clock(), 0.03)
        self.assertAlmostEqual(timer.stats().lateness_s, 0.0)

    def test_slow_callback_counts_deadline_misses(self) -> None:
        clock = _Clock()
        timer = LoopTimer(0.01, clock=clock)

        def slow() -> None:
            clock.advance(0.05)

        self.assertEqual(timer.run(slow), 0.0)
        stats = timer.stats()
        self.assertEqual(stats.deadline_misses, 5)
        self.assertAlmostEqual(stats.cycle_time_s, 0.05)
        self.assertGreater(stats.lateness_s, 0.0)
        self.assertGreaterEqual(stats.p99_s, stats.p95_s)
        self.assertGreaterEqual(stats.p95_s, stats.p50_s)

    def test_exception_in_callback_still_records_the_miss(self) -> None:
        clock = _Clock()
        timer = LoopTimer(0.01, clock=clock)

        def slow() -> None:
            clock.advance(0.05)
            raise RuntimeError("overran")

        with self.assertRaises(RuntimeError):
            timer.run(slow)
        stats = timer.stats()
        self.assertEqual(stats.samples, 1)
        self.assertGreater(stats.deadline_misses, 0)

    def test_drift_over_10000_ticks_stays_within_one_period(self) -> None:
        period = 0.01
        clock = _Clock()
        timer = LoopTimer(period, clock=clock)
        timer.start()
        origin = clock()
        max_err = 0.0
        for i in range(10000):
            timer.begin()
            clock.advance(0.0005 + (i % 5) * 0.001)
            delay = timer.end()
            if delay > 0.0:
                clock.advance(delay)
            err = abs(clock() - (origin + (i + 1) * period))
            if err > max_err:
                max_err = err
        self.assertLess(max_err, period)
        self.assertLess(max_err, 1e-6)
        stats = timer.stats()
        self.assertEqual(stats.samples, 10000)
        self.assertEqual(stats.deadline_misses, 0)
        self.assertEqual(len(timer._cycles), 256)
        self.assertLess(stats.jitter_s, 1e-9)
        self.assertAlmostEqual(clock(), timer.next_deadline)

    def test_overrun_returns_to_the_grid_for_the_next_10000_ticks(self) -> None:
        period = 0.01
        clock = _Clock()
        timer = LoopTimer(period, clock=clock)
        timer.start()
        timer.begin()
        clock.advance(0.025)
        delay = timer.end()
        clock.advance(delay)
        self.assertEqual(timer.stats().deadline_misses, 3)
        self.assertAlmostEqual(clock(), 0.03)
        misses = timer.stats().deadline_misses
        anchor = clock()
        max_err = 0.0
        for i in range(10000):
            timer.begin()
            clock.advance(0.001)
            delay = timer.end()
            if delay > 0.0:
                clock.advance(delay)
            err = abs(clock() - (anchor + (i + 1) * period))
            if err > max_err:
                max_err = err
        self.assertLess(max_err, period)
        self.assertEqual(timer.stats().deadline_misses, misses)

    def test_rolling_window_percentiles(self) -> None:
        clock = _Clock()
        timer = LoopTimer(1.0, clock=clock, window=4)
        for duration in (0.001, 0.002, 0.003, 0.004, 0.005, 0.006):
            timer.begin()
            clock.advance(duration)
            timer.end(wait=False)
        stats = timer.stats()
        self.assertEqual(stats.samples, 6)
        self.assertAlmostEqual(stats.p50_s, 0.004)
        self.assertAlmostEqual(stats.p95_s, 0.006)
        self.assertAlmostEqual(stats.p99_s, 0.006)
        self.assertAlmostEqual(stats.cycle_time_s, 0.006)
        self.assertLessEqual(stats.p50_s, stats.p95_s)
        self.assertLessEqual(stats.p95_s, stats.p99_s)
        self.assertEqual(len(timer._cycles), 4)

    def test_jitter_is_stddev_of_inter_start_intervals(self) -> None:
        clock = _Clock()
        timer = LoopTimer(1.0, clock=clock)
        for start in (0.0, 0.01, 0.03, 0.06):
            clock.set(start)
            timer.begin()
            clock.advance(0.001)
            timer.end(wait=False)
        expected = math.sqrt(0.0002 / 3.0)
        stats = timer.stats()
        self.assertAlmostEqual(stats.jitter_s, expected)
        self.assertAlmostEqual(stats.observed_period_s, 0.03)
        self.assertAlmostEqual(stats.period_p50_s, 0.02)

    def test_reanchor_when_the_loop_must_not_wait(self) -> None:
        clock = _Clock()
        timer = LoopTimer(0.01, clock=clock)
        timer.start()
        timer.begin()
        clock.advance(0.001)
        self.assertEqual(timer.end(wait=False), 0.0)
        timer.begin()
        clock.advance(0.001)
        delay = timer.end()
        self.assertAlmostEqual(delay, 0.009)
        self.assertEqual(timer.stats().deadline_misses, 0)


class TestNativeRuntimeTiming(unittest.TestCase):
    def test_loop_stats_before_start(self) -> None:
        rt = NativeRuntime(0.01)
        stats = rt.loop_stats()
        self.assertEqual(stats.samples, 0)
        self.assertEqual(stats.deadline_misses, 0)
        self.assertAlmostEqual(stats.period_s, 0.01)

    def test_spin_records_a_miss_when_a_callback_overruns(self) -> None:
        clock = _Clock()
        rt = NativeRuntime(0.01, clock=clock)
        calls = {"n": 0}

        def slow() -> None:
            clock.advance(0.05)
            calls["n"] += 1
            if calls["n"] >= 2:
                rt._stop.set()

        rt.create_timer(0.01, slow)
        waits: list[float] = []

        def _wait(delay: float) -> bool:
            waits.append(delay)
            clock.advance(delay)
            return rt._stop.is_set()

        rt._stop.wait = _wait  # type: ignore[method-assign]
        rt._spin()
        stats = rt.loop_stats()
        self.assertGreaterEqual(calls["n"], 2)
        self.assertGreater(stats.deadline_misses, 0)
        self.assertGreater(stats.samples, 0)
        self.assertAlmostEqual(stats.period_s, 0.01)
        self.assertTrue(waits)

    def test_late_poll_fires_once_and_keeps_phase(self) -> None:
        clock = _Clock()
        rt = NativeRuntime(0.02, clock=clock)
        fires: list[float] = []
        rt.create_timer(0.02, lambda: fires.append(clock()))
        clock.set(0.20)
        rt._dispatch(clock())
        rt._dispatch(clock())
        self.assertEqual(fires, [0.20])
        timer = rt._timers[0]
        self.assertAlmostEqual(timer.origin + timer.index * timer.period, 0.22)

    def test_phase_over_10000_late_polls_stays_within_one_period(self) -> None:
        period = 0.02
        clock = _Clock()
        rt = NativeRuntime(period, clock=clock)
        fires: list[float] = []
        rt.create_timer(period, lambda: fires.append(clock()))
        for i in range(1, 10001):
            clock.set(i * period + 0.001)
            rt._dispatch(clock())
        self.assertEqual(len(fires), 10000)
        timer = rt._timers[0]
        scheduled = timer.origin + timer.index * timer.period
        self.assertLess(abs(scheduled - (10001 * period)), period)
        self.assertLess(abs(scheduled - (10001 * period)), 1e-6)


class _Serial:
    def write(self, data: bytes) -> int:
        return len(data)

    def read(self, n: int = 1) -> bytes:
        return b""

    def close(self) -> None:
        return None


class _Bus:
    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def read(self, key: str) -> dict[str, int]:
        return {}

    def write(self, key: str, values: dict[str, int]) -> None:
        return None


class TestWiredReaderTiming(unittest.TestCase):
    def test_readers_expose_stats_and_record_samples(self) -> None:
        unitree = UnitreeDdsTransport(
            get_spec("unitree-go2"),
            "lo",
            state_factory=lambda: (
                lambda: {
                    "position": [0.0, 0.0, 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "imu_rpy": [0.0, 0.0, 0.0],
                }
            ),
        )
        feetech = FeetechTransport(
            get_spec("omnibot"),
            "/dev/fake",
            bus_factory=_Bus,
        )
        yahboom = YahboomTransport(
            get_spec("omnibot"),
            "/dev/fake",
            serial_factory=_Serial,
        )
        self.assertEqual(unitree.loop_stats().samples, 0)
        self.assertEqual(feetech.loop_stats().samples, 0)
        self.assertEqual(yahboom.loop_stats().samples, 0)
        self.assertAlmostEqual(unitree.loop_stats().period_s, 1.0 / 20.0)
        self.assertAlmostEqual(feetech.loop_stats().period_s, 1.0 / 50.0)
        self.assertAlmostEqual(yahboom.loop_stats().period_s, 0.005)
        unitree.connect()
        feetech.connect()
        yahboom.connect()
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                stats = (
                    unitree.loop_stats(),
                    feetech.loop_stats(),
                    yahboom.loop_stats(),
                )
                if all(item.samples >= 1 for item in stats):
                    break
                time.sleep(0.01)
            for item in (
                unitree.loop_stats(),
                feetech.loop_stats(),
                yahboom.loop_stats(),
            ):
                self.assertGreaterEqual(item.samples, 1)
                self.assertGreaterEqual(item.deadline_misses, 0)
                self.assertLessEqual(item.p50_s, item.p95_s)
                self.assertLessEqual(item.p95_s, item.p99_s)
        finally:
            unitree.disconnect()
            feetech.disconnect()
            yahboom.disconnect()


if __name__ == "__main__":
    unittest.main()
