"""Re-export of LoopTimer and the service timing report."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from ohho.loop_timer import LoopTimer as CoreLoopTimer
from ohho_runtime.service.lifecycle import ManualClock
from ohho_runtime.timing import LoopStats, LoopTimer, loop_report

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "runtime" / "src"


class TestLoopReport(unittest.TestCase):
    def test_reexport_is_the_ohho_primitive(self) -> None:
        self.assertIs(LoopTimer, CoreLoopTimer)
        import ohho_runtime

        self.assertIs(ohho_runtime.LoopTimer, CoreLoopTimer)
        self.assertIs(ohho_runtime.LoopStats, LoopStats)
        self.assertIs(ohho_runtime.loop_report, loop_report)

    def test_slow_callback_is_visible_in_the_report(self) -> None:
        clock = ManualClock()
        timer = LoopTimer(0.01, clock=clock)

        def slow() -> None:
            clock.advance(0.05)

        self.assertEqual(timer.run(slow), 0.0)
        report = loop_report(timer.stats())
        self.assertGreater(report["deadline_misses"], 0)
        self.assertAlmostEqual(report["cycle_time_s"], 0.05)
        self.assertAlmostEqual(report["period_s"], 0.01)
        self.assertGreaterEqual(report["p99_s"], report["p95_s"])
        self.assertGreaterEqual(report["p95_s"], report["p50_s"])
        self.assertEqual(set(report), set(LoopStats.__dataclass_fields__))

    def test_drift_over_10000_ticks_stays_within_one_period(self) -> None:
        period = 0.01
        clock = ManualClock()
        timer = LoopTimer(period, clock=clock)
        timer.start()
        origin = clock()
        max_err = 0.0
        for i in range(10000):
            timer.begin()
            clock.advance(0.001)
            delay = timer.end()
            if delay > 0.0:
                clock.advance(delay)
            err = abs(clock() - (origin + (i + 1) * period))
            if err > max_err:
                max_err = err
        self.assertLess(max_err, period)
        self.assertEqual(timer.stats().deadline_misses, 0)
        report = loop_report(timer.stats())
        self.assertEqual(report["samples"], 10000)
        self.assertEqual(report["deadline_misses"], 0)

    def test_report_rejects_other_objects(self) -> None:
        with self.assertRaises(TypeError):
            loop_report(object())  # type: ignore[arg-type]

    def test_importing_heartbeat_does_not_import_ohho(self) -> None:
        env = dict(os.environ)
        prior = env.get("PYTHONPATH", "")
        paths = [str(_SRC), str(_ROOT)]
        if prior:
            paths.append(prior)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        probe = (
            "import ohho_runtime.heartbeat, sys\n"
            "raise SystemExit(0 if 'ohho' not in sys.modules else 1)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
