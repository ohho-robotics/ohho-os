"""Loop-timing reports for the service that owns a periodic loop.

Measurement is :class:`ohho.loop_timer.LoopTimer`. This module re-exports
that primitive and turns a snapshot into a flat report. It does not start
loops, command motion, or change a safety gate.
"""

from __future__ import annotations

from dataclasses import asdict

from ohho.loop_timer import LoopStats, LoopTimer

__all__ = ["LoopStats", "LoopTimer", "loop_report"]


def loop_report(stats: LoopStats) -> dict[str, float | int]:
    """Copy ``stats`` into a JSON-ready mapping.

    Keys match :class:`ohho.loop_timer.LoopStats`. ``deadline_misses`` is a
    count for the service that owns the loop. This report does not latch an
    e-stop and does not arm a robot.
    """
    if not isinstance(stats, LoopStats):
        raise TypeError("stats must be a LoopStats snapshot")
    report: dict[str, float | int] = {}
    for key, value in asdict(stats).items():
        report[key] = value
    return report
