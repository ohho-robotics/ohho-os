"""Latency distributions for one bench sample set.

Percentiles use the linear method (Hyndman-Fan type 7): the same
interpolation NumPy's default ``percentile`` uses. ``std`` is the population
standard deviation. A single sample has ``std`` 0. ``median`` and ``p50``
are the same number.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def summarize(samples: Sequence[float]) -> dict[str, float]:
    """Return mean, median, p50, p90, p95, p99, min, max, and std.

    ``samples`` are in one unit (the harness passes milliseconds). Every
    value must be finite and non-negative. The result uses plain floats,
    with negative zero normalized to zero.
    """
    if not samples:
        raise ValueError("summarize needs at least one sample")
    values = [_nonnegative(sample) for sample in samples]
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    if len(ordered) < 2:
        std = 0.0
    else:
        std = statistics.pstdev(ordered)
    median = percentile(ordered, 50)
    return {
        "mean": _nonnegative(mean),
        "median": median,
        "p50": median,
        "p90": percentile(ordered, 90),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
        "min": ordered[0],
        "max": ordered[-1],
        "std": _nonnegative(std),
    }


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear percentile of a non-empty ascending sequence. ``pct`` is 0–100."""
    if not sorted_values:
        raise ValueError("percentile needs at least one sample")
    if len(sorted_values) == 1 or pct <= 0:
        return _nonnegative(sorted_values[0])
    if pct >= 100:
        return _nonnegative(sorted_values[-1])
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return _nonnegative(sorted_values[low])
    weight = rank - low
    blended = sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight
    return _nonnegative(blended)


def _nonnegative(value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"expected a finite non-negative number, got {value!r}")
    if number == 0:
        return 0.0
    return number
