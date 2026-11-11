"""Latency drift across a sustained bench window.

The measured iterations are split into consecutive time windows. Each window
reports its own p95. Drift is the last window's p95 minus the first window's
p95, in milliseconds. A positive drift means later iterations were slower.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ohho.bench.stats import summarize


def window_seconds(sustained_s: float) -> float:
    """One-second windows, or two equal windows when the run is shorter."""
    if sustained_s >= 2.0:
        return 1.0
    return sustained_s / 2.0


def drift_report(
    stamps: Sequence[tuple[float, float]],
    *,
    sustained_s: float,
    throttling_flagged: bool,
    truncated: bool = False,
) -> dict[str, Any]:
    """Build the sustained block from ``(t_end_s, latency_ms)`` pairs.

    ``t_end_s`` is seconds from the start of the measured window. A sample
    past the last window is kept in that window.
    """
    window_s = window_seconds(sustained_s)
    count = max(1, math.ceil(sustained_s / window_s - 1e-12))
    buckets: list[list[float]] = [[] for _ in range(count)]
    for t_s, latency_ms in stamps:
        if t_s < 0:
            continue
        index = int(t_s / window_s)
        if index >= count:
            index = count - 1
        if index < 0:
            index = 0
        buckets[index].append(latency_ms)
    windows: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        stats = summarize(bucket)
        t1 = min(sustained_s, (index + 1) * window_s)
        windows.append(
            {
                "index": index,
                "t0_s": index * window_s,
                "t1_s": t1,
                "p95_ms": stats["p95"],
                "samples": len(bucket),
            }
        )
    if len(windows) >= 2:
        drift = windows[-1]["p95_ms"] - windows[0]["p95_ms"]
        first = windows[0]["p95_ms"]
        pct: float | None = None if first == 0 else 100.0 * drift / first
    else:
        drift = 0.0
        pct = 0.0
    return {
        "seconds": sustained_s,
        "window_s": window_s,
        "windows": windows,
        "p95_drift_ms": drift,
        "p95_drift_pct": pct,
        "throttling_flagged": bool(throttling_flagged),
        "truncated": bool(truncated),
    }
