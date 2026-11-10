"""Binomial confidence intervals for ohho.eval/v1.

The report's 95% interval is the Wilson score interval. ``z`` is the
standard-normal quantile from :class:`statistics.NormalDist`, so the base
install does not need SciPy.
"""

from __future__ import annotations

import math
from statistics import NormalDist

# Two-sided 95% normal quantile. Computed once; NormalDist is deterministic.
Z_95 = NormalDist().inv_cdf(0.975)


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = Z_95,
) -> tuple[float, float]:
    """Return the Wilson score ``(low, high)`` for ``successes`` out of ``trials``.

    ``trials == 0`` has no information: the interval is ``(0, 1)``.
    Bounds are clamped to ``[0, 1]``.
    """
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise TypeError("successes and trials must be integers")
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise TypeError("successes and trials must be integers")
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    if z <= 0.0 or not math.isfinite(z):
        raise ValueError("z must be a positive finite number")
    if trials == 0:
        return 0.0, 1.0
    phat = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (phat + z2 / (2.0 * trials)) / denom
    margin = (z / denom) * math.sqrt(
        phat * (1.0 - phat) / trials + z2 / (4.0 * trials * trials)
    )
    return max(0.0, center - margin), min(1.0, center + margin)
