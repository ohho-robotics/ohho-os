"""Exponential backoff for supervisor restarts.

The ohho-os link helper added in PR #15 doubles a base interval and caps
the wait at 10 seconds. That helper is a private name in an adapter
package. ``ohho_runtime`` may import only public ``ohho`` modules, so
this is the same schedule without that import.

The first wait is the base interval. Each later wait doubles, and no
wait is longer than :data:`MAX_BACKOFF_S`.
"""

from __future__ import annotations

MAX_BACKOFF_S = 10.0


def _positive(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("backoff interval must be a number > 0")
    number = float(value)
    if number <= 0:
        raise ValueError("backoff interval must be > 0")
    return number


class ExponentialBackoff:
    """Doubling restart delay with a fixed ceiling.

    ``next_delay`` is the wait before the next restart attempt. ``reset``
    starts the schedule over after the failure window slides empty.
    """

    def __init__(
        self,
        interval_s: float,
        *,
        ceiling_s: float = MAX_BACKOFF_S,
    ) -> None:
        self._interval_s = _positive(interval_s)
        self._ceiling_s = _positive(ceiling_s)
        self._backoff_s = 0.0

    @property
    def interval_s(self) -> float:
        return self._interval_s

    @property
    def ceiling_s(self) -> float:
        return self._ceiling_s

    def next_delay(self) -> float:
        """Return the next wait in seconds and remember it."""
        if self._backoff_s <= 0.0:
            delay = self._interval_s
        else:
            delay = max(self._interval_s, self._backoff_s * 2.0)
        self._backoff_s = min(self._ceiling_s, delay)
        return self._backoff_s

    def reset(self) -> None:
        """Forget the current delay so the next wait is the base interval."""
        self._backoff_s = 0.0


__all__ = ["MAX_BACKOFF_S", "ExponentialBackoff"]
