"""Shared serial-link resilience: telemetry timeout, watchdog stop, reconnect.

Used by the serial base and Feetech adapters. Pure stdlib — no hardware imports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

MonotonicFn = Callable[[], float]

DEFAULT_TELEMETRY_TIMEOUT_S: float = 2.0
DEFAULT_RECONNECT_INTERVAL_S: float = 1.0
MAX_RECONNECT_INTERVAL_S: float = 10.0


@dataclass
class LinkResilience:
    """Tracks RX freshness and reconnect backoff for one serial-style link."""

    telemetry_timeout_s: float = DEFAULT_TELEMETRY_TIMEOUT_S
    reconnect_interval_s: float = DEFAULT_RECONNECT_INTERVAL_S
    monotonic: MonotonicFn = time.monotonic
    _last_rx: Optional[float] = field(default=None, init=False, repr=False)
    _reconnect_after: Optional[float] = field(default=None, init=False, repr=False)
    _backoff_s: float = field(default=0.0, init=False, repr=False)
    link_up: bool = field(default=False, init=False, repr=False)

    def on_connected(self) -> None:
        now = self.monotonic()
        self._last_rx = now
        self.link_up = True
        self._reconnect_after = None
        self._backoff_s = 0.0

    def on_rx(self) -> None:
        self._last_rx = self.monotonic()
        self.link_up = True

    def on_link_lost(self) -> None:
        self.link_up = False
        if self._backoff_s <= 0.0:
            self._backoff_s = self.reconnect_interval_s
        self._reconnect_after = self.monotonic() + self._backoff_s

    def on_reconnect_failed(self) -> None:
        self.link_up = False
        self._backoff_s = min(
            MAX_RECONNECT_INTERVAL_S,
            max(self.reconnect_interval_s, self._backoff_s * 2.0),
        )
        self._reconnect_after = self.monotonic() + self._backoff_s

    def telemetry_stale(self) -> bool:
        if not self.link_up or self._last_rx is None:
            return False
        return (self.monotonic() - self._last_rx) > self.telemetry_timeout_s

    def should_reconnect(self) -> bool:
        if self._reconnect_after is None:
            return False
        return self.monotonic() >= self._reconnect_after

    def reset(self) -> None:
        self._last_rx = None
        self._reconnect_after = None
        self._backoff_s = 0.0
        self.link_up = False
