"""Sensor readings for the bench sampler.

A source is optional. ``probe`` reports whether it can be read and, when it
cannot, a reason. Readings never invent a number for a quantity the source
did not report.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

DOMAIN_BOARD = "board"
DOMAIN_DEVICE = "device"
DOMAIN_PACKAGE = "package"

_DOMAINS = frozenset({DOMAIN_BOARD, DOMAIN_DEVICE, DOMAIN_PACKAGE})


@dataclass(frozen=True)
class SourceStatus:
    """Whether one source can be sampled.

    ``reason`` is required when ``available`` is false and must be empty
    when the source can be read. ``power_domain`` is ``board`` (whole-board
    input), ``device`` (an accelerator), or ``package`` (a CPU package).
    The sampler uses that label to avoid adding those domains together.
    """

    id: str
    available: bool
    reason: str | None = None
    power_domain: str | None = None
    observes_throttle: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("source id must be a non-empty string")
        if self.power_domain not in _DOMAINS and self.power_domain is not None:
            raise ValueError(f"unknown power domain {self.power_domain!r}")
        if self.available:
            if self.reason:
                raise ValueError("an available source does not take a failure reason")
            return
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("an unavailable source needs a reason")


@dataclass(frozen=True)
class SensorSample:
    """One reading. Unmeasured quantities stay ``None``."""

    source_id: str
    power_w: float | None = None
    device_memory_mb: float | None = None
    host_memory_mb: float | None = None
    temperature_c: float | None = None
    throttled: bool = False
    throttle_detail: str | None = None
    clocks_mhz: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "clocks_mhz", dict(self.clocks_mhz))


class SensorSource:
    """One optional power, memory, or thermal source.

    ``start`` runs after a successful probe and before the sampler thread
    reads. ``stop`` runs at the end of a measured window. ``close`` runs
    once when the benchmark finishes. The defaults do nothing.
    """

    source_id = "sensor"

    def probe(self) -> SourceStatus:
        """Return availability. Do not start a long-running reader here."""
        raise NotImplementedError

    def read(self) -> SensorSample:
        """Return the latest reading. Missing quantities are ``None``."""
        raise NotImplementedError

    def start(self) -> None:
        """Prepare to be read for one measured window."""

    def stop(self) -> None:
        """Stop any reader started for the current window."""

    def close(self) -> None:
        """Release anything ``start`` acquired."""
        self.stop()
