"""Shared types for a task-accuracy comparison.

A task evaluator scores a reference policy and an optimised policy on the
same held-out data. The comparison stores both scores, the delta, and
whether that delta is inside the tolerance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

Policy = Callable[..., Any]


@dataclass(frozen=True)
class Frame:
    """One held-out observation and its ground-truth action."""

    observation: tuple[float, ...]
    action: tuple[float, ...]


@dataclass(frozen=True)
class Episode:
    """Frames from one episode. ``seed`` is set when the episode was generated."""

    frames: tuple[Frame, ...]
    seed: int | None = None


@dataclass(frozen=True)
class Comparison:
    """One metric for the reference policy and the optimised policy.

    ``direction`` is ``lower_is_better`` for errors and ``higher_is_better``
    for rates. ``delta`` is always ``optimized - reference``. A value equal
    to the tolerance passes.
    """

    task: str
    metric: str
    reference: float
    optimized: float
    tolerance: float
    direction: str
    unit: str
    samples: int
    per_joint: tuple[dict[str, Any], ...] | None = None
    seeds: tuple[int, ...] | None = None

    @property
    def delta(self) -> float:
        return self.optimized - self.reference

    def passed(self) -> bool:
        """True when the regression is within tolerance."""
        if self.direction == "lower_is_better":
            return self.delta <= self.tolerance
        return (self.reference - self.optimized) <= self.tolerance


@dataclass
class TaskContext:
    """Inputs shared by every evaluator in one task-gate run."""

    reference: Policy
    optimized: Policy
    episodes: tuple[Episode, ...]
    seeds: tuple[int, ...]
    tolerances: dict[str, float]
    detections: tuple[dict[str, Any], ...] | None
    tokens: tuple[dict[str, Any], ...] | None
    sim_steps: int
    robot_id: str
    model: str
    family: str | None
    latencies_ms: list[float] = field(default_factory=list)


def as_floats(values: Sequence[object], label: str) -> list[float]:
    """Copy a sequence of finite numbers."""
    from ohho.parity.errors import ParityError

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ParityError(f"{label} must be a list of numbers.")
    out: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ParityError(f"{label} must be a list of numbers.")
        number = float(item)
        if number != number or number in {float("inf"), float("-inf")}:
            raise ParityError(f"{label} has a non-finite value.")
        out.append(number)
    return out
