"""Named QoS profiles for runtime manifests.

The set is small on purpose. A later change can add profiles. A deadline
miss is a fault for that later layer: the last command is not replayed.
These profiles do not change safety-gate watchdogs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class Reliability(str, Enum):
    """Delivery guarantee for one named profile."""

    RELIABLE = "reliable"
    BEST_EFFORT = "best_effort"


@dataclass(frozen=True)
class QosProfile:
    """Reliability, history depth, and an optional deadline."""

    name: str
    reliability: Reliability
    history_depth: int
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("QoS profile name must be a non-empty string")
        if not isinstance(self.reliability, Reliability):
            object.__setattr__(self, "reliability", Reliability(self.reliability))
        depth = self.history_depth
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError(
                f"QoS profile {self.name!r} history_depth must be an int >= 1"
            )
        if self.deadline_s is None:
            return
        if isinstance(self.deadline_s, bool) or not isinstance(
            self.deadline_s, (int, float)
        ):
            raise ValueError(
                f"QoS profile {self.name!r} deadline_s must be a number > 0"
            )
        deadline = float(self.deadline_s)
        if not math.isfinite(deadline) or deadline <= 0:
            raise ValueError(f"QoS profile {self.name!r} deadline_s must be > 0")
        object.__setattr__(self, "deadline_s", deadline)


QOS_PROFILES: Mapping[str, QosProfile] = MappingProxyType(
    {
        "command": QosProfile(
            "command",
            Reliability.RELIABLE,
            history_depth=1,
            deadline_s=0.1,
        ),
        "state": QosProfile(
            "state",
            Reliability.RELIABLE,
            history_depth=1,
            deadline_s=0.5,
        ),
        "telemetry": QosProfile(
            "telemetry",
            Reliability.BEST_EFFORT,
            history_depth=5,
            deadline_s=1.0,
        ),
        "sensor": QosProfile(
            "sensor",
            Reliability.BEST_EFFORT,
            history_depth=1,
            deadline_s=None,
        ),
    }
)


__all__ = ["QOS_PROFILES", "QosProfile", "Reliability"]
