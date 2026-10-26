"""In-process targets for a simulated service set.

Calling one does not connect to a robot, send a command, or arm.
Dev mode runs the target on the supervisor thread. ``ready`` returns and
the dev service becomes ready. ``fail`` raises so start fails.
"""

from __future__ import annotations


def ready() -> None:
    """Sim target that lets the dev service become ready."""
    return None


def fail() -> None:
    """Sim target that fails start. The supervisor records the failure."""
    raise RuntimeError("sim service failed")


__all__ = ["fail", "ready"]
