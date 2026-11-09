"""Legged safety policy for the quadruped.

The numbers are the caps the gate already applied for this robot. A plugin
policy may only tighten them; these match the class caps, so load accepts them.
"""

from __future__ import annotations

from ohho.safety import SPORT_ACTIONS, SafetyProfile, register_baseline
from ohho.safety_policy import LeggedPolicy

from .spec import ROBOT_ID

# Same object the gate used when this profile lived in core. Tests compare
# identity with ``profile_for``.
GO2_PROFILE = SafetyProfile(
    max_vx=0.5,
    max_vy=0.3,
    max_wz=1.0,
    max_accel_x=1.0,
    max_accel_y=0.6,
    max_accel_wz=2.0,
    require_orientation=True,
    allowlist=SPORT_ACTIONS,
)
register_baseline(ROBOT_ID, GO2_PROFILE)


def policy_for(spec: object) -> LeggedPolicy | None:
    """``LeggedPolicy`` for this robot, or ``None`` for any other id.

    ``None`` keeps the class default. Discovery does not call this.
    """
    if spec is None or getattr(spec, "id", "") != ROBOT_ID:
        return None
    return LeggedPolicy(GO2_PROFILE)
