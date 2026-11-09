"""RobotDescriptor v1 for this quadruped.

Core asks the plugin for this document and then fills ``telemetry.kind`` and
``capability_params``. The module ceilings stay in ``ohho.descriptor``.
"""

from __future__ import annotations

from ohho import capabilities as caps
from ohho.registry import RobotSpec

from .policy import GO2_PROFILE


def _min_cap(*values: float) -> float:
    return min(float(value) for value in values)


def unitree_document(spec: RobotSpec) -> dict:
    """Descriptor body. ``telemetry.kind`` defaults to ``real``; core sets it."""
    from ohho.descriptor import (
        DESCRIPTOR_VERSION,
        MAX_ACCEL_WZ_RAD_S2,
        MAX_ACCEL_X_MPS2,
        MAX_ACCEL_Y_MPS2,
        MAX_JOINT_STEP_RAD,
        MAX_VX_MPS,
        MAX_VY_MPS,
        MAX_WZ_RAD_S,
        TELEMETRY_SCHEMA_ID,
    )

    profile = GO2_PROFILE
    vy_spec = spec.max_lin if spec.has(caps.BASE_HOLONOMIC) else 0.0
    vx = _min_cap(spec.max_lin, profile.max_vx, MAX_VX_MPS)
    vy = _min_cap(vy_spec, profile.max_vy, MAX_VY_MPS)
    wz = _min_cap(spec.max_ang, profile.max_wz, MAX_WZ_RAD_S)
    hard = {
        "max_vx_mps": vx,
        "max_vy_mps": vy,
        "max_wz_rad_s": wz,
        "max_accel_x_mps2": _min_cap(profile.max_accel_x, MAX_ACCEL_X_MPS2),
        "max_accel_y_mps2": _min_cap(profile.max_accel_y, MAX_ACCEL_Y_MPS2),
        "max_accel_wz_rad_s2": _min_cap(profile.max_accel_wz, MAX_ACCEL_WZ_RAD_S2),
        "max_joint_step_rad": _min_cap(profile.max_joint_step, MAX_JOINT_STEP_RAD),
    }
    fields = ["base"]
    if spec.joint_names or spec.dof_arm:
        fields.append("joints")
    return {
        "v": DESCRIPTOR_VERSION,
        "id": spec.id,
        "model": spec.name,
        "drive": {
            "kind": "quadruped",
            "holonomic": spec.has(caps.BASE_HOLONOMIC),
            "caps": {"vx_mps": vx, "vy_mps": vy, "wz_rad_s": wz},
        },
        "arms": [],
        "topics": {
            "cmd": "/cmd_vel",
            "state": "/odom",
            "estop": "/emergency_stop",
            "heartbeat": "/ohho/heartbeat",
            "telemetry": "/ohho/telemetry",
            "images": [],
        },
        "cameras": [],
        "urdf_url": f"package://ohho/robots/{spec.id}.urdf",
        "apriltags": [],
        "telemetry": {
            "schema_id": TELEMETRY_SCHEMA_ID,
            "robot_id": spec.id,
            "kind": "real",
            "fields": fields,
        },
        "safety": {"hard": hard},
    }
