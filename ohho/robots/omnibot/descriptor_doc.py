"""RobotDescriptor v1 for this robot: topics, cameras, URDF, and arm joints.

Core asks the plugin for this document. The module ceilings stay in
``ohho.descriptor``; nothing here raises a cap.
"""

from __future__ import annotations

from ohho import capabilities as caps
from ohho.registry import RobotSpec

from .policy import OMNIBOT_PROFILE

_JOINT_LABELS = {
    "arm_shoulder_pan": "Pan",
    "arm_shoulder_lift": "Lift",
    "arm_elbow_flex": "Elbow",
    "arm_wrist_flex": "Wrist",
    "arm_wrist_roll": "Roll",
    "arm_gripper": "Grip",
}

_TOPICS = {
    "cmd": "/cmd_vel",
    "state": "/odom",
    "estop": "/emergency_stop",
    "heartbeat": "/ohho/heartbeat",
    "telemetry": "/ohho/telemetry",
    "images": [
        "/camera/front/image_raw",
        "/camera/wrist/image_raw",
        "/camera/depth/image_raw",
        "/camera/bev/image_raw",
    ],
}
_CAMERAS = (
    {
        "name": "front",
        "kind": "rgb",
        "frame": "front_camera_link",
        "topic": "/camera/front/image_raw",
    },
    {
        "name": "wrist",
        "kind": "rgb",
        "frame": "wrist_camera_link",
        "topic": "/camera/wrist/image_raw",
    },
    {
        "name": "depth",
        "kind": "depth",
        "frame": "depth_camera_optical_frame",
        "topic": "/camera/depth/image_raw",
    },
    {
        "name": "bev",
        "kind": "bev",
        "frame": "bev_frame",
        "topic": "/camera/bev/image_raw",
    },
)
_URDF = "package://omnibot_description/urdf/omnibot.urdf.xacro"
_FIELDS = ("base", "joints", "imu")


def _min_cap(*values: float) -> float:
    return min(float(value) for value in values)


def omnibot_document(spec: RobotSpec) -> dict:
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

    profile = OMNIBOT_PROFILE
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
    joints = []
    for name in spec.joint_names:
        if name not in profile.joint_limits:
            raise ValueError(f"joint {name} has no safety limit")
        lo, hi = profile.joint_limits[name]
        if name not in _JOINT_LABELS:
            raise ValueError(f"joint {name} has no label")
        joints.append(
            {
                "name": name,
                "label": _JOINT_LABELS[name],
                "min": lo,
                "max": hi,
                "home": 0.0,
            }
        )
    return {
        "v": DESCRIPTOR_VERSION,
        "id": spec.id,
        "model": spec.name,
        "drive": {
            "kind": "mecanum",
            "holonomic": spec.has(caps.BASE_HOLONOMIC),
            "caps": {"vx_mps": vx, "vy_mps": vy, "wz_rad_s": wz},
        },
        "arms": [{"name": "so101", "joints": joints}] if joints else [],
        "topics": {
            "cmd": _TOPICS["cmd"],
            "state": _TOPICS["state"],
            "estop": _TOPICS["estop"],
            "heartbeat": _TOPICS["heartbeat"],
            "telemetry": _TOPICS["telemetry"],
            "images": list(_TOPICS["images"]),
        },
        "cameras": [dict(camera) for camera in _CAMERAS],
        "urdf_url": _URDF,
        "apriltags": [],
        "telemetry": {
            "schema_id": TELEMETRY_SCHEMA_ID,
            "robot_id": spec.id,
            "kind": "real",
            "fields": list(_FIELDS),
        },
        "safety": {"hard": hard},
    }
