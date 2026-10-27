"""The capability vocabulary.

A robot declares *what it can do*; behaviors ask ``robot.has(...)`` rather than
assuming hardware. This is what lets one behavior — or a whole agent — run across
a wheeled base, a quadruped, a humanoid, an arm, or an aircraft, degrading
gracefully when a capability is absent.

``has`` stays a string check. Typed parameters for the names that carry them
live in ``ohho.capability_params`` and are stored on ``RobotSpec.capability_params``.
"""

from __future__ import annotations

from collections.abc import Mapping

# Base / locomotion. Parameters live in ohho.capability_params.
BASE_DRIVE = "base.drive"
BASE_HOLONOMIC = "base.holonomic_drive"  # implies BASE_DRIVE, plus strafing (linear_y)
LEGGED_POSE = "legged.pose"

# Humanoid
HUMANOID_LOCOMOTION = "humanoid.locomotion"
HUMANOID_POSTURE = "humanoid.posture"
WHOLE_BODY_JOINT = "whole_body.joint"

# Aerial. Planar ``base.drive`` does not imply this.
AERIAL = "aerial"

# Manipulation
MANIPULATION = "manipulation"
MANIPULATION_DUAL_ARM = "manipulation.dual_arm"
END_EFFECTOR_GRIPPER = "end_effector.gripper"
END_EFFECTOR_DEX_HAND = "end_effector.dex_hand"
HEAD_GAZE = "head.gaze"

# Perception. Strings only: the descriptor has no typed parameter schema for these.
PERCEPTION_RGB = "perception.rgb"
PERCEPTION_DEPTH = "perception.depth"
PERCEPTION_SCAN = "perception.scan"  # planar range scan (lidar / depth-derived)

ALL: frozenset[str] = frozenset(
    {
        BASE_DRIVE,
        BASE_HOLONOMIC,
        LEGGED_POSE,
        AERIAL,
        HUMANOID_LOCOMOTION,
        HUMANOID_POSTURE,
        WHOLE_BODY_JOINT,
        MANIPULATION,
        MANIPULATION_DUAL_ARM,
        END_EFFECTOR_GRIPPER,
        END_EFFECTOR_DEX_HAND,
        HEAD_GAZE,
        PERCEPTION_RGB,
        PERCEPTION_DEPTH,
        PERCEPTION_SCAN,
    }
)


def is_known(capability: str) -> bool:
    """True if ``capability`` is part of the recognised vocabulary."""
    return capability in ALL


def capabilities_from_descriptor(document: Mapping[str, object]) -> tuple[str, ...]:
    """Capabilities implied by an ohho.descriptor.v1 document.

    A holonomic drive adds strafing. A non-empty ``arms`` list adds
    manipulation. Camera kinds add rgb and depth. Names present under the
    optional ``capability_params`` object are included when they are known.
    ``perception.scan`` is not inferred from cameras: the descriptor has no
    lidar field, and this function does not invent one.
    """
    if not isinstance(document, Mapping):
        raise TypeError("descriptor must be an object")
    found: list[str] = []
    drive = document.get("drive")
    if isinstance(drive, dict) and drive.get("kind") != "fixed-base":
        found.append(BASE_DRIVE)
        if drive.get("holonomic") is True:
            found.append(BASE_HOLONOMIC)
    arms = document.get("arms")
    if isinstance(arms, list) and arms:
        found.append(MANIPULATION)
    saw_rgb = False
    saw_depth = False
    cameras = document.get("cameras")
    if isinstance(cameras, list):
        for camera in cameras:
            if not isinstance(camera, dict):
                continue
            if camera.get("kind") == "rgb":
                saw_rgb = True
            elif camera.get("kind") == "depth":
                saw_depth = True
    if saw_rgb:
        found.append(PERCEPTION_RGB)
    if saw_depth:
        found.append(PERCEPTION_DEPTH)
    params = document.get("capability_params")
    if isinstance(params, Mapping):
        for name in params:
            if isinstance(name, str) and is_known(name) and name not in found:
                found.append(name)
    return tuple(found)
