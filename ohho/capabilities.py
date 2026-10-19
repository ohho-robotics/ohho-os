"""The capability vocabulary.

A robot declares *what it can do*; behaviors ask ``robot.has(...)`` rather than
assuming hardware. This is what lets one behavior — or a whole agent — run across
a wheeled base, a quadruped, a humanoid and an arm, degrading gracefully when a
capability is absent.
"""

from __future__ import annotations

# Base / locomotion
BASE_DRIVE = "base.drive"
BASE_HOLONOMIC = "base.holonomic_drive"  # implies BASE_DRIVE, plus strafing (linear_y)
LEGGED_POSE = "legged.pose"

# Manipulation
MANIPULATION = "manipulation"

# Perception
PERCEPTION_RGB = "perception.rgb"
PERCEPTION_DEPTH = "perception.depth"
PERCEPTION_SCAN = "perception.scan"  # planar range scan (lidar / depth-derived)

ALL: frozenset[str] = frozenset(
    {
        BASE_DRIVE,
        BASE_HOLONOMIC,
        LEGGED_POSE,
        MANIPULATION,
        PERCEPTION_RGB,
        PERCEPTION_DEPTH,
        PERCEPTION_SCAN,
    }
)


def is_known(capability: str) -> bool:
    """True if ``capability`` is part of the recognised vocabulary."""
    return capability in ALL


def capabilities_from_descriptor(document: dict) -> tuple[str, ...]:
    """Capabilities implied by an ohho.descriptor.v1 document.

    A holonomic drive adds strafing. A non-empty ``arms`` list adds
    manipulation. Camera kinds add rgb and depth. ``perception.scan`` is not
    inferred: the descriptor has no lidar field, and this function does not
    invent one.
    """
    if not isinstance(document, dict):
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
    return tuple(found)
