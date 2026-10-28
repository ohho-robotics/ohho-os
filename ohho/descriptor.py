"""RobotDescriptor v1 adapter.

Emits the ohho.descriptor.v1 document from the registry spec, the safety
profile, and the capability vocabulary. Hard caps are the minimum of those
sources and the module ceilings (0.2 m/s linear, 1.0 rad/s yaw, 0.15 rad
joint step). This module does not raise a cap.

A plugin may return its own document from ``descriptor(spec)``. Core fills
``telemetry.kind`` and ``capability_params`` and validates the result.

``capability_params`` is an optional descriptive field. It copies the spec's
parameters and is not copied into ``drive.caps`` or ``safety.hard``.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from . import capabilities as caps
from .capability_params import validate_capability_params
from .registry import RobotSpec, get_spec
from .robot_spec import RobotAssets, RobotClass, thaw_json
from .safety import PROFILES

DESCRIPTOR_VERSION = "1.0.0"
TELEMETRY_SCHEMA_ID = "ohho.telemetry.v1"

# Ceilings. Do not raise these. A tighter producer clamp stays in force.
MAX_VX_MPS = 0.2
MAX_VY_MPS = 0.2
MAX_WZ_RAD_S = 1.0
MAX_ACCEL_X_MPS2 = 0.5
MAX_ACCEL_Y_MPS2 = 0.5
MAX_ACCEL_WZ_RAD_S2 = 2.0
MAX_JOINT_STEP_RAD = 0.15

_VERSION_RE = re.compile(r"^1\.0\.(0|[1-9]\d*)$")
_ROBOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_./:-]{0,63}$")
_TOPIC_RE = re.compile(r"^/[A-Za-z][A-Za-z0-9_./-]*$")
_URL_RE = re.compile(r"^(package://|https://|http://)\S+$")
_DISPLAY_RE = re.compile(r"^\S(?:.*\S)?$")

_DRIVE_KINDS = {
    "mecanum",
    "differential",
    "ackermann",
    "skid-steer",
    "tracked",
    "quadruped",
    "bipedal",
    "fixed-base",
}
_CAMERA_KINDS = {"rgb", "depth", "bev"}
_SAMPLE_KINDS = {"sim", "real"}
_TELEMETRY_BLOCKS = {
    "base",
    "joints",
    "gripper",
    "imu",
    "battery",
    "nav",
    "policy",
    "rec",
    "sys",
}


def _min_cap(*values: float) -> float:
    return min(float(v) for v in values)


def _drive_kind(spec: RobotSpec) -> str:
    """Drive kind in the v1 enum.

    ``mecanum`` is that enum's holonomic wheeled kind. The enum has no aerial
    kind; flight is described by the ``aerial`` capability parameters, and the
    drive block stays ``fixed-base``.
    """
    robot_class = spec.robot_class
    if robot_class is RobotClass.QUADRUPED:
        return "quadruped"
    if robot_class is RobotClass.HUMANOID:
        return "bipedal"
    if robot_class is RobotClass.ARM or robot_class is RobotClass.AERIAL:
        return "fixed-base"
    if spec.has(caps.BASE_HOLONOMIC):
        return "mecanum"
    return "differential"


def _hard_and_drive_caps(spec: RobotSpec) -> tuple[dict[str, float], dict[str, float]]:
    """Safety block and drive caps. Neither value exceeds the module ceilings."""
    profile = PROFILES.get(spec.id)
    vy_spec = spec.max_lin if spec.has(caps.BASE_HOLONOMIC) else 0.0
    if profile is None:
        vx = _min_cap(spec.max_lin, MAX_VX_MPS)
        vy = _min_cap(vy_spec, MAX_VY_MPS)
        wz = _min_cap(spec.max_ang, MAX_WZ_RAD_S)
        accel_x = MAX_ACCEL_X_MPS2
        accel_y = MAX_ACCEL_Y_MPS2
        accel_wz = MAX_ACCEL_WZ_RAD_S2
        joint_step = MAX_JOINT_STEP_RAD
    else:
        vx = _min_cap(spec.max_lin, profile.max_vx, MAX_VX_MPS)
        vy = _min_cap(vy_spec, profile.max_vy, MAX_VY_MPS)
        wz = _min_cap(spec.max_ang, profile.max_wz, MAX_WZ_RAD_S)
        accel_x = _min_cap(profile.max_accel_x, MAX_ACCEL_X_MPS2)
        accel_y = _min_cap(profile.max_accel_y, MAX_ACCEL_Y_MPS2)
        accel_wz = _min_cap(profile.max_accel_wz, MAX_ACCEL_WZ_RAD_S2)
        joint_step = _min_cap(profile.max_joint_step, MAX_JOINT_STEP_RAD)
    drive_caps = {"vx_mps": vx, "vy_mps": vy, "wz_rad_s": wz}
    hard = {
        "max_vx_mps": vx,
        "max_vy_mps": vy,
        "max_wz_rad_s": wz,
        "max_accel_x_mps2": accel_x,
        "max_accel_y_mps2": accel_y,
        "max_accel_wz_rad_s2": accel_wz,
        "max_joint_step_rad": joint_step,
    }
    return hard, drive_caps


def _urdf_url(spec: RobotSpec) -> str:
    """Use a declared package or http URL. Otherwise a stable locator."""
    assets = spec.assets
    urdf = assets.urdf if isinstance(assets, RobotAssets) else ""
    if _URL_RE.fullmatch(urdf):
        return urdf
    return f"package://ohho/robots/{spec.id}.urdf"


def _plugin_document(spec: RobotSpec) -> dict | None:
    """A plugin descriptor, or ``None`` when the plugin does not supply one."""
    from .plugins import plugin_named

    source = spec.source if isinstance(spec.source, str) else ""
    plugin = plugin_named(source) if source else None
    if plugin is None:
        return None
    custom = plugin.descriptor(spec)
    if custom is None:
        return None
    if isinstance(custom, (str, bytes)) or not isinstance(custom, Mapping):
        raise TypeError("plugin descriptor() must return a mapping or None")
    return copy.deepcopy(dict(custom))


def _generic_document(spec: RobotSpec, kind: str, hard: dict, drive_caps: dict) -> dict:
    fields = ["base"]
    if spec.joint_names or spec.dof_arm:
        fields.append("joints")
    return {
        "v": DESCRIPTOR_VERSION,
        "id": spec.id,
        "model": spec.name,
        "drive": {
            "kind": _drive_kind(spec),
            "holonomic": spec.has(caps.BASE_HOLONOMIC),
            "caps": drive_caps,
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
        "urdf_url": _urdf_url(spec),
        "apriltags": [],
        "telemetry": {
            "schema_id": TELEMETRY_SCHEMA_ID,
            "robot_id": spec.id,
            "kind": kind,
            "fields": fields,
        },
        "safety": {"hard": hard},
    }


def robot_descriptor(robot_id: str, *, kind: str = "real") -> dict:
    """Build a RobotDescriptor for a registered robot.

    A plugin document keeps that robot's topic map, cameras and URDF. Every
    other registered spec gets a document in the same schema.
    ``capability_params`` is copied from the spec. Drive and safety caps
    cannot exceed the module ceilings.
    """
    if kind not in _SAMPLE_KINDS:
        raise ValueError("telemetry kind must be sim or real")
    spec = get_spec(robot_id)
    document = _plugin_document(spec)
    if document is None:
        hard, drive_caps = _hard_and_drive_caps(spec)
        document = _generic_document(spec, kind, hard, drive_caps)
    else:
        telemetry = dict(document.get("telemetry") or {})
        telemetry["kind"] = kind
        document["telemetry"] = telemetry
    document["capability_params"] = thaw_json(spec.capability_params)
    validate_descriptor(document)
    return document


def emit_descriptor(document: Mapping[str, Any]) -> str:
    """Canonical JSON text. Refuses a document that exceeds a ceiling."""
    validate_descriptor(document)
    return json.dumps(document, indent=2) + "\n"


def load_descriptor(path: Union[str, Path]) -> dict:
    """Load a RobotDescriptor JSON document. YAML needs the yaml extra."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised only without pyyaml
            raise RuntimeError(
                "YAML descriptors require pyyaml: pip install 'ohho-os[yaml]'"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    validate_descriptor(data)
    return data


def spec_from_descriptor(document: Mapping[str, Any]) -> RobotSpec:
    """Registry spec implied by a descriptor. Does not register it.

    ``perception.scan`` is not in the descriptor, so it is not invented here.
    ``max_lin`` is the smaller of the two linear hard caps.
    """
    validate_descriptor(document)
    joints = tuple(joint["name"] for arm in document["arms"] for joint in arm["joints"])
    hard = document["safety"]["hard"]
    drive = document["drive"]
    if drive["holonomic"]:
        max_lin = min(hard["max_vx_mps"], hard["max_vy_mps"])
    else:
        max_lin = hard["max_vx_mps"]
    return RobotSpec(
        id=document["id"],
        name=document["model"],
        category="mobile-manipulator" if joints else "wheeled",
        capabilities=caps.capabilities_from_descriptor(document),
        adapter="serial" if joints else "sim",
        dof_base=0 if drive["kind"] == "fixed-base" else 3,
        dof_arm=len(joints),
        max_lin=max_lin,
        max_ang=hard["max_wz_rad_s"],
        joint_names=joints,
        capability_params=document.get("capability_params"),
    )


def validate_descriptor(document: Mapping[str, Any]) -> None:
    """Refuse a document that is incomplete or above a hard ceiling."""
    if not isinstance(document, Mapping):
        raise ValueError("descriptor must be an object")
    _keys(
        document,
        {
            "v",
            "id",
            "model",
            "drive",
            "arms",
            "topics",
            "cameras",
            "urdf_url",
            "apriltags",
            "telemetry",
            "safety",
        },
        "descriptor",
        optional={"capability_params"},
    )
    if "capability_params" in document:
        validate_capability_params(document["capability_params"])
    _match("v", document["v"], _VERSION_RE)
    _match("id", document["id"], _ROBOT_ID_RE)
    _match("model", document["model"], _DISPLAY_RE)
    _validate_drive(document["drive"])
    _validate_arms(document["arms"])
    _validate_topics(document["topics"])
    _validate_cameras(document["cameras"], document["topics"]["images"])
    _match("urdf_url", document["urdf_url"], _URL_RE)
    _validate_tags(document["apriltags"])
    _validate_telemetry(document["telemetry"], document["id"])
    _validate_safety(document["safety"], document["drive"]["caps"])


def _keys(
    obj: Mapping[str, Any],
    required: set[str],
    where: str,
    optional: set[str] | None = None,
) -> None:
    if not isinstance(obj, Mapping):
        raise ValueError(f"{where} must be an object")
    allowed = set(required) if optional is None else set(required) | set(optional)
    extra = set(obj) - allowed
    missing = set(required) - set(obj)
    if missing:
        raise ValueError(f"{where} missing {sorted(missing)}")
    if extra:
        raise ValueError(f"{where} unknown keys {sorted(extra)}")


def _exact_keys(obj: Mapping[str, Any], allowed: set, where: str) -> None:
    _keys(obj, set(allowed), where)


def _match(name: str, value: Any, pattern: re.Pattern) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")


def _number(
    name: str,
    value: Any,
    *,
    ceiling: Optional[float] = None,
    gt: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if gt is not None and number <= gt:
        raise ValueError(f"{name} must be greater than {gt}")
    if gt is None and number < 0:
        raise ValueError(f"{name} must be >= 0")
    if ceiling is not None and number > ceiling:
        raise ValueError(f"{name} exceeds the hard ceiling {ceiling}")
    return number


def _validate_drive(drive: Any) -> None:
    _exact_keys(drive, {"kind", "holonomic", "caps"}, "drive")
    if drive["kind"] not in _DRIVE_KINDS:
        raise ValueError("drive.kind is invalid")
    if not isinstance(drive["holonomic"], bool):
        raise ValueError("drive.holonomic must be a boolean")
    caps_obj = drive["caps"]
    _exact_keys(caps_obj, {"vx_mps", "vy_mps", "wz_rad_s"}, "drive.caps")
    _number("drive.caps.vx_mps", caps_obj["vx_mps"], ceiling=MAX_VX_MPS)
    _number("drive.caps.vy_mps", caps_obj["vy_mps"], ceiling=MAX_VY_MPS)
    _number("drive.caps.wz_rad_s", caps_obj["wz_rad_s"], ceiling=MAX_WZ_RAD_S)


def _validate_arms(arms: Any) -> None:
    if not isinstance(arms, list):
        raise ValueError("arms must be a list")
    for index, arm in enumerate(arms):
        where = f"arms[{index}]"
        _exact_keys(arm, {"name", "joints"}, where)
        _match(f"{where}.name", arm["name"], _NAME_RE)
        joints = arm["joints"]
        if not isinstance(joints, list) or not joints:
            raise ValueError(f"{where}.joints must be a non-empty list")
        for jndex, joint in enumerate(joints):
            jwhere = f"{where}.joints[{jndex}]"
            _exact_keys(joint, {"name", "label", "min", "max", "home"}, jwhere)
            _match(f"{jwhere}.name", joint["name"], _NAME_RE)
            _match(f"{jwhere}.label", joint["label"], _DISPLAY_RE)
            lo = _finite(f"{jwhere}.min", joint["min"])
            hi = _finite(f"{jwhere}.max", joint["max"])
            home = _finite(f"{jwhere}.home", joint["home"])
            if lo > hi:
                raise ValueError(f"{jwhere} min is above max")
            if home < lo or home > hi:
                raise ValueError(f"{jwhere} home is outside min/max")


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_topics(topics: Any) -> None:
    _exact_keys(
        topics,
        {"cmd", "state", "estop", "heartbeat", "telemetry", "images"},
        "topics",
    )
    for key in ("cmd", "state", "estop", "heartbeat", "telemetry"):
        _match(f"topics.{key}", topics[key], _TOPIC_RE)
    images = topics["images"]
    if not isinstance(images, list):
        raise ValueError("topics.images must be a list")
    for index, topic in enumerate(images):
        _match(f"topics.images[{index}]", topic, _TOPIC_RE)


def _validate_cameras(cameras: Any, images: list) -> None:
    if not isinstance(cameras, list):
        raise ValueError("cameras must be a list")
    image_set = set(images)
    for index, camera in enumerate(cameras):
        where = f"cameras[{index}]"
        _exact_keys(camera, {"name", "kind", "frame", "topic"}, where)
        _match(f"{where}.name", camera["name"], _NAME_RE)
        if camera["kind"] not in _CAMERA_KINDS:
            raise ValueError(f"{where}.kind is invalid")
        _match(f"{where}.frame", camera["frame"], _NAME_RE)
        _match(f"{where}.topic", camera["topic"], _TOPIC_RE)
        if camera["topic"] not in image_set:
            raise ValueError(f"{where}.topic is not listed in topics.images")


def _validate_tags(tags: Any) -> None:
    if not isinstance(tags, list):
        raise ValueError("apriltags must be a list")
    for index, tag in enumerate(tags):
        where = f"apriltags[{index}]"
        _exact_keys(tag, {"id", "size", "T_tag_base"}, where)
        if (
            isinstance(tag["id"], bool)
            or not isinstance(tag["id"], int)
            or tag["id"] < 0
        ):
            raise ValueError(f"{where}.id must be an integer >= 0")
        _number(f"{where}.size", tag["size"], ceiling=1.0, gt=0.0)
        matrix = tag["T_tag_base"]
        if not isinstance(matrix, list) or len(matrix) != 16:
            raise ValueError(f"{where}.T_tag_base must have 16 numbers")
        for mndex, value in enumerate(matrix):
            _finite(f"{where}.T_tag_base[{mndex}]", value)


def _validate_telemetry(telemetry: Any, robot_id: str) -> None:
    _exact_keys(telemetry, {"schema_id", "robot_id", "kind", "fields"}, "telemetry")
    if telemetry["schema_id"] != TELEMETRY_SCHEMA_ID:
        raise ValueError("telemetry.schema_id must be ohho.telemetry.v1")
    _match("telemetry.robot_id", telemetry["robot_id"], _ROBOT_ID_RE)
    if telemetry["robot_id"] != robot_id:
        raise ValueError("telemetry.robot_id must match id")
    if telemetry["kind"] not in _SAMPLE_KINDS:
        raise ValueError("telemetry.kind is invalid")
    fields = telemetry["fields"]
    if not isinstance(fields, list):
        raise ValueError("telemetry.fields must be a list")
    for index, field in enumerate(fields):
        if field not in _TELEMETRY_BLOCKS:
            raise ValueError(f"telemetry.fields[{index}] is invalid")


def _validate_safety(safety: Any, drive_caps: Mapping[str, Any]) -> None:
    _exact_keys(safety, {"hard"}, "safety")
    hard = safety["hard"]
    _exact_keys(
        hard,
        {
            "max_vx_mps",
            "max_vy_mps",
            "max_wz_rad_s",
            "max_accel_x_mps2",
            "max_accel_y_mps2",
            "max_accel_wz_rad_s2",
            "max_joint_step_rad",
        },
        "safety.hard",
    )
    ceilings = {
        "max_vx_mps": MAX_VX_MPS,
        "max_vy_mps": MAX_VY_MPS,
        "max_wz_rad_s": MAX_WZ_RAD_S,
        "max_accel_x_mps2": MAX_ACCEL_X_MPS2,
        "max_accel_y_mps2": MAX_ACCEL_Y_MPS2,
        "max_accel_wz_rad_s2": MAX_ACCEL_WZ_RAD_S2,
        "max_joint_step_rad": MAX_JOINT_STEP_RAD,
    }
    for key, ceiling in ceilings.items():
        _number(f"safety.hard.{key}", hard[key], ceiling=ceiling)
    if drive_caps["vx_mps"] > hard["max_vx_mps"]:
        raise ValueError("drive.caps.vx_mps exceeds safety.hard.max_vx_mps")
    if drive_caps["vy_mps"] > hard["max_vy_mps"]:
        raise ValueError("drive.caps.vy_mps exceeds safety.hard.max_vy_mps")
    if drive_caps["wz_rad_s"] > hard["max_wz_rad_s"]:
        raise ValueError("drive.caps.wz_rad_s exceeds safety.hard.max_wz_rad_s")
