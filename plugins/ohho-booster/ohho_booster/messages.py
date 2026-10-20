"""Pure mappings from vendor state onto the generic humanoid block.

No vendor SDK is imported here. Names that are not a supported model, and
fall values this module does not recognise, stay fail-closed: an unknown
model is rejected by the transport, and an unknown fall state is ``unknown``
(not upright).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .modes import VENDOR_MODE_MAP

TOPIC_LOW_STATE = "rt/low_state"
TOPIC_FALL_DOWN = "rt/fall_down"
TOPIC_ODOMETER = "rt/odometer_state"
TOPIC_ROBOT_STATES = "rt/robot_states"
TOPIC_JOINT_CTRL = "rt/joint_ctrl"
TOPIC_LOCO_API = "rt/LocoApiTopic"

# Tokens reported by GetRobotInfo.model for this plugin's robot.
# Anything else, including other models from the same SDK, is refused.
SUPPORTED_MODEL_TOKENS = frozenset(
    {
        "t2",
        "booster t2",
        "t2 31dof",
        "t2 pro",
        "t2 edu",
    }
)

_MODE_BY_VALUE = {
    -1: "kUnknown",
    0: "kDamping",
    1: "kPrepare",
    2: "kWalking",
    3: "kCustom",
    4: "kSoccer",
}
_VENDOR_NAMES = frozenset(VENDOR_MODE_MAP) | frozenset(_MODE_BY_VALUE.values())

_FALL_BY_NAME = {
    "is_ready": "upright",
    "is_falling": "falling",
    "has_fallen": "fallen",
    "is_getting_up": "recovering",
    "upright": "upright",
    "falling": "falling",
    "fallen": "fallen",
    "recovering": "recovering",
    "unknown": "unknown",
}
_FALL_BY_VALUE = {0: "upright", 1: "falling", 2: "fallen", 3: "recovering"}

_MISSING = object()


class CommandRefused(RuntimeError):
    """The command was not sent."""

    def __init__(self, action: str, reason: str) -> None:
        self.action = action
        self.reason = reason
        super().__init__(f"{action} refused: {reason}")


class UnknownRobotModel(RuntimeError):
    """Connect stopped because the reported model is not this plugin's robot."""

    def __init__(self, model: object) -> None:
        self.model = model
        super().__init__(f"unknown robot model: {model!r}")


def normalize_model(model: object) -> str:
    """Fold a reported model into the token table. Non-strings are empty."""
    if not isinstance(model, str):
        return ""
    return " ".join(
        model.strip().casefold().replace("_", " ").replace("-", " ").split()
    )


def is_supported_model(model: object) -> bool:
    """True only for this plugin's model tokens."""
    return normalize_model(model) in SUPPORTED_MODEL_TOKENS


def vendor_mode_name(value: object) -> str:
    """Vendor mode name (``kWalking`` and the rest).

    Integers follow ``RobotMode``. ``kSoccer`` is returned so the caller can
    store it, and :func:`ohho_booster.modes.generic_mode` maps it to
    ``UNKNOWN``. Unrecognised values are ``kUnknown``.
    """
    if isinstance(value, str):
        text = value.strip()
        if "." in text:
            text = text.rsplit(".", 1)[-1]
        if text in _VENDOR_NAMES:
            return text
        try:
            return _MODE_BY_VALUE.get(int(text), "kUnknown")
        except ValueError:
            return "kUnknown"
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return vendor_mode_name(name)
    if isinstance(value, bool):
        return "kUnknown"
    if isinstance(value, int):
        return _MODE_BY_VALUE.get(value, "kUnknown")
    if isinstance(value, float):
        return _MODE_BY_VALUE.get(int(value), "kUnknown")
    return "kUnknown"


def fall_state_name(value: object) -> str:
    """Generic fall state. Unrecognised values are ``unknown``."""
    if isinstance(value, str):
        key = value.strip().casefold().replace(" ", "_")
        if "." in key:
            key = key.rsplit(".", 1)[-1]
        return _FALL_BY_NAME.get(key, "unknown")
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return fall_state_name(name)
    if isinstance(value, bool):
        return "unknown"
    if isinstance(value, int):
        return _FALL_BY_VALUE.get(value, "unknown")
    if isinstance(value, float):
        return _FALL_BY_VALUE.get(int(value), "unknown")
    return "unknown"


def field(obj: object, name: str, default: object = _MISSING) -> object:
    """Attribute, mapping entry, or zero-arg getter. ``default`` if absent."""
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        return default
    value = getattr(obj, name, _MISSING)
    if value is _MISSING:
        return default
    return value


def fall_from_message(msg: object) -> str:
    """Fall state carried by an ``rt/fall_down`` sample."""
    raw = field(msg, "fall_down_state")
    if raw is _MISSING:
        raw = field(msg, "fall_state")
    if raw is _MISSING:
        return "unknown"
    return fall_state_name(raw)


def _as_float(value: object, default: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def vec3(value: object) -> tuple[float, float, float] | None:
    """Three finite numbers, or ``None`` when the sample is not that shape."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    if len(value) < 3:
        return None
    parsed: list[float] = []
    for item in value[:3]:
        number = _as_float(item)
        if number is None:
            return None
        parsed.append(number)
    return (parsed[0], parsed[1], parsed[2])


def motor_sample(motor: object) -> dict[str, float] | None:
    """``q``, ``dq``, ``tau`` and ``temp`` from one motor, when each is finite."""
    q = _as_float(field(motor, "q", None), None)
    dq = _as_float(field(motor, "dq", None), None)
    tau = _as_float(field(motor, "tau_est", field(motor, "tau", None)), None)
    temp = _as_float(field(motor, "temperature", field(motor, "temp", None)), None)
    if q is None or dq is None or tau is None:
        return None
    sample = {"q": q, "dq": dq, "tau": tau}
    if temp is not None:
        sample["temp"] = temp
    return sample


def group_table(spec: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Joint groups from ``spec``, or the plugin table when the spec has none."""
    from .joints import JOINT_GROUPS

    groups = getattr(spec, "joint_groups", None) if spec is not None else None
    if (
        isinstance(groups, (str, bytes))
        or not isinstance(groups, Mapping)
        or not groups
    ):
        groups = JOINT_GROUPS
    table: list[tuple[str, tuple[str, ...]]] = []
    for name, joints in groups.items():
        if not isinstance(name, str) or isinstance(joints, (str, bytes)):
            continue
        if not isinstance(joints, Sequence):
            continue
        table.append((name, tuple(str(joint) for joint in joints)))
    return tuple(table)


def joint_count(groups: Sequence[tuple[str, Sequence[str]]]) -> int:
    return sum(len(joints) for _name, joints in groups)


def group_indexes(
    groups: Sequence[tuple[str, Sequence[str]]], group: str
) -> tuple[int, ...] | None:
    """Indexes of ``group`` in the flat joint order, or ``None`` if absent."""
    offset = 0
    for name, joints in groups:
        count = len(joints)
        if name == group:
            return tuple(range(offset, offset + count))
        offset += count
    return None


def _motor_list(msg: object) -> list[object] | None:
    """Serial motors when they fill the 31-name layout, else parallel."""
    from .joints import T2_JOINT_NAMES

    expected = len(T2_JOINT_NAMES)
    for name in ("motor_state_serial", "motor_state_parallel"):
        raw = field(msg, name)
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            continue
        if len(raw) == expected:
            return list(raw)
    return None


def joints_by_group(
    msg: object, groups: Sequence[tuple[str, Sequence[str]]]
) -> dict[str, dict[str, list[float]]]:
    """Per-group ``q``, ``dq``, ``tau`` and ``temp`` from a low-state sample.

    Groups are always present. Channels are filled only when the motor list
    matches the joint-name count and every used sample is finite.
    """
    empty: dict[str, dict[str, list[float]]] = {name: {} for name, _joints in groups}
    motors = _motor_list(msg)
    if motors is None or len(motors) != joint_count(groups):
        return empty
    samples = [motor_sample(motor) for motor in motors]
    if any(sample is None for sample in samples):
        return empty
    offset = 0
    out: dict[str, dict[str, list[float]]] = {}
    for name, joints in groups:
        count = len(joints)
        chunk = samples[offset : offset + count]
        offset += count
        body: dict[str, list[float]] = {
            "q": [float(sample["q"]) for sample in chunk if sample is not None],
            "dq": [float(sample["dq"]) for sample in chunk if sample is not None],
            "tau": [float(sample["tau"]) for sample in chunk if sample is not None],
        }
        if all(sample is not None and "temp" in sample for sample in chunk):
            body["temp"] = [
                float(sample["temp"]) for sample in chunk if sample is not None
            ]
        out[name] = body
    return out
