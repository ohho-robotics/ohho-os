"""MAVLink multicopter backend (OHH-305).

Speaks native MAVSDK 4.x (``mavsdk`` on PyPI). There is no ``mavsdk_server``
process. pymavlink is used only where MAVSDK's offboard setters cannot say
the thing the D-02 interface says: a local-frame velocity with a yaw rate,
a ``BODY_FRD`` position, and yaw alone.

Ohho URIs, and the MAVSDK 4 strings they become (``add_any_connection``):

* ``mavlink+udp://HOST:PORT`` → ``udpin://HOST:PORT`` (the ground station
  listens; PX4 SITL sends to ``127.0.0.1:14540``)
* ``mavlink+serial://DEVICE?baud=921600`` → ``serial://DEVICE:921600``

``BODY_FRD`` is ``MAV_FRAME_BODY_NED`` (frame id 8). ``LOCAL_ENU`` is
converted to ``LOCAL_NED`` before it is published. ``LOCAL_NED`` is published
as ``MAV_FRAME_LOCAL_NED`` (frame id 1).

E-stop uses ``ohho.drone.estop.estop_commands``. Disarm is a normal
``MAV_CMD_COMPONENT_ARM_DISARM`` with param2 0, and only when that decision
says disarm. This module never calls ``kill``, ``terminate``, or
``arm_force``.

Experimental: nothing has flown. ``flight_ready`` stays false.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import Callable, Mapping

from ..aerial import (
    Frame,
    convert_vector,
    convert_yaw,
    convert_yaw_rate,
    finite_number,
    require_frame,
)
from ..drone.estop import (
    DISARM_COMMAND,
    HOLD_COMMAND,
    LAND_COMMAND,
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_DO_FLIGHTTERMINATION,
    MAV_FORCE_DISARM_MAGIC,
    command_is_forbidden,
    disarm_allowed,
    estop_commands,
)
from ..drone.mock_link import simulation_target_allowed
from ..drone.telemetry import (
    DRONE_TELEMETRY_KEYS,
    LOCAL_NED,
    DronePose,
    DroneTelemetry,
    DroneVelocity,
    coerce_landed_state,
    gps_fix_from_mav,
    positive_up_m,
)
from ..registry import RobotSpec
from ..schema import ConnectionState, Telemetry, TransportStatus, Velocity
from ..transport import BaseTransport
from .errors import AdapterUnavailable

log = logging.getLogger("ohho.adapters.mavlink")

# MAVLink common.xml. BODY_FRD is MAV_FRAME_BODY_NED, not BODY_FRD (12).
MAV_FRAME_LOCAL_NED = 1
MAV_FRAME_BODY_NED = 8

# SET_POSITION_TARGET_LOCAL_NED type_mask bits.
_IGNORE_X = 1
_IGNORE_Y = 2
_IGNORE_Z = 4
_IGNORE_VX = 8
_IGNORE_VY = 16
_IGNORE_VZ = 32
_IGNORE_AX = 64
_IGNORE_AY = 128
_IGNORE_AZ = 256
_IGNORE_YAW = 1024
_IGNORE_YAW_RATE = 2048

TYPEMASK_VELOCITY_YAW_RATE = (
    _IGNORE_X
    | _IGNORE_Y
    | _IGNORE_Z
    | _IGNORE_AX
    | _IGNORE_AY
    | _IGNORE_AZ
    | _IGNORE_YAW
)
TYPEMASK_POSITION_YAW = (
    _IGNORE_VX
    | _IGNORE_VY
    | _IGNORE_VZ
    | _IGNORE_AX
    | _IGNORE_AY
    | _IGNORE_AZ
    | _IGNORE_YAW_RATE
)
TYPEMASK_YAW_ONLY = (
    _IGNORE_X
    | _IGNORE_Y
    | _IGNORE_Z
    | _IGNORE_VX
    | _IGNORE_VY
    | _IGNORE_VZ
    | _IGNORE_AX
    | _IGNORE_AY
    | _IGNORE_AZ
    | _IGNORE_YAW_RATE
)

DEFAULT_SERIAL_BAUD = 921600
DEFAULT_DISCOVERY_TIMEOUT_S = 10.0

_UDP_SCHEME = "mavlink+udp"
_SERIAL_SCHEME = "mavlink+serial"
_TRUE = frozenset({"1", "true", "yes"})
_FORBIDDEN_NAMES = frozenset(
    {
        "kill",
        "terminate",
        "arm_force",
        "force_disarm",
        "flight_termination",
        "motor_kill",
    }
)
# Plugin methods this backend is allowed to call. kill / terminate / arm_force
# are absent on purpose: kill is MAVSDK's force-disarm (param2 21196).
_ACTION_METHODS = {
    "arm": "arm",
    "disarm": "disarm",
    "takeoff": "takeoff",
    "land": "land",
    "hold": "hold",
    "rtl": "return_to_launch",
}
_MODE_ALIASES = {
    "return_to_launch": "rtl",
    "altctl": "altitude",
    "posctl": "position",
}

_SampleCallback = Callable[[Mapping[str, object]], None]


def _near_magic(value: float) -> bool:
    return abs(value - float(MAV_FORCE_DISARM_MAGIC)) < 1e-3


def reject_forbidden(command: "MavlinkCommand") -> None:
    """Raise if ``command`` is a force-disarm or a flight termination.

    A normal disarm (param1 0, param2 0, command 400) passes. Any other
    disarm, command 185, or a parameter equal to the force-disarm magic
    does not.
    """
    if command.name in _FORBIDDEN_NAMES:
        raise RuntimeError(
            "refusing to emit force-disarm (param2 21196) or flight termination"
        )
    if command.mav_command in (
        MAV_CMD_DO_FLIGHTTERMINATION,
        MAV_FORCE_DISARM_MAGIC,
    ):
        raise RuntimeError(
            "refusing to emit force-disarm (param2 21196) or flight termination"
        )
    for value in (command.param1, command.param2, command.param3):
        if _near_magic(value):
            raise RuntimeError(
                "refusing to emit force-disarm (param2 21196) or flight termination"
            )
    if command.name == "disarm" and (command.param1 != 0.0 or command.param2 != 0.0):
        raise RuntimeError(
            "refusing to emit force-disarm (param2 21196) or flight termination"
        )
    if (
        command.mav_command == MAV_CMD_COMPONENT_ARM_DISARM
        and command.name == "disarm"
        and command.param2 != 0.0
    ):
        raise RuntimeError(
            "refusing to emit force-disarm (param2 21196) or flight termination"
        )


@dataclass(frozen=True, slots=True)
class MavlinkCommand:
    """One command the backend has decided to send. Not yet on the wire."""

    name: str
    mav_command: int | None = None
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    ohho_frame: str | None = None
    mav_frame: str | None = None
    mav_frame_id: int | None = None
    x: float | None = None
    y: float | None = None
    z: float | None = None
    yaw_deg: float | None = None
    yawspeed_deg_s: float | None = None
    yaw_rad: float | None = None
    yaw_rate_rad_s: float | None = None
    alt_m: float | None = None
    type_mask: int | None = None


@dataclass(frozen=True, slots=True)
class MavlinkTarget:
    """Parsed Ohho URI plus the MAVSDK 4 connection string."""

    kind: str
    host: str
    port: int | None
    device: str
    baud: int
    sitl: bool
    mavsdk_url: str

    @property
    def simulation_target(self) -> bool:
        """Loopback UDP only, by ``simulation_target_allowed``.

        Serial is never a simulation target. A UDP host outside loopback
        (and outside docker-local names when ``sitl`` is set) is not either.
        """
        if self.kind != "udp":
            return False
        return simulation_target_allowed(self.host, sitl=self.sitl)

    @property
    def requires_hardware_arming(self) -> bool:
        """Serial and non-loopback UDP need a human. Loopback does not."""
        return not self.simulation_target


def _query_flag(params: Mapping[str, list[str]], key: str) -> bool:
    values = params.get(key, [])
    if not values:
        return False
    return values[-1].strip().lower() in _TRUE


def _query_int(params: Mapping[str, list[str]], key: str) -> int | None:
    values = params.get(key, [])
    if not values or not str(values[-1]).strip():
        return None
    text = str(values[-1]).strip()
    if not text.isdigit():
        raise AdapterUnavailable(f"mavlink URI {key} must be a positive integer")
    number = int(text, 10)
    if number <= 0:
        raise AdapterUnavailable(f"mavlink URI {key} must be a positive integer")
    return number


def _format_host(host: str) -> str:
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return host
    if isinstance(parsed, ipaddress.IPv6Address):
        return f"[{parsed.compressed}]"
    return str(parsed)


def _split_device_baud(device: str, baud: int | None) -> tuple[str, int]:
    if baud is not None:
        return device, baud
    if ":" in device:
        head, tail = device.rsplit(":", 1)
        if head and tail.isdigit():
            number = int(tail, 10)
            if number <= 0:
                raise AdapterUnavailable("mavlink serial baud must be positive")
            return head, number
    return device, DEFAULT_SERIAL_BAUD


def parse_mavlink_uri(uri: str) -> MavlinkTarget:
    """Parse ``mavlink+udp://`` or ``mavlink+serial://`` into a MAVSDK 4 URL.

    Confirmed against MAVSDK 4 connection grammar: ``udpin://[ip][:port]``
    listens, ``serial://[path][:baudrate]`` opens a UART. This function does
    not open a socket.
    """
    parts = urllib.parse.urlsplit(uri.strip())
    scheme = parts.scheme.lower()
    if scheme not in (_UDP_SCHEME, _SERIAL_SCHEME) or "://" not in uri:
        raise AdapterUnavailable(
            "mavlink URI must be mavlink+udp://HOST:PORT or "
            "mavlink+serial://DEVICE?baud=921600"
        )
    query = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
    sitl = _query_flag(query, "sitl")
    if scheme == _UDP_SCHEME:
        host = parts.hostname
        port = parts.port
        if not host or port is None:
            raise AdapterUnavailable(
                "mavlink+udp://HOST:PORT needs a host and a port "
                "(PX4 SITL is mavlink+udp://127.0.0.1:14540)"
            )
        if port < 1 or port > 65535:
            raise AdapterUnavailable("mavlink+udp port is out of range")
        url = f"udpin://{_format_host(host)}:{port}"
        return MavlinkTarget(
            kind="udp",
            host=host,
            port=port,
            device="",
            baud=0,
            sitl=sitl,
            mavsdk_url=url,
        )
    device = parts.path or ""
    baud = _query_int(query, "baud")
    if not device and parts.netloc:
        # netloc keeps COM3's case. hostname would lowercase it.
        netloc = parts.netloc.rsplit("@", 1)[-1]
        head, sep, tail = netloc.rpartition(":")
        if sep and tail.isdigit() and head:
            device = head
            if baud is None:
                baud = int(tail, 10)
        else:
            device = netloc
    device, baud = _split_device_baud(device, baud)
    if not device or device == "/":
        raise AdapterUnavailable("mavlink+serial:// needs a device path")
    return MavlinkTarget(
        kind="serial",
        host="",
        port=None,
        device=device,
        baud=baud,
        sitl=False,
        mavsdk_url=f"serial://{device}:{baud}",
    )


def to_mavsdk_url(uri: str) -> str:
    """MAVSDK 4 ``add_any_connection`` string for an Ohho MAVLink URI."""
    return parse_mavlink_uri(uri).mavsdk_url


def _arm_command() -> MavlinkCommand:
    return MavlinkCommand(
        name="arm",
        mav_command=MAV_CMD_COMPONENT_ARM_DISARM,
        param1=1.0,
        param2=0.0,
    )


def _disarm_command() -> MavlinkCommand:
    """Normal disarm. param2 is 0, which is not the force-disarm magic."""
    command = MavlinkCommand(
        name="disarm",
        mav_command=MAV_CMD_COMPONENT_ARM_DISARM,
        param1=0.0,
        param2=0.0,
    )
    reject_forbidden(command)
    return command


def _from_estop(command_name: str) -> MavlinkCommand:
    """Copy one e-stop step. Refuses anything that decision forbids."""
    table = {
        "land": LAND_COMMAND,
        "hold": HOLD_COMMAND,
        "disarm": DISARM_COMMAND,
    }
    source = table.get(command_name)
    if source is None or command_is_forbidden(source):
        raise RuntimeError(
            "refusing to emit force-disarm (param2 21196) or flight termination"
        )
    built = MavlinkCommand(
        name=source.name,
        mav_command=source.mav_command,
        param1=source.param1,
        param2=source.param2,
        param3=source.param3,
    )
    reject_forbidden(built)
    return built


def map_offboard_velocity(
    vx: float, vy: float, vz: float, yaw_rate: float, frame: Frame
) -> MavlinkCommand:
    """Map a D-02 velocity setpoint onto an explicit MAVLink frame.

    ``BODY_FRD`` is sent unchanged as ``MAV_FRAME_BODY_NED``. ``LOCAL_ENU``
    is converted to ``LOCAL_NED`` first. Yaw rate stays a rate (radians per
    second on the raw message, degrees per second for MAVSDK body velocity).
    It is not written into an absolute yaw field.
    """
    source = require_frame(frame)
    vx = finite_number("vx", vx)
    vy = finite_number("vy", vy)
    vz = finite_number("vz", vz)
    yaw_rate = finite_number("yaw_rate", yaw_rate)
    if source is Frame.BODY_FRD:
        return MavlinkCommand(
            name="offboard_velocity",
            ohho_frame=source.value,
            mav_frame="MAV_FRAME_BODY_NED",
            mav_frame_id=MAV_FRAME_BODY_NED,
            x=vx,
            y=vy,
            z=vz,
            yawspeed_deg_s=math.degrees(yaw_rate),
            yaw_rate_rad_s=yaw_rate,
            type_mask=TYPEMASK_VELOCITY_YAW_RATE,
        )
    north, east, down = vx, vy, vz
    rate = yaw_rate
    if source is Frame.LOCAL_ENU:
        north, east, down = convert_vector(
            vx, vy, vz, source=source, target=Frame.LOCAL_NED
        )
        rate = convert_yaw_rate(yaw_rate, source=source, target=Frame.LOCAL_NED)
    return MavlinkCommand(
        name="offboard_velocity",
        ohho_frame=source.value,
        mav_frame="MAV_FRAME_LOCAL_NED",
        mav_frame_id=MAV_FRAME_LOCAL_NED,
        x=north,
        y=east,
        z=down,
        yaw_rate_rad_s=rate,
        yawspeed_deg_s=math.degrees(rate),
        type_mask=TYPEMASK_VELOCITY_YAW_RATE,
    )


def map_offboard_position(
    x: float, y: float, z: float, yaw: float, frame: Frame
) -> MavlinkCommand:
    """Map a D-02 position setpoint onto an explicit MAVLink frame.

    ``LOCAL_ENU`` is converted to ``LOCAL_NED`` (the plan's publish frame).
    ``BODY_FRD`` is not rotated: the autopilot sees ``MAV_FRAME_BODY_NED``.
    """
    source = require_frame(frame)
    x = finite_number("x", x)
    y = finite_number("y", y)
    z = finite_number("z", z)
    yaw = finite_number("yaw", yaw)
    if source is Frame.BODY_FRD:
        return MavlinkCommand(
            name="offboard_position",
            ohho_frame=source.value,
            mav_frame="MAV_FRAME_BODY_NED",
            mav_frame_id=MAV_FRAME_BODY_NED,
            x=x,
            y=y,
            z=z,
            yaw_deg=math.degrees(yaw),
            yaw_rad=yaw,
            type_mask=TYPEMASK_POSITION_YAW,
        )
    north, east, down, heading = x, y, z, yaw
    if source is Frame.LOCAL_ENU:
        north, east, down = convert_vector(
            x, y, z, source=source, target=Frame.LOCAL_NED
        )
        heading = convert_yaw(yaw, source=source, target=Frame.LOCAL_NED)
    return MavlinkCommand(
        name="offboard_position",
        ohho_frame=source.value,
        mav_frame="MAV_FRAME_LOCAL_NED",
        mav_frame_id=MAV_FRAME_LOCAL_NED,
        x=north,
        y=east,
        z=down,
        yaw_deg=math.degrees(heading),
        yaw_rad=heading,
        type_mask=TYPEMASK_POSITION_YAW,
    )


def map_yaw(yaw: float, frame: Frame) -> MavlinkCommand:
    """Absolute yaw in ``frame``. ``LOCAL_ENU`` is converted to ``LOCAL_NED``."""
    source = require_frame(frame)
    yaw = finite_number("yaw", yaw)
    heading = yaw
    mav_frame = "MAV_FRAME_LOCAL_NED"
    mav_frame_id = MAV_FRAME_LOCAL_NED
    if source is Frame.LOCAL_ENU:
        heading = convert_yaw(yaw, source=source, target=Frame.LOCAL_NED)
    elif source is Frame.BODY_FRD:
        mav_frame = "MAV_FRAME_BODY_NED"
        mav_frame_id = MAV_FRAME_BODY_NED
        heading = yaw
    return MavlinkCommand(
        name="set_yaw",
        ohho_frame=source.value,
        mav_frame=mav_frame,
        mav_frame_id=mav_frame_id,
        yaw_deg=math.degrees(heading),
        yaw_rad=heading,
        type_mask=TYPEMASK_YAW_ONLY,
    )


def position_target_fields(command: MavlinkCommand) -> dict[str, object]:
    """``SET_POSITION_TARGET_LOCAL_NED`` fields for a setpoint MAVSDK cannot say.

    Yaw and yaw rate are radians, which is what that MAVLink message uses.
    Unused channels are zero and masked off by ``type_mask``.
    """
    if command.mav_frame_id is None or command.type_mask is None:
        raise ValueError("setpoint is missing a MAVLink frame")
    yaw = 0.0 if command.yaw_rad is None else command.yaw_rad
    yaw_rate = 0.0 if command.yaw_rate_rad_s is None else command.yaw_rate_rad_s
    if command.name == "offboard_velocity":
        x = y = z = 0.0
        vx = 0.0 if command.x is None else command.x
        vy = 0.0 if command.y is None else command.y
        vz = 0.0 if command.z is None else command.z
    elif command.name == "set_yaw":
        x = y = z = 0.0
        vx = vy = vz = 0.0
    else:
        x = 0.0 if command.x is None else command.x
        y = 0.0 if command.y is None else command.y
        z = 0.0 if command.z is None else command.z
        vx = vy = vz = 0.0
    return {
        "message": "SET_POSITION_TARGET_LOCAL_NED",
        "coordinate_frame": command.mav_frame_id,
        "type_mask": command.type_mask,
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "yaw": yaw,
        "yaw_rate": yaw_rate,
    }


def _blank_sample() -> dict[str, object]:
    return {
        "armed": False,
        "flight_mode": "unknown",
        "landed_state": "unknown",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
        "yaw_rate": 0.0,
        "battery": None,
        "voltage_v": 0.0,
        "current_a": 0.0,
        "gps_fix": "none",
        "failsafe": [],
        "link_quality": None,
        "timestamp": 0.0,
    }


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not math.isfinite(number):
        return default
    return number


def _mode_name(value: object) -> str:
    if value is None:
        return "unknown"
    raw = getattr(value, "name", value)
    text = str(raw).strip().lower()
    if not text:
        return "unknown"
    return _MODE_ALIASES.get(text, text)


class RecordingLink:
    """In-memory MAVSDK stand-in. Records commands and does not open a socket."""

    def __init__(self, sample: Mapping[str, object] | None = None) -> None:
        self.commands: list[MavlinkCommand] = []
        self.urls: list[str] = []
        self.closed = False
        self.sample: dict[str, object] = _blank_sample()
        if sample:
            self.sample.update(dict(sample))
        self._lock = threading.Lock()

    def open(self, url: str) -> None:
        with self._lock:
            self.urls.append(url)
            self.closed = False

    def close(self) -> None:
        with self._lock:
            self.closed = True

    def send(self, command: MavlinkCommand) -> None:
        reject_forbidden(command)
        with self._lock:
            self.commands.append(command)

    def read_sample(self) -> dict[str, object]:
        with self._lock:
            copied = dict(self.sample)
            flags = copied.get("failsafe")
            if isinstance(flags, list):
                copied["failsafe"] = list(flags)
            return copied


def _call_plugin(plugin: object, method: str, *args: object) -> None:
    if method in _FORBIDDEN_NAMES or method in {"kill", "terminate", "arm_force"}:
        raise RuntimeError(
            "refusing to emit force-disarm (param2 21196) or flight termination"
        )
    fn = getattr(plugin, method, None)
    if not callable(fn):
        raise AdapterUnavailable(f"MAVSDK plugin has no {method}()")
    fn(*args)


class MavsdkLink:
    """Native MAVSDK 4 session. Tests inject plugins and never import mavsdk."""

    def __init__(
        self,
        *,
        action: object | None = None,
        offboard: object | None = None,
        raw_sender: Callable[[Mapping[str, object]], None] | None = None,
        opener: Callable[[str], None] | None = None,
        timeout_s: float = DEFAULT_DISCOVERY_TIMEOUT_S,
        on_sample: _SampleCallback | None = None,
    ) -> None:
        self._action = action
        self._offboard = offboard
        self._raw_sender = raw_sender
        self._opener = opener
        self._timeout_s = timeout_s
        self._on_sample = on_sample
        self._offboard_started = False
        self._sdk: object | None = None
        self._sample = _blank_sample()
        self._lock = threading.Lock()

    def open(self, url: str) -> None:
        if self._action is not None or self._opener is not None:
            if self._opener is not None:
                self._opener(url)
            with self._lock:
                self._sample["link_ok"] = True
            return
        self._open_native(url)

    def close(self) -> None:
        sdk = self._sdk
        self._sdk = None
        closer = getattr(sdk, "close", None) if sdk is not None else None
        if callable(closer):
            try:
                closer()
            except Exception as exc:
                log.warning("mavlink: close failed: %r", exc)

    def send(self, command: MavlinkCommand) -> None:
        reject_forbidden(command)
        if command.name in _ACTION_METHODS:
            self._send_action(command)
            return
        if (
            command.name == "offboard_velocity"
            and command.mav_frame_id == MAV_FRAME_BODY_NED
        ):
            self._send_body_velocity(command)
            return
        if (
            command.name == "offboard_position"
            and command.mav_frame_id == MAV_FRAME_LOCAL_NED
        ):
            self._send_ned_position(command)
            return
        if command.name in {"offboard_velocity", "offboard_position", "set_yaw"}:
            self._send_raw(command)
            return
        raise RuntimeError(f"unmapped mavlink command {command.name}")

    def read_sample(self) -> dict[str, object]:
        with self._lock:
            copied = dict(self._sample)
            flags = copied.get("failsafe")
            if isinstance(flags, list):
                copied["failsafe"] = list(flags)
            return copied

    def _send_action(self, command: MavlinkCommand) -> None:
        plugin = self._action
        if plugin is None:
            raise AdapterUnavailable("MAVSDK action plugin is not open")
        method = _ACTION_METHODS[command.name]
        if command.name == "takeoff":
            altitude = 0.0 if command.alt_m is None else command.alt_m
            _call_plugin(plugin, "set_takeoff_altitude", altitude)
        _call_plugin(plugin, method)

    def _send_body_velocity(self, command: MavlinkCommand) -> None:
        plugin = self._offboard
        if plugin is None:
            raise AdapterUnavailable("MAVSDK offboard plugin is not open")
        values = (
            0.0 if command.x is None else command.x,
            0.0 if command.y is None else command.y,
            0.0 if command.z is None else command.z,
            0.0 if command.yawspeed_deg_s is None else command.yawspeed_deg_s,
        )
        setter = getattr(plugin, "set_velocity_body", None)
        if not callable(setter):
            raise AdapterUnavailable("MAVSDK offboard has no set_velocity_body()")
        if getattr(plugin, "wrap_mavsdk_types", False):
            body = _mavsdk_body_velocity(values)
            setter(body)
        else:
            setter(values)
        self._ensure_offboard(plugin)

    def _send_ned_position(self, command: MavlinkCommand) -> None:
        plugin = self._offboard
        if plugin is None:
            raise AdapterUnavailable("MAVSDK offboard plugin is not open")
        values = (
            0.0 if command.x is None else command.x,
            0.0 if command.y is None else command.y,
            0.0 if command.z is None else command.z,
            0.0 if command.yaw_deg is None else command.yaw_deg,
        )
        setter = getattr(plugin, "set_position_ned", None)
        if not callable(setter):
            raise AdapterUnavailable("MAVSDK offboard has no set_position_ned()")
        if getattr(plugin, "wrap_mavsdk_types", False):
            setter(_mavsdk_ned_position(values))
        else:
            setter(values)
        self._ensure_offboard(plugin)

    def _ensure_offboard(self, plugin: object) -> None:
        if self._offboard_started:
            return
        _call_plugin(plugin, "start")
        self._offboard_started = True

    def _send_raw(self, command: MavlinkCommand) -> None:
        fields = position_target_fields(command)
        sender = self._raw_sender
        if sender is None:
            raise AdapterUnavailable(
                "local yaw-rate, body position, and yaw-only setpoints need "
                "the MAVSDK passthrough (pymavlink encodes the message). "
                "Install the [mavlink] extra."
            )
        sender(fields)

    def _open_native(self, url: str) -> None:
        """Connect with native MAVSDK 4. Does not start ``mavsdk_server``."""
        try:
            from mavsdk import ComponentType, Configuration, Mavsdk
            from mavsdk.plugins.action import Action
            from mavsdk.plugins.offboard import Offboard
            from mavsdk.plugins.telemetry import Telemetry
        except ImportError as exc:
            raise AdapterUnavailable(
                "mavlink needs the [mavlink] extra (mavsdk>=4). "
                "Native MAVSDK 4.x does not use mavsdk_server."
            ) from exc
        configuration = Configuration.create_with_component_type(
            ComponentType.GROUND_STATION
        )
        sdk = Mavsdk(configuration)
        result = sdk.add_any_connection(url)
        result_name = getattr(result, "name", None)
        if isinstance(result_name, str) and "success" not in result_name.lower():
            raise AdapterUnavailable(f"MAVSDK refused {url}: {result_name}")
        system = sdk.first_autopilot(self._timeout_s)
        if system is None:
            raise AdapterUnavailable(f"no autopilot discovered on {url}")
        self._sdk = sdk
        self._action = Action(system)
        offboard = Offboard(system)
        setattr(offboard, "wrap_mavsdk_types", True)
        self._offboard = offboard
        self._bind_telemetry(Telemetry(system))
        passthrough = _optional_passthrough(system)
        if passthrough is not None:
            self._raw_sender = _passthrough_sender(passthrough)
        with self._lock:
            self._sample["link_ok"] = True
        self._publish()

    def _bind_telemetry(self, telemetry: object) -> None:
        bindings = (
            ("subscribe_armed", self._on_armed),
            ("subscribe_flight_mode", self._on_mode),
            ("subscribe_landed_state", self._on_landed),
            ("subscribe_position_velocity_ned", self._on_position_velocity),
            ("subscribe_attitude_euler", self._on_attitude),
            ("subscribe_attitude_angular_velocity_body", self._on_yaw_rate),
            ("subscribe_battery", self._on_battery),
            ("subscribe_gps_info", self._on_gps),
        )
        for name, handler in bindings:
            fn = getattr(telemetry, name, None)
            if not callable(fn):
                log.warning("mavlink: telemetry has no %s", name)
                continue
            try:
                fn(handler)
            except Exception as exc:
                log.warning("mavlink: %s failed: %r", name, exc)

    def _update(self, **fields: object) -> None:
        with self._lock:
            self._sample.update(fields)
            self._sample["link_ok"] = True
        self._publish()

    def _publish(self) -> None:
        callback = self._on_sample
        if callback is not None:
            callback(self.read_sample())

    def _on_armed(self, armed: object) -> None:
        self._update(armed=bool(armed))

    def _on_mode(self, mode: object) -> None:
        self._update(flight_mode=_mode_name(mode))

    def _on_landed(self, state: object) -> None:
        self._update(landed_state=_mode_name(getattr(state, "name", state)))

    def _on_position_velocity(self, sample: object) -> None:
        position = getattr(sample, "position", sample)
        velocity = getattr(sample, "velocity", None)
        fields: dict[str, object] = {
            "x": _as_float(getattr(position, "north_m", 0.0)),
            "y": _as_float(getattr(position, "east_m", 0.0)),
            "z": _as_float(getattr(position, "down_m", 0.0)),
        }
        if velocity is not None:
            fields["vx"] = _as_float(getattr(velocity, "north_m_s", 0.0))
            fields["vy"] = _as_float(getattr(velocity, "east_m_s", 0.0))
            fields["vz"] = _as_float(getattr(velocity, "down_m_s", 0.0))
        self._update(**fields)

    def _on_attitude(self, attitude: object) -> None:
        self._update(
            roll=math.radians(_as_float(getattr(attitude, "roll_deg", 0.0))),
            pitch=math.radians(_as_float(getattr(attitude, "pitch_deg", 0.0))),
            yaw=math.radians(_as_float(getattr(attitude, "yaw_deg", 0.0))),
        )

    def _on_yaw_rate(self, rate: object) -> None:
        self._update(yaw_rate=_as_float(getattr(rate, "yaw_rad_s", 0.0)))

    def _on_battery(self, battery: object) -> None:
        remaining = getattr(battery, "remaining_percent", None)
        fraction: float | None
        if isinstance(remaining, bool) or not isinstance(remaining, (int, float)):
            fraction = None
        else:
            number = float(remaining)
            if number > 1.0:
                number = number / 100.0
            fraction = number
        self._update(
            battery=fraction,
            voltage_v=_as_float(getattr(battery, "voltage_v", 0.0)),
            current_a=_as_float(getattr(battery, "current_battery_a", 0.0)),
        )

    def _on_gps(self, info: object) -> None:
        fix = getattr(info, "fix_type", info)
        value = getattr(fix, "value", fix)
        if isinstance(value, int) and not isinstance(value, bool):
            self._update(gps_fix=gps_fix_from_mav(value).value)


def _optional_passthrough(system: object) -> object | None:
    try:
        from mavsdk.plugins.mavlink_passthrough import MavlinkPassthrough
    except ImportError:
        return None
    try:
        return MavlinkPassthrough(system)
    except Exception as exc:
        log.warning("mavlink: passthrough unavailable: %r", exc)
        return None


def _passthrough_sender(
    passthrough: object,
) -> Callable[[Mapping[str, object]], None]:
    def _send(fields: Mapping[str, object]) -> None:
        message = encode_position_target(fields)
        send = getattr(passthrough, "send_message", None)
        if not callable(send):
            raise AdapterUnavailable(
                "MAVSDK MavlinkPassthrough.send_message is not available"
            )
        send(message)

    return _send


def _mavsdk_body_velocity(values: tuple[float, float, float, float]) -> object:
    from mavsdk.plugins.offboard import VelocityBodyYawspeed

    return VelocityBodyYawspeed(*values)


def _mavsdk_ned_position(values: tuple[float, float, float, float]) -> object:
    from mavsdk.plugins.offboard import PositionNedYaw

    return PositionNedYaw(*values)


def encode_position_target(fields: Mapping[str, object]) -> object:
    """Encode one local-NED position target with pymavlink. No socket."""
    try:
        from pymavlink.dialects.v20 import common as dialect
    except ImportError as exc:
        raise AdapterUnavailable(
            "this setpoint needs pymavlink (the [mavlink] extra)"
        ) from exc
    return dialect.MAVLink_set_position_target_local_ned_message(
        0,
        1,
        1,
        int(fields["coordinate_frame"]),  # type: ignore[arg-type]
        int(fields["type_mask"]),  # type: ignore[arg-type]
        float(fields["x"]),  # type: ignore[arg-type]
        float(fields["y"]),  # type: ignore[arg-type]
        float(fields["z"]),  # type: ignore[arg-type]
        float(fields["vx"]),  # type: ignore[arg-type]
        float(fields["vy"]),  # type: ignore[arg-type]
        float(fields["vz"]),  # type: ignore[arg-type]
        0.0,
        0.0,
        0.0,
        float(fields["yaw"]),  # type: ignore[arg-type]
        float(fields["yaw_rate"]),  # type: ignore[arg-type]
    )


def _snapshot_from_sample(sample: Mapping[str, object], *, linked: bool) -> Telemetry:
    landed = coerce_landed_state(sample.get("landed_state", "unknown"), strict=False)
    gps = sample.get("gps_fix", "none")
    if isinstance(gps, int) and not isinstance(gps, bool):
        gps_text = gps_fix_from_mav(gps).value
    else:
        gps_text = "none" if gps is None else str(gps)
    flags = sample.get("failsafe", [])
    failsafe = list(flags) if isinstance(flags, (list, tuple)) else []
    battery = sample.get("battery")
    fraction: float | None
    if isinstance(battery, bool) or not isinstance(battery, (int, float)):
        fraction = None
    else:
        fraction = float(battery)
    quality = sample.get("link_quality")
    if isinstance(quality, bool) or not isinstance(quality, (int, float)):
        link_quality: float | None = 0.0 if not linked else None
    else:
        link_quality = float(quality)
    if "link_ok" in sample:
        link_ok = bool(sample.get("link_ok")) and linked
    else:
        link_ok = linked
    stamp = sample.get("timestamp")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        timestamp = time.time()
    elif not math.isfinite(float(stamp)):
        timestamp = time.time()
    else:
        timestamp = float(stamp)
    drone = DroneTelemetry(
        pose_frame=LOCAL_NED,
        velocity_frame=LOCAL_NED,
        armed=bool(sample.get("armed", False)),
        flight_mode=_mode_name(sample.get("flight_mode")),
        landed_state=landed,
        pose=DronePose(
            x=_as_float(sample.get("x")),
            y=_as_float(sample.get("y")),
            z=_as_float(sample.get("z")),
            roll=_as_float(sample.get("roll")),
            pitch=_as_float(sample.get("pitch")),
            yaw=_as_float(sample.get("yaw")),
        ),
        velocity=DroneVelocity(
            vx=_as_float(sample.get("vx")),
            vy=_as_float(sample.get("vy")),
            vz=_as_float(sample.get("vz")),
            yaw_rate=_as_float(sample.get("yaw_rate")),
        ),
        battery=fraction,
        voltage_v=_as_float(sample.get("voltage_v")),
        current_a=_as_float(sample.get("current_a")),
        gps_fix=gps_text,
        failsafe=failsafe,
        link_ok=link_ok,
        link_quality=link_quality,
        timestamp=timestamp,
    )
    if tuple(drone.as_custom()) != DRONE_TELEMETRY_KEYS:
        raise RuntimeError("mavlink telemetry keys drifted from the drone schema")
    return drone.to_ohho_telemetry(include_planar_odom=True)


def _altitude_up(sample: Mapping[str, object]) -> object:
    """Positive-up metres for the e-stop decision. Missing is not zero."""
    agl = sample.get("agl_m")
    if agl is not None:
        if isinstance(agl, bool) or not isinstance(agl, (int, float)):
            return None
        if not math.isfinite(float(agl)):
            return None
        return float(agl)
    z = sample.get("z")
    if (
        isinstance(z, bool)
        or not isinstance(z, (int, float))
        or not math.isfinite(float(z))
    ):
        return None
    return positive_up_m(float(z), LOCAL_NED)


class MavlinkDroneTransport(BaseTransport):
    """MAVLink flight link. ``link`` injects a mock MAVSDK system for tests."""

    protocol = "mavlink"
    backend = "mavlink"
    backend_id = "mavlink"

    def __init__(
        self,
        spec: RobotSpec,
        uri: str,
        *,
        link: RecordingLink | MavsdkLink | None = None,
        timeout_s: float = DEFAULT_DISCOVERY_TIMEOUT_S,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.uri = uri
        self.target = parse_mavlink_uri(uri)
        self._timeout_s = timeout_s
        self._link: RecordingLink | MavsdkLink | None = link
        self._state = ConnectionState.IDLE
        self._connected_since: float | None = None
        self._latched = False
        self._lock = threading.Lock()

    @property
    def mavsdk_url(self) -> str:
        return self.target.mavsdk_url

    @property
    def simulation_target(self) -> bool:
        return self.target.simulation_target

    @property
    def requires_hardware_arming(self) -> bool:
        return self.target.requires_hardware_arming

    def connect(self) -> TransportStatus:
        with self._lock:
            self._state = ConnectionState.CONNECTING
        self._emit_status(self.status())
        link = self._link
        if link is None:
            link = MavsdkLink(
                timeout_s=self._timeout_s,
                on_sample=self._on_link_sample,
            )
            self._link = link
        try:
            link.open(self.mavsdk_url)
        except AdapterUnavailable:
            with self._lock:
                self._state = ConnectionState.ERROR
            self._emit_status(self.status())
            raise
        except Exception as exc:
            with self._lock:
                self._state = ConnectionState.ERROR
            self._emit_status(self.status())
            raise AdapterUnavailable(f"mavlink connect failed: {exc}") from exc
        with self._lock:
            self._state = ConnectionState.CONNECTED
            self._connected_since = time.time()
        status = self.status()
        self._emit_status(status)
        self._emit_telemetry(self._snapshot())
        return status

    def disconnect(self) -> None:
        link = self._link
        if link is not None:
            try:
                link.close()
            except Exception as exc:
                log.warning("mavlink: disconnect failed: %r", exc)
        with self._lock:
            self._state = ConnectionState.DISCONNECTED
        self._emit_status(self.status())

    def status(self) -> TransportStatus:
        return TransportStatus(
            protocol=self.protocol,
            state=self._state,
            label=f"mavlink · {self.spec.id} · {self.mavsdk_url}",
            connected_since=self._connected_since,
        )

    def read(self) -> Telemetry:
        return self._snapshot()

    def send_velocity(self, vel: Velocity) -> None:
        """Planar body velocity. Zero is hover, not land."""
        vx = finite_number("linear_x", vel.linear_x)
        vy = finite_number("linear_y", vel.linear_y)
        yaw_rate = finite_number("angular_z", vel.angular_z)
        if self._latched or self._state != ConnectionState.CONNECTED:
            return
        self._emit(map_offboard_velocity(vx, vy, 0.0, yaw_rate, Frame.BODY_FRD))

    def send_joint_command(self, name: str, position: float) -> None:
        return None

    def send_flight(self, action: str, **fields: object) -> None:
        """D-02 flight request. ``force`` is ignored. Airborne disarm sends nothing."""
        if action == "arm":
            self._cmd_arm()
        elif action == "disarm":
            self._cmd_disarm()
        elif action == "takeoff":
            self._cmd_takeoff(fields.get("alt_m"))
        elif action == "land":
            self._emit(_from_estop("land"))
        elif action == "rtl":
            self._cmd_mode("rtl")
        elif action == "hold":
            self._emit(_from_estop("hold"))
        elif action == "offboard_position":
            self._cmd_position(fields)
        elif action == "offboard_velocity":
            self._cmd_velocity(fields)
        elif action == "set_yaw":
            self._cmd_yaw(fields)

    def emergency_stop(self) -> None:
        """Land, or hold-then-land, or disarm on the ground. Never force-disarm."""
        self._latched = True
        sample = self._current_sample()
        state = coerce_landed_state(sample.get("landed_state", "unknown"), strict=False)
        altitude = _altitude_up(sample)
        for step in estop_commands(state, altitude):
            if command_is_forbidden(step):
                raise RuntimeError(
                    "refusing to emit force-disarm (param2 21196) or flight termination"
                )
            if step.name == "disarm" and not disarm_allowed(state, altitude):
                continue
            self._emit(_from_estop(step.name))

    def release_stop(self) -> None:
        """Clear the local latch. Does not arm and does not resume offboard."""
        self._latched = False

    def _on_link_sample(self, _sample: Mapping[str, object]) -> None:
        if self._state == ConnectionState.CONNECTED:
            self._emit_telemetry(self._snapshot())

    def _current_sample(self) -> Mapping[str, object]:
        link = self._link
        if link is None:
            return _blank_sample()
        return link.read_sample()

    def _snapshot(self) -> Telemetry:
        linked = self._state == ConnectionState.CONNECTED
        return _snapshot_from_sample(self._current_sample(), linked=linked)

    def _emit(self, command: MavlinkCommand) -> None:
        reject_forbidden(command)
        if command_is_forbidden_name(command.name):
            raise RuntimeError(
                "refusing to emit force-disarm (param2 21196) or flight termination"
            )
        link = self._link
        if link is None or self._state != ConnectionState.CONNECTED:
            return
        link.send(command)

    def _motion_blocked(self) -> bool:
        return self._latched or self._state != ConnectionState.CONNECTED

    def _cmd_arm(self) -> None:
        if self._motion_blocked():
            return
        self._emit(_arm_command())

    def _cmd_disarm(self) -> None:
        """Refuse unless the e-stop rule would disarm. Never a force-disarm."""
        sample = self._current_sample()
        state = coerce_landed_state(sample.get("landed_state", "unknown"), strict=False)
        altitude = _altitude_up(sample)
        if not disarm_allowed(state, altitude):
            return
        self._emit(_disarm_command())

    def _cmd_takeoff(self, alt_m: object) -> None:
        altitude = finite_number("alt_m", alt_m)
        if altitude <= 0.0:
            raise ValueError("alt_m must be positive up, in metres")
        if self._motion_blocked():
            return
        self._emit(
            MavlinkCommand(
                name="takeoff",
                alt_m=altitude,
            )
        )

    def _cmd_mode(self, name: str) -> None:
        if self._motion_blocked():
            return
        self._emit(MavlinkCommand(name=name))

    def _cmd_position(self, fields: Mapping[str, object]) -> None:
        command = map_offboard_position(
            finite_number("x", fields.get("x")),
            finite_number("y", fields.get("y")),
            finite_number("z", fields.get("z")),
            finite_number("yaw", fields.get("yaw")),
            require_frame(fields.get("frame")),
        )
        if self._motion_blocked():
            return
        self._emit(command)

    def _cmd_velocity(self, fields: Mapping[str, object]) -> None:
        command = map_offboard_velocity(
            finite_number("vx", fields.get("vx")),
            finite_number("vy", fields.get("vy")),
            finite_number("vz", fields.get("vz")),
            finite_number("yaw_rate", fields.get("yaw_rate")),
            require_frame(fields.get("frame")),
        )
        if self._motion_blocked():
            return
        self._emit(command)

    def _cmd_yaw(self, fields: Mapping[str, object]) -> None:
        command = map_yaw(
            finite_number("yaw", fields.get("yaw")),
            require_frame(fields.get("frame")),
        )
        if self._motion_blocked():
            return
        self._emit(command)


def command_is_forbidden_name(name: str) -> bool:
    return name in _FORBIDDEN_NAMES


def mavsdk_importable() -> bool:
    """True when native MAVSDK 4 can be imported. Does not connect."""
    import importlib.util

    return importlib.util.find_spec("mavsdk") is not None


__all__ = [
    "DEFAULT_SERIAL_BAUD",
    "MAV_FRAME_BODY_NED",
    "MAV_FRAME_LOCAL_NED",
    "TYPEMASK_POSITION_YAW",
    "TYPEMASK_VELOCITY_YAW_RATE",
    "TYPEMASK_YAW_ONLY",
    "MavlinkCommand",
    "MavlinkDroneTransport",
    "MavlinkTarget",
    "MavsdkLink",
    "RecordingLink",
    "encode_position_target",
    "map_offboard_position",
    "map_offboard_velocity",
    "map_yaw",
    "mavsdk_importable",
    "parse_mavlink_uri",
    "position_target_fields",
    "reject_forbidden",
    "to_mavsdk_url",
]
