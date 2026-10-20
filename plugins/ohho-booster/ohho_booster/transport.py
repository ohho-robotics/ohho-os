"""``booster://`` transport.

High-level calls map onto the generic humanoid actions. ``Move`` is sent only
while the reported mode is ``WALK``. Low-level ``rt/joint_ctrl`` writes happen
only from :meth:`send_humanoid` ``move_joints``, and only in ``CUSTOM``.
``send_joint_command`` does not publish.

The vendor SDK is not imported here. A session is passed in; the factory in
:mod:`ohho_booster.factory` is what imports it. Dances, kicks, zero-torque
drag, trajectory record and replay, custom trained trajectories, and AI/LUI
clients are refused and are not forwarded to the session.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from ohho.humanoid import HumanoidMode, parse_mode
from ohho.humanoid_telemetry import (
    build_humanoid_telemetry,
    telemetry_with_humanoid,
)
from ohho.schema import (
    ConnectionState,
    JointReading,
    Telemetry,
    TransportStatus,
    Velocity,
)
from ohho.transport import BaseTransport

from .messages import (
    TOPIC_FALL_DOWN,
    TOPIC_LOW_STATE,
    TOPIC_ODOMETER,
    TOPIC_ROBOT_STATES,
    CommandRefused,
    UnknownRobotModel,
    fall_from_message,
    field,
    group_indexes,
    group_table,
    is_supported_model,
    joint_count,
    joints_by_group,
    vec3,
    vendor_mode_name,
)
from .modes import generic_mode

_MODE_FOR_ACTION = {
    "prepare": "kPrepare",
    "walk_mode": "kWalking",
    "damp": "kDamping",
    "custom": "kCustom",
}
_REFUSED_PARTS = (
    "dance",
    "kick",
    "shoot",
    "lion",
    "zero_torque",
    "zerotorque",
    "trajectory",
    "trained_traj",
    "trainedtraj",
)
_MOTOR_KEYS = ("mode", "q", "dq", "tau", "kp", "kd", "weight")


def _refused(action: str) -> bool:
    key = action.strip().lower().replace("-", "_")
    compact = key.replace("_", "")
    if any(part in key or part in compact for part in _REFUSED_PARTS):
        return True
    return bool(set(key.split("_")) & {"ai", "lui"})


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


class BoosterTransport(BaseTransport):
    """Humanoid link over a session that speaks the vendor client."""

    protocol = "booster"

    def __init__(self, spec: object, address: str, session: Any) -> None:
        super().__init__()
        self.spec = spec
        self.address = address
        self.host = address.partition("?")[0]
        self._session = session
        self._lock = threading.Lock()
        self._state = ConnectionState.IDLE
        self._connected_since: float | None = None
        self._vendor_mode = "kUnknown"
        self._fall: str | None = None
        self._low: object | None = None
        self._odom_msg: object | None = None
        self._latest = Telemetry()
        self._groups = group_table(spec)

    def connect(self) -> TransportStatus:
        if self._state is ConnectionState.CONNECTED:
            return self.status()
        self._state = ConnectionState.CONNECTING
        try:
            model = self._session.connect()
        except Exception:
            self._state = ConnectionState.ERROR
            self._close_quietly()
            raise
        if not is_supported_model(model):
            self._state = ConnectionState.ERROR
            self._close_quietly()
            raise UnknownRobotModel(model)
        self._session.subscribe(self._on_channel)
        with self._lock:
            self._vendor_mode = str(self._session.mode())
            self._state = ConnectionState.CONNECTED
            self._connected_since = time.time()
        status = self.status()
        self._emit_status(status)
        return status

    def disconnect(self) -> None:
        self._close_quietly()
        with self._lock:
            self._state = ConnectionState.DISCONNECTED
        self._emit_status(self.status())

    def status(self) -> TransportStatus:
        return TransportStatus(
            protocol=self.protocol,
            state=self._state,
            label=self.address,
            connected_since=self._connected_since,
        )

    def read(self) -> Telemetry:
        with self._lock:
            return self._latest

    def send_velocity(self, vel: Velocity) -> None:
        """Send ``Move`` only in ``WALK``. Other modes drop the command."""
        with self._lock:
            walking = self._generic() is HumanoidMode.WALK
            connected = self._state is ConnectionState.CONNECTED
        if not connected or not walking:
            return
        self._session.move(
            float(vel.linear_x), float(vel.linear_y), float(vel.angular_z)
        )

    def send_joint_command(self, name: str, position: float) -> None:
        """Per-joint writes are not the low-level path. Nothing is published."""
        del name, position

    def send_humanoid(self, action: str, *args: object, **fields: object) -> None:
        """Map one generic action onto the session, or refuse it."""
        if not isinstance(action, str) or not action.strip():
            raise CommandRefused(repr(action), "invalid_action")
        name = action.strip().lower()
        if _refused(name):
            raise CommandRefused(name, "not_used")
        if self._state is not ConnectionState.CONNECTED:
            raise CommandRefused(name, "not_connected")
        if name in _MODE_FOR_ACTION:
            self._change_mode(_MODE_FOR_ACTION[name])
            return
        if name == "get_up":
            self._session.get_up()
            return
        if name == "lie_down":
            self._session.lie_down()
            return
        if name == "look":
            self._look(args, fields)
            return
        if name == "move_ee":
            self._move_ee(args, fields)
            return
        if name == "gripper":
            self._gripper(args, fields)
            return
        if name == "hand":
            self._hand(args, fields)
            return
        if name == "move_joints":
            self._move_joints(args, fields)
            return
        if name == "reset_odometry":
            self._session.reset_odometry()
            return
        raise CommandRefused(name, "not_used")

    def emergency_stop(self) -> None:
        """Zero a walk command. Does not damp and does not change mode."""
        self.send_velocity(Velocity())

    def release_stop(self) -> None:
        """The gate owns release. This link has nothing to clear."""

    def _change_mode(self, vendor_mode: str) -> None:
        self._session.change_mode(vendor_mode)
        reported = vendor_mode_name(self._session.mode())
        with self._lock:
            self._vendor_mode = reported

    def _look(self, args: tuple[object, ...], fields: Mapping[str, object]) -> None:
        pitch = _finite(fields.get("pitch", args[0] if args else None))
        yaw = _finite(fields.get("yaw", args[1] if len(args) > 1 else None))
        if pitch is None or yaw is None:
            raise CommandRefused("look", "bad_arguments")
        self._session.rotate_head(pitch, yaw)

    def _move_ee(self, args: tuple[object, ...], fields: Mapping[str, object]) -> None:
        side = fields.get("side", args[0] if args else None)
        pose = fields.get("pose", args[1] if len(args) > 1 else None)
        duration = _finite(fields.get("t", args[2] if len(args) > 2 else 0.0))
        which = _side(side)
        target = _pose(pose)
        if which is None or target is None or duration is None or duration < 0:
            raise CommandRefused("move_ee", "bad_arguments")
        self._session.move_hand(which, target, duration)

    def _gripper(self, args: tuple[object, ...], fields: Mapping[str, object]) -> None:
        side = _side(fields.get("side", args[0] if args else None))
        command = fields.get("cmd", args[1] if len(args) > 1 else None)
        if side is None or command is None:
            raise CommandRefused("gripper", "bad_arguments")
        if isinstance(command, str) and command.strip().casefold() not in {
            "open",
            "close",
        }:
            raise CommandRefused("gripper", "bad_arguments")
        if not isinstance(command, str) and _finite(command) is None:
            raise CommandRefused("gripper", "bad_arguments")
        self._session.control_gripper(side, command)

    def _hand(self, args: tuple[object, ...], fields: Mapping[str, object]) -> None:
        side = _side(fields.get("side", args[0] if args else None))
        fingers = fields.get("fingers", args[1] if len(args) > 1 else None)
        values = _floats(fingers)
        if side is None or values is None:
            raise CommandRefused("hand", "bad_arguments")
        self._session.control_hand(side, values)

    def _move_joints(
        self, args: tuple[object, ...], fields: Mapping[str, object]
    ) -> None:
        """Publish ``rt/joint_ctrl`` only in ``CUSTOM``."""
        with self._lock:
            custom = self._generic() is HumanoidMode.CUSTOM
        if not custom:
            raise CommandRefused("move_joints", "not_custom")
        group = fields.get("group", args[0] if args else None)
        positions = _floats(fields.get("q", args[1] if len(args) > 1 else None))
        kp = fields.get("kp", args[2] if len(args) > 2 else None)
        kd = fields.get("kd", args[3] if len(args) > 3 else None)
        if not isinstance(group, str) or not group.strip() or positions is None:
            raise CommandRefused("move_joints", "bad_arguments")
        indexes = group_indexes(self._groups, group.strip())
        if indexes is None or len(indexes) != len(positions):
            raise CommandRefused("move_joints", "bad_joint_shape")
        kp_v = 0.0 if kp is None else _finite(kp)
        kd_v = 0.0 if kd is None else _finite(kd)
        if kp_v is None or kd_v is None:
            raise CommandRefused("move_joints", "bad_gains")
        kind = str(fields.get("cmd_type", "SERIAL")).strip().upper()
        if kind not in {"SERIAL", "PARALLEL"}:
            raise CommandRefused("move_joints", "bad_cmd_type")
        chosen = set(indexes)
        motors: list[dict[str, float]] = []
        cursor = 0
        for index in range(joint_count(self._groups)):
            if index in chosen:
                motors.append(
                    {
                        "mode": 1.0,
                        "q": positions[cursor],
                        "dq": 0.0,
                        "tau": 0.0,
                        "kp": kp_v,
                        "kd": kd_v,
                        "weight": 1.0,
                    }
                )
                cursor += 1
            else:
                motors.append(
                    {
                        "mode": 0.0,
                        "q": 0.0,
                        "dq": 0.0,
                        "tau": 0.0,
                        "kp": 0.0,
                        "kd": 0.0,
                        "weight": 0.0,
                    }
                )
        for motor in motors:
            if set(motor) != set(_MOTOR_KEYS):
                raise CommandRefused("move_joints", "bad_motor")
        self._session.publish_joints(kind, motors)

    def _on_channel(self, topic: str, msg: object) -> None:
        emit: Telemetry | None = None
        with self._lock:
            if topic == TOPIC_LOW_STATE:
                self._low = msg
            elif topic == TOPIC_ODOMETER:
                self._odom_msg = msg
            elif topic == TOPIC_ROBOT_STATES:
                raw_mode = _maybe(msg, "current_mode")
                if raw_mode is not None:
                    self._vendor_mode = vendor_mode_name(raw_mode)
            elif topic == TOPIC_FALL_DOWN:
                self._fall = fall_from_message(msg)
            else:
                return
            if self._fall is None:
                return
            try:
                self._latest = self._telemetry()
            except (TypeError, ValueError):
                return
            emit = self._latest
        if emit is not None:
            self._emit_telemetry(emit)

    def _telemetry(self) -> Telemetry:
        """Build a block from the latest samples. Caller holds the lock."""
        rpy = (0.0, 0.0, 0.0)
        gyro = (0.0, 0.0, 0.0)
        if self._low is not None:
            imu = field(self._low, "imu_state", None)
            if imu is not None:
                parsed_rpy = vec3(field(imu, "rpy", None))
                parsed_gyro = vec3(field(imu, "gyro", None))
                if parsed_rpy is not None:
                    rpy = parsed_rpy
                if parsed_gyro is not None:
                    gyro = parsed_gyro
        planar = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        if self._odom_msg is not None:
            x = _finite(field(self._odom_msg, "x", 0.0))
            y = _finite(field(self._odom_msg, "y", 0.0))
            yaw = _finite(field(self._odom_msg, "theta", 0.0))
            if x is not None and y is not None and yaw is not None:
                planar = {"x": x, "y": y, "yaw": yaw}
        joints = (
            joints_by_group(self._low, self._groups)
            if self._low is not None
            else {name: {} for name, _joints in self._groups}
        )
        mode = self._generic()
        block = build_humanoid_telemetry(
            mode=mode,
            vendor_mode=self._vendor_mode,
            fall_state=self._fall if self._fall is not None else "unknown",
            base_rpy=rpy,
            base_ang_vel=gyro,
            joints=joints,
            odom=planar,
            battery={"fraction": 0.0, "voltage": 0.0},
            experimental=True,
            spec=self.spec,
        )
        readings: list[JointReading] = []
        if self._low is not None:
            flat_q: list[float] = []
            for _name, body in joints.items():
                flat_q.extend(body.get("q", ()))
            names = [joint for _group, joints_ in self._groups for joint in joints_]
            if len(flat_q) == len(names):
                readings = [
                    JointReading(name=name, position=position)
                    for name, position in zip(names, flat_q)
                ]
        return telemetry_with_humanoid(
            block,
            joints=readings,
            battery=block.battery.fraction,
            timestamp=time.time(),
        )

    def _generic(self) -> HumanoidMode:
        return parse_mode(generic_mode(self._vendor_mode))

    def _close_quietly(self) -> None:
        close = getattr(self._session, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            return


def _maybe(msg: object, name: str) -> object | None:
    if isinstance(msg, Mapping):
        if name not in msg:
            return None
        return msg[name]
    if not hasattr(msg, name):
        return None
    return field(msg, name, None)


def _side(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    which = value.strip().lower()
    if which not in {"left", "right"}:
        return None
    return which


def _pose(value: object) -> tuple[float, ...] | None:
    numbers = _floats(value)
    if numbers is None or len(numbers) != 6:
        return None
    return numbers


def _floats(value: object) -> tuple[float, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    if len(value) == 0:
        return None
    numbers: list[float] = []
    for item in value:
        number = _finite(item)
        if number is None:
            return None
        numbers.append(number)
    return tuple(numbers)
