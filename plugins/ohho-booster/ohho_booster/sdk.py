"""Vendor SDK session. Imported only when a transport factory runs.

:func:`_load_sdk` is the only place that imports ``booster_robotics_sdk_python``.
Importing this module does not. The wheel is ``booster_robotics_sdk_python``
1.6.3 (Python >= 3.10, manylinux_2_34 x86_64 and aarch64). High-level calls
go through ``B1LocoClient`` on ``rt/LocoApiTopic``. Low-level writes go through
``B1LowCmdPublisher`` (``rt/joint_ctrl``). State is read from ``rt/low_state``,
``rt/fall_down``, ``rt/odometer_state`` and ``rt/robot_states``.

Dances, kicks, zero-torque drag, trajectory record/replay, custom trained
trajectories, and the AI and LUI clients are not constructed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .messages import (
    TOPIC_FALL_DOWN,
    TOPIC_JOINT_CTRL,
    TOPIC_LOCO_API,
    TOPIC_LOW_STATE,
    TOPIC_ODOMETER,
    TOPIC_ROBOT_STATES,
    CommandRefused,
    UnknownRobotModel,
    is_supported_model,
    vendor_mode_name,
)

SDK_DISTRIBUTION = "booster_robotics_sdk_python"
SDK_VERSION = "1.6.3"

_MODE_ACTIONS = {
    "kDamping": "kDamping",
    "kPrepare": "kPrepare",
    "kWalking": "kWalking",
    "kCustom": "kCustom",
}

# RPCs that change robot state. Read-only mode must not call any of these,
# and must not publish on the joint-control topic.
STATE_CHANGING_RPCS = frozenset(
    {
        "ChangeMode",
        "Move",
        "RotateHead",
        "GetUp",
        "LieDown",
        "MoveHandEndEffector",
        "MoveDualHandEndEffector",
        "ControlGripper",
        "ControlDexterousHand",
        "ResetOdometry",
    }
)
READ_ONLY_TOPICS = frozenset({TOPIC_JOINT_CTRL, TOPIC_LOCO_API})


class SdkUnavailable(RuntimeError):
    """The pinned SDK cannot be imported."""


class CommandFailed(RuntimeError):
    """The SDK accepted the call and returned a non-zero status."""

    def __init__(self, what: str, status: object) -> None:
        self.what = what
        self.status = status
        super().__init__(f"{what} failed: {status!r}")


def _load_sdk() -> Any:
    """Import the pinned SDK. This is the only import of that package."""
    try:
        import booster_robotics_sdk_python as sdk
    except ImportError as exc:
        raise SdkUnavailable(
            f"{SDK_DISTRIBUTION}=={SDK_VERSION} is not installed "
            "(Linux x86_64/aarch64 wheel). "
            "pip install 'ohho-booster[booster]' or 'ohho-os[booster]'."
        ) from exc
    version = getattr(sdk, "__version__", None)
    if version not in (None, "", "dev", SDK_VERSION):
        raise SdkUnavailable(
            f"{SDK_DISTRIBUTION} {version} is installed; {SDK_VERSION} is required"
        )
    return sdk


def parse_address(address: str) -> tuple[str, int]:
    """``host`` and optional ``?domain=N`` for ``ChannelFactory.Init``."""
    text = address.strip()
    host, _sep, query = text.partition("?")
    host = host.strip()
    domain = 0
    for item in filter(None, query.split("&")):
        key, _eq, value = item.partition("=")
        if key != "domain":
            continue
        try:
            domain = int(value, 10)
        except ValueError as exc:
            raise ValueError(f"invalid domain {value!r}") from exc
    return host, domain


def open_session(
    address: str, sdk: Any | None = None, *, read_only: bool = False
) -> SdkSession:
    """Build a session. The SDK is imported only when ``sdk`` is omitted.

    ``read_only=True`` still connects and subscribes. State-changing RPCs and
    joint-control publishes raise :class:`CommandRefused` before the client
    is called.
    """
    host, domain = parse_address(address)
    if not host:
        raise ValueError("booster address is empty")
    module = _load_sdk() if sdk is None else sdk
    return SdkSession(host, domain, module, read_only=read_only)


def _ok(what: str, result: object) -> None:
    if isinstance(result, int) and not isinstance(result, bool) and result != 0:
        raise CommandFailed(what, result)


def _read_robot_info(client: Any, sdk: Any) -> Any:
    """Robot info object from ``GetRobotInfo``.

    A client that returns the object is used as-is. A client that fills a
    response argument is used when that type exists and the no-argument call
    does not return an object. This keeps a one-call mock working.
    """
    response_type = getattr(sdk, "GetRobotInfoResponse", None)
    if response_type is None:
        info = client.GetRobotInfo()
        if isinstance(info, int) and not isinstance(info, bool):
            _ok("GetRobotInfo", info)
            raise CommandFailed("GetRobotInfo", info)
        return info
    try:
        info = client.GetRobotInfo()
    except TypeError:
        info = None
    else:
        if info is not None and not isinstance(info, (bool, int)):
            return info
        if isinstance(info, int) and not isinstance(info, bool):
            _ok("GetRobotInfo", info)
    filled = response_type()
    _ok("GetRobotInfo", client.GetRobotInfo(filled))
    return filled


def _millis(seconds: float) -> int:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("duration must be finite and >= 0")
    return int(round(seconds * 1000.0))


def _hand(sdk: Any, side: str) -> object:
    if side == "left":
        return sdk.HandIndex.kLeftHand
    if side == "right":
        return sdk.HandIndex.kRightHand
    raise ValueError("side must be 'left' or 'right'")


def _posture(sdk: Any, pose: Sequence[float]) -> Any:
    if len(pose) != 6:
        raise ValueError("pose must be x, y, z, roll, pitch, yaw")
    posture = sdk.Posture()
    posture.position = sdk.Position(float(pose[0]), float(pose[1]), float(pose[2]))
    posture.orientation = sdk.Orientation(
        float(pose[3]), float(pose[4]), float(pose[5])
    )
    return posture


class SdkSession:
    """One link. ``connect`` reads the model before any subscription or command."""

    def __init__(
        self, host: str, domain: int, sdk: Any, *, read_only: bool = False
    ) -> None:
        self.host = host
        self.domain = domain
        self.read_only = bool(read_only)
        self._sdk = sdk
        self._client: Any = None
        self._subs: list[Any] = []
        self._publisher: Any = None
        self._vendor_mode = "kUnknown"
        self._handler: Callable[[str, object], None] | None = None
        self._info: Any = None

    def connect(self) -> str:
        """Initialise the client and return the reported model.

        A model this plugin does not support raises :class:`UnknownRobotModel`
        before subscribers are created and before any motion call.
        """
        sdk = self._sdk
        sdk.ChannelFactory.Instance().Init(self.domain, self.host)
        client = sdk.B1LocoClient()
        client.Init()
        self._client = client
        info = _read_robot_info(client, sdk)
        self._info = info
        model = getattr(info, "model", None)
        if not is_supported_model(model):
            self.close()
            raise UnknownRobotModel(model)
        self._read_mode()
        return str(model)

    def close(self) -> None:
        for sub in self._subs:
            _close_channel(sub)
        self._subs = []
        publisher = self._publisher
        self._publisher = None
        if publisher is not None:
            _close_channel(publisher)
        self._client = None

    def mode(self) -> str:
        return self._vendor_mode

    def robot_info(self) -> tuple[str, str]:
        """Reported model and firmware version. Empty version if absent."""
        info = self._info
        if info is None:
            raise RuntimeError("session is not connected")
        model = getattr(info, "model", "")
        version = getattr(info, "version", None)
        if version is None:
            version = getattr(info, "firmware_version", None)
        if version is None:
            version = getattr(info, "firmware", None)
        return (
            "" if model is None else str(model),
            "" if version is None else str(version),
        )

    def subscribe(self, callback: Callable[[str, object], None]) -> None:
        """Subscribe to the four state topics. Does not publish."""
        self._handler = callback
        sdk = self._sdk
        pairs = (
            (sdk.B1LowStateSubscriber, TOPIC_LOW_STATE, self._on_low),
            (sdk.B1FallDownStateSubscriber, TOPIC_FALL_DOWN, self._on_fall),
            (sdk.B1OdometerStateSubscriber, TOPIC_ODOMETER, self._on_odom),
            (sdk.B1RobotStatesSubscriber, TOPIC_ROBOT_STATES, self._on_states),
        )
        subs: list[Any] = []
        for cls, _topic, handler in pairs:
            sub = cls(handler)
            sub.InitChannel()
            subs.append(sub)
        self._subs = subs

    def change_mode(self, vendor_mode: str) -> None:
        client = self._writable("ChangeMode")
        if vendor_mode not in _MODE_ACTIONS:
            raise ValueError(f"mode {vendor_mode} is not used")
        mode = getattr(self._sdk.RobotMode, vendor_mode)
        _ok("ChangeMode", client.ChangeMode(mode))
        self._vendor_mode = vendor_mode

    def move(self, vx: float, vy: float, vyaw: float) -> None:
        client = self._writable("Move")
        _ok("Move", client.Move(float(vx), float(vy), float(vyaw)))

    def rotate_head(self, pitch: float, yaw: float) -> None:
        client = self._writable("RotateHead")
        _ok("RotateHead", client.RotateHead(float(pitch), float(yaw)))

    def get_up(self) -> None:
        client = self._writable("GetUp")
        _ok("GetUp", client.GetUp())

    def lie_down(self) -> None:
        client = self._writable("LieDown")
        _ok("LieDown", client.LieDown())

    def move_hand(self, side: str, pose: Sequence[float], duration_s: float) -> None:
        client = self._writable("MoveHandEndEffector")
        _ok(
            "MoveHandEndEffector",
            client.MoveHandEndEffector(
                _posture(self._sdk, pose), _millis(duration_s), _hand(self._sdk, side)
            ),
        )

    def move_dual_hands(
        self,
        left: Sequence[float],
        right: Sequence[float],
        duration_s: float,
    ) -> None:
        client = self._writable("MoveDualHandEndEffector")
        _ok(
            "MoveDualHandEndEffector",
            client.MoveDualHandEndEffector(
                _posture(self._sdk, left),
                _posture(self._sdk, right),
                _millis(duration_s),
            ),
        )

    def control_gripper(self, side: str, command: object) -> None:
        client = self._writable("ControlGripper")
        sdk = self._sdk
        position, force, speed = _gripper_setpoint(command)
        param = sdk.GripperMotionParameter()
        param.position = position
        param.force = force
        param.speed = speed
        _ok(
            "ControlGripper",
            client.ControlGripper(
                param, sdk.GripperControlMode.kPosition, _hand(sdk, side)
            ),
        )

    def control_hand(self, side: str, fingers: Sequence[float]) -> None:
        client = self._writable("ControlDexterousHand")
        sdk = self._sdk
        params = [
            sdk.DexterousFingerParameter(index, int(round(angle)), 500, 500)
            for index, angle in enumerate(fingers)
        ]
        _ok(
            "ControlDexterousHand",
            client.ControlDexterousHand(
                params, _hand(sdk, side), sdk.BoosterHandType.kInspireHand
            ),
        )

    def reset_odometry(self) -> None:
        client = self._writable("ResetOdometry")
        _ok("ResetOdometry", client.ResetOdometry())

    def publish_joints(
        self, cmd_type: str, motors: Sequence[Mapping[str, float]]
    ) -> None:
        """Publish one ``LowCmd`` on ``rt/joint_ctrl``. The caller gates this."""
        self._writable("publish_joints")
        sdk = self._sdk
        if self._publisher is None:
            publisher = sdk.B1LowCmdPublisher()
            publisher.InitChannel()
            self._publisher = publisher
        cmd = sdk.LowCmd()
        kind = cmd_type.strip().upper()
        cmd.cmd_type = (
            sdk.LowCmdType.SERIAL if kind == "SERIAL" else sdk.LowCmdType.PARALLEL
        )
        built = []
        for item in motors:
            motor = sdk.MotorCmd()
            motor.mode = int(item["mode"])
            motor.q = float(item["q"])
            motor.dq = float(item["dq"])
            motor.tau = float(item["tau"])
            motor.kp = float(item["kp"])
            motor.kd = float(item["kd"])
            motor.weight = float(item["weight"])
            built.append(motor)
        cmd.motor_cmd = built
        self._publisher.Write(cmd)

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("session is not connected")
        return self._client

    def _writable(self, rpc: str) -> Any:
        """Return the client, or refuse before any state-changing call."""
        if self.read_only:
            raise CommandRefused(rpc, "read_only")
        return self._require_client()

    def _read_mode(self) -> None:
        client = self._require_client()
        mode = client.GetMode()
        if not isinstance(mode, int):
            self._vendor_mode = vendor_mode_name(getattr(mode, "mode", "kUnknown"))
        status = client.GetStatus()
        if not isinstance(status, int) and hasattr(status, "current_mode"):
            self._vendor_mode = vendor_mode_name(status.current_mode)

    def _on_low(self, msg: object) -> None:
        self._emit(TOPIC_LOW_STATE, msg)

    def _on_fall(self, msg: object) -> None:
        self._emit(TOPIC_FALL_DOWN, msg)

    def _on_odom(self, msg: object) -> None:
        self._emit(TOPIC_ODOMETER, msg)

    def _on_states(self, msg: object) -> None:
        self._emit(TOPIC_ROBOT_STATES, msg)

    def _emit(self, topic: str, msg: object) -> None:
        handler = self._handler
        if handler is not None:
            handler(topic, msg)


def _close_channel(obj: object) -> None:
    close = getattr(obj, "CloseChannel", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        return


def _gripper_setpoint(command: object) -> tuple[int, int, int]:
    """Position, force and speed for ``ControlGripper``.

    ``open`` and ``close`` use the SDK example setpoints. A number is the
    position. Other strings are refused by the caller before this runs; a
    non-number here is a type error.
    """
    if isinstance(command, str):
        key = command.strip().casefold()
        if key == "open":
            return 500, 100, 100
        if key == "close":
            return 100, 300, 200
        raise TypeError("gripper command must be a number, 'open', or 'close'")
    if isinstance(command, bool) or not isinstance(command, (int, float)):
        raise TypeError("gripper command must be a number, 'open', or 'close'")
    if not math.isfinite(float(command)):
        raise ValueError("gripper command must be finite")
    return int(round(float(command))), 100, 100
