"""Unitree DDS adapter — Go2 / quadrupeds.

Bridges the unified ``Robot`` API to Unitree's high-level sport client over
CycloneDDS (``unitree_sdk2py``). The SDK + cyclonedds are heavy and imported
lazily inside ``connect()`` (the ``[unitree]`` extra), so importing this module is
free and the pure mapping functions below are unit-testable without any DDS.

What's complete: the structure, the protocol mapping (velocity → Move,
SportModeState → Telemetry), and the **DDS state subscription** — a reader thread
polls the sport client's state at ~20 Hz and feeds ``_on_state``, which maps the
raw message to ``Telemetry`` and emits it. A ``state_factory`` can be injected for
tests (returns a callable that yields state dicts — no DDS, no hardware).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Tuple

from ..lidar import pointcloud2_to_scan, pointcloud_to_scan
from ..loop_timer import LoopStats, LoopTimer
from ..registry import RobotSpec
from ..schema import (
    ConnectionState,
    JointReading,
    Odometry,
    Scan,
    Telemetry,
    TransportStatus,
    Velocity,
)
from ..transport import BaseTransport
from .errors import AdapterUnavailable

_READER_HZ = 20.0


# ── pure, testable protocol mapping ────────────────────────────────────────────
def velocity_to_move(vel: Velocity) -> Tuple[float, float, float]:
    """Map a unified ``Velocity`` to Unitree ``SportClient.Move(vx, vy, vyaw)``."""
    return (vel.linear_x, vel.linear_y, vel.angular_z)


def sportstate_to_telemetry(state: dict) -> Telemetry:
    """Map a SportModeState-like dict to ``Telemetry``.

    Expected keys (all optional): ``position`` [x,y,z], ``velocity`` [vx,vy,vz],
    ``yaw_speed`` (rad/s), ``imu_rpy`` [roll,pitch,yaw], ``battery`` (0..1),
    ``error_code``, ``body_height`` (m), ``mode``, ``joints`` (name → radians),
    ``posture``, ``scan``, and ``pointcloud`` (body-frame xyz, converted with
    :func:`ohho.lidar.pointcloud_to_scan`). Roll, pitch, error code, body height
    and mode go into ``Telemetry.custom`` only when the state carries them — the
    SafetyGate treats a missing orientation as "no fresh state".
    """
    pos = state.get("position") or [0.0, 0.0, 0.0]
    vel = state.get("velocity") or [0.0, 0.0, 0.0]
    raw_rpy = state.get("imu_rpy")
    has_rpy = isinstance(raw_rpy, (list, tuple)) and len(raw_rpy) >= 3
    rpy = raw_rpy if has_rpy else [0.0, 0.0, 0.0]
    odom = Odometry(
        x=float(pos[0]),
        y=float(pos[1]),
        theta=float(rpy[2]),
        vx=float(vel[0]),
        vy=float(vel[1]),
        omega=float(state.get("yaw_speed", 0.0)),
    )
    custom: dict = {}
    if has_rpy:
        custom["roll"] = float(rpy[0])
        custom["pitch"] = float(rpy[1])
    for key in ("error_code", "body_height", "mode"):
        if key in state:
            custom[key] = state[key]
    posture = _posture_label(state)
    if posture is not None:
        custom["posture"] = posture
    joints = _joint_readings(state.get("joints"))
    scan: Optional[Scan] = state.get("scan")
    if scan is None and "pointcloud" in state:
        scan = pointcloud_to_scan(state["pointcloud"])
    return Telemetry(
        odom=odom,
        joints=joints,
        battery=state.get("battery"),
        scan=scan,
        custom=custom,
    )


def _joint_readings(raw: object) -> list[JointReading]:
    if not isinstance(raw, dict):
        return []
    joints: list[JointReading] = []
    for name, position in raw.items():
        if isinstance(position, (int, float)) and not isinstance(position, bool):
            joints.append(JointReading(str(name), float(position)))
    return joints


def _posture_label(state: dict) -> Optional[str]:
    """Human posture for teleop. None when the state does not say."""
    named = state.get("posture")
    if isinstance(named, str) and named.strip():
        return named.strip()
    mode = state.get("mode")
    if isinstance(mode, int) and not isinstance(mode, bool):
        label = _MODE_POSTURE.get(mode)
        if label:
            return label
    height = state.get("body_height")
    # A zero height is the unset default on some simulators, not a measured lie.
    if isinstance(height, (int, float)) and not isinstance(height, bool) and height > 0:
        if height < 0.12:
            return "lying"
        if height > 0.2:
            return "standing"
    joints = state.get("joints")
    if isinstance(joints, dict):
        calves = [
            float(value)
            for name, value in joints.items()
            if isinstance(name, str)
            and name.endswith("_calf_joint")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]
        if calves:
            average = sum(calves) / len(calves)
            if average < -2.2:
                return "lying"
            if average > -1.6:
                return "standing"
    return None


# Sport-mode actions this adapter can forward (snake_case → SportClient method).
# The SafetyGate allowlist decides which of these callers may actually use.
_SPORT_METHODS = {
    "stand_up": "StandUp",
    "stand_down": "StandDown",
    "balance_stand": "BalanceStand",
    "recovery_stand": "RecoveryStand",
    "stop_move": "StopMove",
    "euler": "Euler",
    "sit": "Sit",
    "rise_sit": "RiseSit",
    "damp": "Damp",
}
_SPORT_WHILE_ESTOPPED = frozenset({"stop_move", "damp"})

# Go2 lowcmd / lowstate motor order in unitree_mujoco (FR, FL, RR, RL).
_GO2_MOTOR_JOINTS = (
    "fr_hip_joint",
    "fr_thigh_joint",
    "fr_calf_joint",
    "fl_hip_joint",
    "fl_thigh_joint",
    "fl_calf_joint",
    "rr_hip_joint",
    "rr_thigh_joint",
    "rr_calf_joint",
    "rl_hip_joint",
    "rl_thigh_joint",
    "rl_calf_joint",
)

_MODE_POSTURE = {
    1: "standing",
    2: "walking",
    6: "sitting",
    7: "lying",
}


StateFactory = Callable[[], Callable[[], Optional[dict]]]
PointCloudFactory = Callable[[], Optional[object]]


class UnitreeDdsTransport(BaseTransport):
    protocol = "dds"

    def __init__(
        self,
        spec: RobotSpec,
        address: str = "",
        *,
        iface: str = "",
        state_factory: Optional[StateFactory] = None,
        pointcloud_factory: Optional[PointCloudFactory] = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        # ``<iface>?domain=<n>`` — e.g. ``dds://lo?domain=1`` for a local simulator.
        address, _sep, query = address.partition("?")
        domain = 0
        for item in filter(None, query.split("&")):
            key, _eq, value = item.partition("=")
            if key == "domain":
                try:
                    domain = int(value)
                except ValueError:
                    raise ValueError(f"invalid DDS domain '{value}'") from None
        self.address = address  # DDS network interface / host hint
        self.domain_id = domain
        self._iface = iface
        self._state_factory = state_factory
        self._pointcloud_factory = pointcloud_factory
        self._client: Optional[object] = None
        self._state_getter: Optional[Callable[[], Optional[dict]]] = None
        self._dds_snapshot: Optional[Callable[[], dict]] = None
        self._lidar_sub: Optional[object] = None
        self._latest = Telemetry(odom=Odometry())
        self._lidar_scan: Optional[Scan] = None
        self._estopped = False
        self._state = ConnectionState.IDLE
        self._connected_since: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._lidar_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._loop_timer = LoopTimer(1.0 / _READER_HZ)

    @property
    def iface(self) -> str:
        """The DDS network interface (empty = SDK default)."""
        return self.address or self._iface

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def connect(self) -> TransportStatus:
        if self._state_factory is not None:
            self._state_getter = self._state_factory()
        else:
            self._connect_dds()
        self._state = ConnectionState.CONNECTED
        self._connected_since = time.time()
        self._start_reader()
        self._start_lidar_reader()
        s = self.status()
        self._emit_status(s)
        return s

    def _connect_dds(self) -> None:
        try:
            from unitree_sdk2py.core.channel import (  # type: ignore
                ChannelFactoryInitialize,
            )
            from unitree_sdk2py.go2.sport.sport_client import SportClient  # type: ignore
        except Exception as e:
            raise AdapterUnavailable(
                "The Unitree DDS adapter needs unitree_sdk2py + cyclonedds: "
                "pip install 'ohho-os[unitree]' (with a working CycloneDDS). "
                "Use transport='sim://' to explore without hardware."
            ) from e
        ChannelFactoryInitialize(self.domain_id, self.iface)
        client = SportClient()
        client.Init()
        self._client = client
        self._dds_snapshot = self._subscribe_dds_state()
        self._state_getter = self._make_sport_state_getter(client)
        self._connect_lidar_sub()

    def _connect_lidar_sub(self) -> None:
        """Subscribe to Go2 / sim DDS lidar (no-op when SDK or topic unavailable)."""
        if self._pointcloud_factory is not None:
            return
        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber  # type: ignore
            from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_  # type: ignore

            from ..lidar import lidar_dds_topic

            sub = ChannelSubscriber(lidar_dds_topic(), PointCloud2_)
            sub.Init()
            self._lidar_sub = sub
        except Exception:
            self._lidar_sub = None

    def _make_sport_state_getter(self, client: object) -> Callable[[], Optional[dict]]:
        """Build a callable that polls the SportClient for a state dict.

        Uses ``client.GetState(SportModeState)`` if available (SDK version
        dependent); falls back to attribute-sniffing on the client itself.
        """

        def getter() -> Optional[dict]:
            merged: dict = {}
            try:
                from unitree_sdk2py.idl.go2.sportmode import SportModeState  # type: ignore

                msg = getattr(client, "GetState", None)
                if callable(msg):
                    state = msg(SportModeState)
                else:
                    state = getattr(client, "_state", None)
                if state is not None:
                    merged.update(
                        state if isinstance(state, dict) else self._sportstate_to_dict(state)
                    )
            except Exception:
                merged = {}
            snapshot = self._dds_snapshot() if self._dds_snapshot is not None else {}
            for key, value in snapshot.items():
                if key == "joints" or key not in merged or merged.get(key) in (None,):
                    merged[key] = value
            return merged or None

        return getter

    @staticmethod
    def _sportstate_to_dict(msg: object) -> dict:
        d = {
            "position": list(getattr(msg, "position", [0.0, 0.0, 0.0])),
            "velocity": list(getattr(msg, "velocity", [0.0, 0.0, 0.0])),
            "yaw_speed": float(getattr(msg, "yaw_speed", 0.0)),
            "battery": getattr(msg, "battery_soc", None),
        }
        rpy = getattr(getattr(msg, "imu_state", None), "rpy", None)
        if rpy is not None:
            d["imu_rpy"] = list(rpy)
        for key in ("error_code", "body_height", "mode"):
            if hasattr(msg, key):
                d[key] = getattr(msg, key)
        return d

    def _subscribe_dds_state(self) -> Callable[[], dict]:
        """Subscribe to sport and low state. Empty snapshot if the IDL is missing."""
        latest: dict[str, dict] = {}
        lock = threading.Lock()
        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber  # type: ignore
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import (  # type: ignore
                LowState_,
                SportModeState_,
            )
        except Exception:
            return lambda: {}

        def on_sport(msg: object) -> None:
            with lock:
                latest["sport"] = self._sportstate_to_dict(msg)

        def on_low(msg: object) -> None:
            with lock:
                latest["low"] = self._lowstate_to_dict(msg)

        try:
            ChannelSubscriber("rt/sportmodestate", SportModeState_).Init(on_sport, 10)
            ChannelSubscriber("rt/lowstate", LowState_).Init(on_low, 10)
        except Exception:
            return lambda: {}

        def snapshot() -> dict:
            with lock:
                sport = dict(latest.get("sport") or {})
                low = dict(latest.get("low") or {})
            merged = {**sport}
            for key, value in low.items():
                if value is not None:
                    merged[key] = value
            return merged

        return snapshot

    @staticmethod
    def _lowstate_to_dict(msg: object) -> dict:
        out: dict = {}
        imu = getattr(msg, "imu_state", None)
        rpy = getattr(imu, "rpy", None) if imu is not None else None
        if rpy is not None:
            out["imu_rpy"] = [float(v) for v in list(rpy)[:3]]
        motors = getattr(msg, "motor_state", None)
        if motors is not None:
            joints: dict[str, float] = {}
            for name, motor in zip(_GO2_MOTOR_JOINTS, list(motors)):
                position = getattr(motor, "q", None)
                if isinstance(position, (int, float)) and not isinstance(position, bool):
                    joints[name] = float(position)
            if joints:
                out["joints"] = joints
        bms = getattr(msg, "bms_state", None)
        soc = getattr(bms, "soc", None) if bms is not None else None
        if isinstance(soc, (int, float)) and not isinstance(soc, bool) and float(soc) > 0:
            out["battery"] = float(soc) / 100.0
        return out

    def _attach_lidar_scan(self, telem: Telemetry) -> Telemetry:
        if self._lidar_scan is None:
            return telem
        return Telemetry(
            odom=telem.odom,
            joints=telem.joints,
            battery=telem.battery,
            scan=self._lidar_scan,
            custom=telem.custom,
            timestamp=telem.timestamp,
        )

    def _on_state(self, state: dict) -> None:
        """Bridge a state dict to telemetry and emit."""
        self._latest = self._attach_lidar_scan(sportstate_to_telemetry(state))
        self._emit_telemetry(self._latest)

    def _on_lidar(self, cloud: object) -> None:
        """Decode a point cloud message and refresh ``Telemetry.scan``."""
        try:
            if isinstance(cloud, (list, tuple)):
                self._lidar_scan = pointcloud_to_scan(cloud)
            else:
                self._lidar_scan = pointcloud2_to_scan(cloud)
        except Exception:
            return
        self._latest = self._attach_lidar_scan(self._latest)
        self._emit_telemetry(self._latest)

    def disconnect(self) -> None:
        self._stop_reader()
        self._stop_lidar_reader()
        sub = self._lidar_sub
        if sub is not None:
            close = getattr(sub, "Close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self._lidar_sub = None
        if self._client is not None:
            stop = getattr(self._client, "StopMove", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
        self._state = ConnectionState.DISCONNECTED
        self._emit_status(self.status())

    def status(self) -> TransportStatus:
        return TransportStatus(
            protocol=self.protocol,
            state=self._state,
            label=f"Unitree DDS · {self.address or 'default'}",
            connected_since=self._connected_since,
        )

    # ── reader thread ─────────────────────────────────────────────────────────
    def loop_stats(self) -> LoopStats:
        """Timing of the state reader thread."""
        return self._loop_timer.stats()

    def _start_reader(self, hz: float = _READER_HZ) -> None:
        if self._thread is not None:
            return
        period = 1.0 / hz
        if self._loop_timer.period_s != period:
            self._loop_timer = LoopTimer(period)
        self._stop.clear()
        timer = self._loop_timer

        def loop() -> None:
            timer.start()
            while not self._stop.is_set():
                timer.begin()
                try:
                    if self._state_getter is not None:
                        state = self._state_getter()
                        if state is not None:
                            self._on_state(state)
                finally:
                    delay = timer.end()
                if self._stop.is_set():
                    return
                if delay > 0.0 and self._stop.wait(delay):
                    return

        self._thread = threading.Thread(
            target=loop, name="ohho-unitree-rx", daemon=True
        )
        self._thread.start()

    def _stop_reader(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=1.0)
            self._thread = None

    def _start_lidar_reader(self, hz: float = 10.0) -> None:
        if self._lidar_thread is not None:
            return
        if self._pointcloud_factory is None and self._lidar_sub is None:
            return
        period = 1.0 / hz

        def loop() -> None:
            while not self._stop.is_set():
                if self._pointcloud_factory is not None:
                    cloud = self._pointcloud_factory()
                    if cloud is not None:
                        self._on_lidar(cloud)
                elif self._lidar_sub is not None:
                    read = getattr(self._lidar_sub, "Read", None)
                    if callable(read):
                        try:
                            msg = read(timeout=period)
                        except TypeError:
                            msg = read()
                        if msg is not None:
                            self._on_lidar(msg)
                time.sleep(period)

        self._lidar_thread = threading.Thread(
            target=loop, name="ohho-unitree-lidar", daemon=True
        )
        self._lidar_thread.start()

    def _stop_lidar_reader(self) -> None:
        t = self._lidar_thread
        if t is not None:
            t.join(timeout=1.0)
            self._lidar_thread = None

    # ── commands / telemetry ──────────────────────────────────────────────────
    def read(self) -> Telemetry:
        return self._latest

    def send_velocity(self, vel: Velocity) -> None:
        if self._estopped or self._client is None:
            return
        vx, vy, vyaw = velocity_to_move(vel)
        move = getattr(self._client, "Move", None)
        if callable(move):
            try:
                move(vx, vy, vyaw)
            except Exception:
                pass

    def send_joint_command(self, name: str, position: float) -> None:
        return  # Go2 has no arm; low-level joint control is a future adapter

    def sport_command(self, name: str, *args: float) -> None:
        """Call a sport-mode action (``"stand_up"``, ``"euler"``, …) on the client.

        Unknown names raise ``ValueError``. While e-stopped only ``stop_move``
        and ``damp`` go through.
        """
        method = _SPORT_METHODS.get(name)
        if method is None:
            raise ValueError(f"unsupported sport command '{name}'")
        if self._client is None:
            return
        if self._estopped and name not in _SPORT_WHILE_ESTOPPED:
            return
        fn = getattr(self._client, method, None)
        if callable(fn):
            fn(*args)

    def emergency_stop(self) -> None:
        self._estopped = True
        if self._client is not None:
            stop = getattr(self._client, "StopMove", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass

    def release_stop(self) -> None:
        self._estopped = False
