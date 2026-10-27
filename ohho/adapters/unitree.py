"""Unitree DDS adapter — Go2 / quadrupeds.

Bridges the unified ``Robot`` API to Unitree's high-level sport client over
CycloneDDS (``unitree_sdk2py``). The SDK + cyclonedds are heavy and imported
lazily inside ``connect()`` (the ``[unitree]`` extra), so importing this module is
free and the pure mapping functions below are unit-testable without any DDS.

State arrives on DDS topics ``rt/sportmodestate`` (pose, IMU, gait) and
``rt/lowstate`` (joints, battery). A ``state_factory`` or ``subscriber_factory``
can be injected for tests (no DDS, no hardware).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Optional, Tuple

from ..loop_timer import LoopStats, LoopTimer
from ..registry import RobotSpec
from ..schema import (
    ConnectionState,
    JointReading,
    Odometry,
    Telemetry,
    TransportStatus,
    Velocity,
)
from ..transport import BaseTransport
from .errors import AdapterUnavailable

_log = logging.getLogger(__name__)

# Go2 motor order: FR, FL, RR, RL × (hip, thigh, calf) — indices 0..11 in LowState.
_GO2_JOINT_NAMES: Tuple[str, ...] = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)

_DEFAULT_SIM_IFACE = "lo"
_DEFAULT_SIM_DOMAIN = 1
_STALE_STATE_SEC = 1.0
_MSG_RATE_WINDOW_SEC = 5.0
_SPORT_RPC_TIMEOUT_SEC = 1.0
_READER_HZ = 20.0


def parse_dds_address(address: str, iface_kw: str = "") -> Tuple[str, int]:
    """Parse ``<iface>?domain=<n>`` from the part after ``dds://``.

    Bare ``dds://`` (empty address) defaults to ``lo`` + domain ``1`` (unitree_mujoco).
    An explicit iface without ``domain=`` uses domain ``0`` (real robot / safety arming).
    """
    iface_part, _sep, query = address.partition("?")
    explicit_iface = iface_part.strip()
    iface = explicit_iface or iface_kw.strip() or _DEFAULT_SIM_IFACE
    domain: Optional[int] = None
    for item in filter(None, query.split("&")):
        key, _eq, value = item.partition("=")
        if key == "domain":
            try:
                domain = int(value)
            except ValueError:
                raise ValueError(f"invalid DDS domain '{value}'") from None
    if domain is not None:
        return iface, domain
    if not explicit_iface and not iface_kw.strip():
        return _DEFAULT_SIM_IFACE, _DEFAULT_SIM_DOMAIN
    return iface, 0


def dds_uri_is_sim_only(uri: Optional[str]) -> bool:
    """True for ``dds://`` URIs that target local sim (``lo`` + domain ``1``)."""
    if not uri:
        return False
    scheme, _sep, address = uri.partition("://")
    if scheme.lower() not in ("dds", "unitree-dds"):
        return False
    iface, domain = parse_dds_address(address)
    return iface == _DEFAULT_SIM_IFACE and domain == _DEFAULT_SIM_DOMAIN


def dds_uri_is_hardware(uri: Optional[str]) -> bool:
    """True for ``dds://`` URIs that are not the local sim-only pair."""
    if not uri:
        return False
    scheme, _sep, _address = uri.partition("://")
    if scheme.lower() not in ("dds", "unitree-dds"):
        return False
    return not dds_uri_is_sim_only(uri)


# ── pure, testable protocol mapping ────────────────────────────────────────────
def velocity_to_move(vel: Velocity) -> Tuple[float, float, float]:
    """Map a unified ``Velocity`` to Unitree ``SportClient.Move(vx, vy, vyaw)``."""
    return (vel.linear_x, vel.linear_y, vel.angular_z)


def sportstate_to_telemetry(state: dict) -> Telemetry:
    """Map a merged sport/low state dict to ``Telemetry``.

    Expected keys (all optional): ``position`` [x,y,z], ``velocity`` [vx,vy,vz],
    ``yaw_speed`` (rad/s), ``imu_rpy`` [roll,pitch,yaw], ``battery`` (0..1),
    ``joints`` [{name, position, velocity?}, …], ``error_code``, ``body_height``,
    ``mode``, ``gait_type``, ``foot_force``. Roll/pitch and fault fields go into
    ``Telemetry.custom`` only when present — the SafetyGate treats missing
    orientation as "no fresh state".
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
    for key in ("error_code", "body_height", "mode", "gait_type", "foot_force"):
        if key in state:
            custom[key] = state[key]
    joints_in = state.get("joints") or []
    joints: list[JointReading] = []
    for item in joints_in:
        if isinstance(item, JointReading):
            joints.append(item)
        elif isinstance(item, dict) and "name" in item:
            joints.append(
                JointReading(
                    name=str(item["name"]),
                    position=float(item["position"]),
                    velocity=(
                        float(item["velocity"])
                        if item.get("velocity") is not None
                        else None
                    ),
                )
            )
    ts = state.get("msg_timestamp")
    if ts is None:
        ts = time.time()
    return Telemetry(
        odom=odom,
        joints=joints,
        battery=state.get("battery"),
        custom=custom,
        timestamp=float(ts),
    )


def _stamp_to_sec(stamp: object) -> Optional[float]:
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    if sec is not None:
        nsec = float(getattr(stamp, "nanosec", 0) or 0)
        return float(sec) + nsec / 1e9
    return None


def _seq_to_list(value: object, size: Optional[int] = None) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    try:
        out = [float(value[i]) for i in range(len(value))]  # type: ignore[arg-type]
        return out
    except (TypeError, IndexError):
        pass
    if size is not None:
        try:
            return [float(value[i]) for i in range(size)]
        except (TypeError, IndexError):
            return []
    return []


def sport_message_to_dict(msg: object) -> dict:
    """Extract high-level sport fields from a ``SportModeState_`` sample."""
    d: dict = {
        "position": _seq_to_list(getattr(msg, "position", None), 3),
        "velocity": _seq_to_list(getattr(msg, "velocity", None), 3),
        "yaw_speed": float(getattr(msg, "yaw_speed", 0.0)),
    }
    rpy = getattr(getattr(msg, "imu_state", None), "rpy", None)
    if rpy is not None:
        d["imu_rpy"] = _seq_to_list(rpy, 3)
    ts = _stamp_to_sec(getattr(msg, "stamp", None))
    if ts is not None:
        d["msg_timestamp"] = ts
    for key in ("error_code", "body_height", "mode", "gait_type"):
        if hasattr(msg, key):
            d[key] = getattr(msg, key)
    if hasattr(msg, "foot_force"):
        d["foot_force"] = _seq_to_list(getattr(msg, "foot_force", None), 4)
    return d


def low_message_to_dict(msg: object) -> dict:
    """Extract joints and battery from a ``LowState_`` sample."""
    d: dict = {}
    motors = getattr(msg, "motor_state", None)
    joints: list[dict] = []
    if motors is not None:
        for i, name in enumerate(_GO2_JOINT_NAMES):
            try:
                m = motors[i]
            except (IndexError, TypeError):
                break
            joints.append(
                {
                    "name": name,
                    "position": float(getattr(m, "q", 0.0)),
                    "velocity": float(getattr(m, "dq", 0.0)),
                }
            )
    if joints:
        d["joints"] = joints
    bms = getattr(msg, "bms_state", None)
    if bms is not None:
        soc = getattr(bms, "soc", None)
        if soc is not None:
            d["battery"] = float(soc) / 100.0
    tick = getattr(msg, "tick", None)
    if tick is not None:
        d["msg_timestamp"] = float(tick) / 1_000_000.0
    return d


def merge_state_dicts(sport: Optional[dict], low: Optional[dict]) -> Optional[dict]:
    if sport is None and low is None:
        return None
    merged: dict = {}
    if sport:
        merged.update(sport)
    if low:
        if "battery" in low:
            merged["battery"] = low["battery"]
        if "joints" in low:
            merged["joints"] = low["joints"]
        if "msg_timestamp" in low and "msg_timestamp" not in merged:
            merged["msg_timestamp"] = low["msg_timestamp"]
    return merged


# Sport-mode actions this adapter can forward (snake_case → SportClient method).
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


StateFactory = Callable[[], Callable[[], Optional[dict]]]
SubscriberFactory = Callable[[str, Any, Callable[[Any], None]], Any]


class UnitreeDdsTransport(BaseTransport):
    protocol = "dds"

    def __init__(
        self,
        spec: RobotSpec,
        address: str = "",
        *,
        iface: str = "",
        state_factory: Optional[StateFactory] = None,
        subscriber_factory: Optional[SubscriberFactory] = None,
        read_only: bool = False,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.read_only = read_only
        self._iface, self.domain_id = parse_dds_address(address, iface)
        self.address = self._iface
        self._state_factory = state_factory
        self._subscriber_factory = subscriber_factory
        self._client: Optional[object] = None
        self._state_getter: Optional[Callable[[], Optional[dict]]] = None
        self._sport_sub: Optional[object] = None
        self._low_sub: Optional[object] = None
        self._sport_partial: Optional[dict] = None
        self._low_partial: Optional[dict] = None
        self._lock = threading.Lock()
        self._latest = Telemetry(odom=Odometry())
        self._estopped = False
        self._state = ConnectionState.IDLE
        self._connected_since: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._watchdog: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_msg_mono: Optional[float] = None
        self._last_msg_timestamp: Optional[float] = None
        self._recv_times: Deque[float] = deque()
        self._loop_timer = LoopTimer(1.0 / _READER_HZ)

    @property
    def iface(self) -> str:
        """The DDS network interface."""
        return self._iface

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def connect(self) -> TransportStatus:
        self._stop.clear()
        if self._state_factory is not None:
            self._state_getter = self._state_factory()
        else:
            self._connect_dds()
        self._state = ConnectionState.CONNECTED
        self._connected_since = time.time()
        self._start_reader()
        self._start_watchdog()
        s = self.status()
        self._emit_status(s)
        return s

    def _connect_dds(self) -> None:
        try:
            from unitree_sdk2py.core.channel import (  # type: ignore
                ChannelFactoryInitialize,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import (  # type: ignore
                LowState_,
                SportModeState_,
            )

            if not self.read_only:
                from unitree_sdk2py.go2.sport.sport_client import (  # type: ignore
                    SportClient,
                )
        except Exception as e:
            raise AdapterUnavailable(
                "The Unitree DDS adapter needs unitree_sdk2py (install from "
                "https://github.com/unitreerobotics/unitree_sdk2_python) plus "
                "cyclonedds: pip install 'ohho-os[unitree]'. "
                "Use transport='sim://' to explore without hardware."
            ) from e
        try:
            ChannelFactoryInitialize(self.domain_id, self._iface)
        except Exception as e:
            raise AdapterUnavailable(
                f"DDS channel factory failed for domain={self.domain_id} "
                f"iface={self._iface!r}: {e}"
            ) from e
        if not self.read_only:
            client = SportClient()
            set_timeout = getattr(client, "SetTimeout", None)
            if callable(set_timeout):
                set_timeout(_SPORT_RPC_TIMEOUT_SEC)
            client.Init()
            self._client = client

        if self._subscriber_factory is not None:
            self._sport_sub = self._subscriber_factory(
                "rt/sportmodestate", SportModeState_, self._on_sport_message
            )
            self._low_sub = self._subscriber_factory(
                "rt/lowstate", LowState_, self._on_low_message
            )
        else:
            self._sport_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
            self._sport_sub.Init(self._on_sport_message, 10)
            self._low_sub = ChannelSubscriber("rt/lowstate", LowState_)
            self._low_sub.Init(self._on_low_message, 10)

    def _on_sport_message(self, msg: object) -> None:
        partial = sport_message_to_dict(msg)
        with self._lock:
            self._sport_partial = partial
            self._merge_and_emit()

    def _on_low_message(self, msg: object) -> None:
        partial = low_message_to_dict(msg)
        with self._lock:
            self._low_partial = partial
            self._merge_and_emit()

    def _merge_and_emit(self) -> None:
        merged = merge_state_dicts(self._sport_partial, self._low_partial)
        if merged is None:
            return
        now = time.monotonic()
        self._last_msg_mono = now
        self._recv_times.append(now)
        cutoff = now - _MSG_RATE_WINDOW_SEC
        while self._recv_times and self._recv_times[0] < cutoff:
            self._recv_times.popleft()
        msg_ts = merged.get("msg_timestamp")
        if msg_ts is not None:
            self._last_msg_timestamp = float(msg_ts)
        self._on_state(merged)

    def _on_state(self, state: dict) -> None:
        self._latest = sportstate_to_telemetry(state)
        self._emit_telemetry(self._latest)

    def disconnect(self) -> None:
        self._stop_reader()
        self._stop_watchdog()
        for sub in (self._sport_sub, self._low_sub):
            if sub is not None:
                close = getattr(sub, "Close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        _log.debug("subscriber close failed", exc_info=True)
        self._sport_sub = None
        self._low_sub = None
        if self._client is not None:
            self._call_stop_move()
        self._state = ConnectionState.DISCONNECTED
        self._emit_status(self.status())

    def status(self) -> TransportStatus:
        link_state = self._state
        err: Optional[str] = None
        latency = 0.0
        rate = 0.0
        if self._connected_since is not None and self._last_msg_mono is not None:
            span = max(time.monotonic() - self._last_msg_mono, 0.0)
            if span > _STALE_STATE_SEC and link_state == ConnectionState.CONNECTED:
                link_state = ConnectionState.ERROR
                err = f"no DDS state for {span:.1f}s"
        if self._recv_times:
            window = max(time.monotonic() - self._recv_times[0], _MSG_RATE_WINDOW_SEC)
            rate = len(self._recv_times) / window if window > 0 else 0.0
        if self._last_msg_timestamp is not None:
            latency = max((time.time() - self._last_msg_timestamp) * 1000.0, 0.0)
        return TransportStatus(
            protocol=self.protocol,
            state=link_state,
            label=f"Unitree DDS · {self._iface} (domain {self.domain_id})",
            latency_ms=latency,
            msg_rate=rate,
            error=err,
            connected_since=self._connected_since,
        )

    # ── reader / watchdog threads ─────────────────────────────────────────────
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
                            with self._lock:
                                self._sport_partial = dict(state)
                                self._merge_and_emit()
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

    def _start_watchdog(self) -> None:
        if self._watchdog is not None:
            return

        def loop() -> None:
            while not self._stop.is_set():
                if self._state == ConnectionState.CONNECTED:
                    st = self.status()
                    if st.state == ConnectionState.ERROR:
                        self._state = ConnectionState.ERROR
                        self._emit_status(st)
                time.sleep(0.2)

        self._watchdog = threading.Thread(
            target=loop, name="ohho-unitree-wd", daemon=True
        )
        self._watchdog.start()

    def _stop_watchdog(self) -> None:
        self._stop.set()
        t = self._watchdog
        if t is not None:
            t.join(timeout=1.0)
            self._watchdog = None

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
                code = move(vx, vy, vyaw)
                _log_rpc_result("Move", code)
            except Exception as e:
                _log.warning("Unitree SportClient.Move failed: %s", e)

    def send_joint_command(self, name: str, position: float) -> None:
        return  # Go2 has no arm; low-level joint control is a future adapter

    def sport_command(self, name: str, *args: float) -> None:
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
        self._call_stop_move()

    def release_stop(self) -> None:
        self._estopped = False

    def _call_stop_move(self) -> None:
        if self._client is None:
            return
        stop = getattr(self._client, "StopMove", None)
        if callable(stop):
            try:
                code = stop()
                _log_rpc_result("StopMove", code)
            except Exception as e:
                _log.warning("Unitree SportClient.StopMove failed: %s", e)

    @staticmethod
    def _sportstate_to_dict(msg: object) -> dict:
        """Backward-compatible alias for tests."""
        return sport_message_to_dict(msg)


def _log_rpc_result(name: str, code: object) -> None:
    if code is None:
        return
    if int(code) != 0:
        _log.warning("Unitree SportClient.%s returned error code %s", name, code)
