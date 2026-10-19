"""Read-only PX4 doctor for ``px4+dds://``.

The report always contains the same checks. A missing import, agent, or
topic fails that check. Nothing here calls ``create_publisher`` or sends a
``VehicleCommand``. A sample that was not observed is not reported as a pass:
``armed`` stays unknown rather than ``false``, and a missing GPS message is
not reported as fix ``none``.
"""

from __future__ import annotations

import importlib
import math
import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .adapters.px4_dds import (
    ARMING_STATE_ARMED,
    MICRO_XRCE_AGENT_VERSION,
    PX4_MSGS_BRANCH,
    PX4_VERSION,
    ROS_DISTRO,
    ROS_DISTRO_ALTERNATIVE,
    SUBSCRIPTION_QOS,
    Px4DdsError,
    fmu_topic,
    gps_fix_name,
    offboard_field_mismatch,
    parse_px4_dds_uri,
    qos_for_node,
    read_offboard_fields,
)

PX4_DOCTOR_ROBOTS = ("px4-sitl", "px4-x500")
PX4_DOCTOR_CHECKS = (
    "xrce_agent",
    "vehicle_status_rate",
    "px4_msgs",
    "ros_distro",
    "qos",
    "armed",
    "gps_fix",
    "battery",
)
MIN_VEHICLE_STATUS_HZ = 1.0
GPS_FIX_NAMES = ("none", "2d", "3d", "dgps", "rtk_float", "rtk_fixed")
_ACCEPTED_DISTROS = frozenset({ROS_DISTRO, ROS_DISTRO_ALTERNATIVE})
_RELIABILITY_INTS = {1: "reliable", 2: "best_effort"}
AgentProbe = Callable[[str, int], tuple[bool, str]]
SpinFn = Callable[[Any, float], None]


@dataclass(frozen=True)
class DoctorCheck:
    """One doctor row. ``value`` is JSON-friendly."""

    name: str
    ok: bool
    detail: str
    value: Any = None


@dataclass(frozen=True)
class DoctorReport:
    """JSON report for ``ohho doctor --robot px4-sitl|px4-x500``."""

    robot: str
    transport: str
    ok: bool
    checks: tuple[DoctorCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot": self.robot,
            "transport": self.transport,
            "ok": self.ok,
            "checks": [
                {
                    "name": check.name,
                    "ok": check.ok,
                    "detail": check.detail,
                    "value": check.value,
                }
                for check in self.checks
            ],
        }


@dataclass(frozen=True)
class Px4DoctorSample:
    """What the doctor was able to observe. ``None`` means not observed."""

    agent_reachable: Optional[bool]
    agent_detail: str
    vehicle_status_hz: Optional[float]
    vehicle_status_detail: str
    px4_msgs_fields: Optional[Mapping[str, str]]
    px4_msgs_detail: str
    ros_distro: Optional[str]
    qos_best_effort: Optional[bool]
    qos_detail: str
    armed: Optional[bool]
    armed_detail: str
    gps_fix: Optional[str]
    gps_detail: str
    battery_remaining: Optional[float]
    battery_voltage_v: Optional[float]
    battery_detail: str


def evaluate_px4_doctor(
    robot: str, transport: str, sample: Px4DoctorSample
) -> DoctorReport:
    """Turn one sample into the fixed check list. Does not touch the network."""
    checks = (
        _check_agent(sample),
        _check_rate(sample),
        _check_msgs(sample),
        _check_distro(sample),
        _check_qos(sample),
        _check_armed(sample),
        _check_gps(sample),
        _check_battery(sample),
    )
    if tuple(check.name for check in checks) != PX4_DOCTOR_CHECKS:
        raise RuntimeError("px4 doctor check list drifted")
    return DoctorReport(
        robot=robot,
        transport=transport,
        ok=all(check.ok for check in checks),
        checks=checks,
    )


def sample_px4_graph(
    uri: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
    probe: Optional[Callable[[str], Px4DoctorSample]] = None,
    node_factory: Optional[Callable[[], Any]] = None,
    agent_probe: Optional[AgentProbe] = None,
    spin: Optional[SpinFn] = None,
    window_s: float = 1.0,
) -> Px4DoctorSample:
    """Collect a sample. ``probe`` replaces the live graph for tests."""
    if probe is not None:
        return probe(uri)
    return live_px4_sample(
        uri,
        environ=environ,
        node_factory=node_factory,
        agent_probe=agent_probe,
        spin=spin,
        window_s=window_s,
    )


def live_px4_sample(
    uri: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
    node_factory: Optional[Callable[[], Any]] = None,
    agent_probe: Optional[AgentProbe] = None,
    spin: Optional[SpinFn] = None,
    window_s: float = 1.0,
) -> Px4DoctorSample:
    """Observe a live graph, or fail the graph checks when ROS 2 is absent.

    When rclpy cannot be imported and no ``node_factory`` is supplied, this
    does not open a socket and does not report the agent as reachable.
    """
    env = os.environ if environ is None else environ
    ros = _ros_distro(env)
    classes, fields, px4_detail = _load_px4_classes()
    rclpy_ok, rclpy_detail = _rclpy_status()
    if node_factory is None and not rclpy_ok:
        return _unobserved(
            ros=ros,
            fields=fields,
            px4_detail=px4_detail,
            rclpy_detail=rclpy_detail,
            uri=uri,
        )
    started = False
    node: Any = None
    try:
        if node_factory is None:
            node, started = _open_rclpy_node()
            chosen_probe = probe_xrce_agent if agent_probe is None else agent_probe
        else:
            node = node_factory()
            chosen_probe = agent_probe
        return observe_px4_graph(
            node,
            classes,
            uri,
            agent_probe=chosen_probe,
            spin=spin,
            window_s=window_s,
            ros_distro=ros,
            px4_fields=fields,
            px4_detail=px4_detail,
        )
    finally:
        if node is not None:
            destroy = getattr(node, "destroy_node", None)
            if callable(destroy):
                try:
                    destroy()
                except Exception:
                    pass
        if started:
            rclpy = _import_module("rclpy")
            if rclpy.ok():
                rclpy.shutdown()


def observe_px4_graph(
    node: Any,
    classes: Mapping[str, Any],
    uri: str,
    *,
    agent_probe: Optional[AgentProbe] = None,
    spin: Optional[SpinFn] = None,
    window_s: float = 1.0,
    ros_distro: Optional[str] = None,
    px4_fields: Optional[Mapping[str, str]] = None,
    px4_detail: str = "",
) -> Px4DoctorSample:
    """Subscribe and count. Does not create a publisher.

    ``agent_probe`` runs only when the status topic has no publisher. Pass
    ``None`` to skip the UDP probe and fail the agent check instead.
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    host, port, namespace = parse_px4_dds_uri(uri)
    status_topic = fmu_topic(namespace, "out", "vehicle_status")
    battery_topic = fmu_topic(namespace, "out", "battery_status")
    gps_topic = fmu_topic(namespace, "out", "sensor_gps")
    state = _Seen()
    sub_error = ""
    create_sub = getattr(node, "create_subscription", None)
    sub_qos = qos_for_node(SUBSCRIPTION_QOS)
    if not callable(create_sub):
        sub_error = "the ROS node has no create_subscription"
    else:
        sub_error = _subscribe(
            create_sub, classes, sub_qos, status_topic, battery_topic, gps_topic, state
        )
    status_ok = sub_error == "" and "VehicleStatus" in classes
    if status_ok:
        (spin or _default_spin)(node, window_s)
        hz: Optional[float] = state.status_count / window_s
        rate_detail = (
            f"{status_topic} measured {state.status_count} messages "
            f"in {window_s:.1f}s ({hz:.2f} Hz)"
        )
    else:
        hz = None
        rate_detail = (
            f"{status_topic} was not sampled. {sub_error or 'subscription failed'}"
        )
    infos, info_error = _publisher_infos(node, status_topic)
    agent_reachable, agent_detail = _agent_from_infos(
        infos, info_error, host, port, status_topic, agent_probe
    )
    qos_ok, qos_detail = _qos_from_infos(infos, info_error, status_topic, sub_qos)
    armed, armed_detail = _armed_from_state(state, status_topic)
    gps, gps_detail = _gps_from_state(state, gps_topic)
    remaining, voltage, battery_detail = _battery_from_state(state, battery_topic)
    return Px4DoctorSample(
        agent_reachable=agent_reachable,
        agent_detail=agent_detail,
        vehicle_status_hz=hz,
        vehicle_status_detail=rate_detail,
        px4_msgs_fields=None if px4_fields is None else dict(px4_fields),
        px4_msgs_detail=px4_detail,
        ros_distro=ros_distro,
        qos_best_effort=qos_ok,
        qos_detail=qos_detail,
        armed=armed,
        armed_detail=armed_detail,
        gps_fix=gps,
        gps_detail=gps_detail,
        battery_remaining=remaining,
        battery_voltage_v=voltage,
        battery_detail=battery_detail,
    )


def probe_xrce_agent(host: str, port: int, timeout: float = 0.4) -> tuple[bool, str]:
    """True only when ``host:port`` sends a UDP reply.

    A send that raises, or a timeout, is not reachable. The payload is one
    zero byte. It is not a ``VehicleCommand`` and not a ``px4_msgs`` message.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive number of seconds")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_DGRAM)
    try:
        sock.settimeout(float(timeout))
        address: tuple[Any, ...]
        if family == socket.AF_INET6:
            address = (host, port, 0, 0)
        else:
            address = (host, port)
        sock.sendto(b"\x00", address)
        try:
            data, _addr = sock.recvfrom(128)
        except socket.timeout:
            return False, (
                f"Micro XRCE-DDS Agent at {host}:{port} did not reply. "
                "Not reported reachable."
            )
        if not data:
            return False, (
                f"Micro XRCE-DDS Agent at {host}:{port} sent an empty reply. "
                "Not reported reachable."
            )
        return True, (
            f"Micro XRCE-DDS Agent at {host}:{port} replied ({len(data)} bytes)."
        )
    except OSError as exc:
        return False, (f"Micro XRCE-DDS Agent at {host}:{port} is not reachable: {exc}")
    finally:
        sock.close()


class _Seen:
    def __init__(self) -> None:
        self.status_count = 0
        self.armed: Optional[bool] = None
        self.gps_seen = False
        self.gps: Optional[str] = None
        self.battery_seen = False
        self.remaining: Optional[float] = None
        self.voltage: Optional[float] = None


def _check_agent(sample: Px4DoctorSample) -> DoctorCheck:
    if sample.agent_reachable is True:
        detail = sample.agent_detail or "XRCE agent is reachable"
        return DoctorCheck("xrce_agent", True, detail, True)
    if sample.agent_reachable is False:
        detail = sample.agent_detail or "XRCE agent is not reachable"
    else:
        detail = (
            sample.agent_detail or "XRCE agent was not checked and is not reachable"
        )
    return DoctorCheck("xrce_agent", False, detail, sample.agent_reachable)


def _check_rate(sample: Px4DoctorSample) -> DoctorCheck:
    hz = sample.vehicle_status_hz
    ok = _finite(hz) and float(hz) >= MIN_VEHICLE_STATUS_HZ  # type: ignore[arg-type]
    value = float(hz) if _finite(hz) else None
    if sample.vehicle_status_detail:
        detail = sample.vehicle_status_detail
    elif ok:
        detail = f"/fmu/out/vehicle_status at {float(hz):.2f} Hz"  # type: ignore[arg-type]
    elif hz is None:
        detail = "/fmu/out/vehicle_status was not sampled"
    else:
        detail = (
            f"/fmu/out/vehicle_status at {hz} Hz, below {MIN_VEHICLE_STATUS_HZ:g} Hz"
        )
    if not ok and hz is not None and _finite(hz) and float(hz) < MIN_VEHICLE_STATUS_HZ:
        detail = (
            f"{detail} Below {MIN_VEHICLE_STATUS_HZ:g} Hz."
            if "Below" not in detail
            else detail
        )
    return DoctorCheck("vehicle_status_rate", bool(ok), detail, value)


def _check_msgs(sample: Px4DoctorSample) -> DoctorCheck:
    if sample.px4_msgs_fields is None:
        detail = sample.px4_msgs_detail or (
            "px4_msgs OffboardControlMode fields were not read. "
            f"Pin is {PX4_MSGS_BRANCH} (PX4 v{PX4_VERSION})."
        )
        return DoctorCheck("px4_msgs", False, detail, None)
    mismatch = offboard_field_mismatch(sample.px4_msgs_fields)
    if mismatch:
        return DoctorCheck("px4_msgs", False, mismatch, "mismatch")
    detail = (
        f"OffboardControlMode matches px4_msgs {PX4_MSGS_BRANCH} (PX4 v{PX4_VERSION})."
    )
    return DoctorCheck("px4_msgs", True, detail, PX4_MSGS_BRANCH)


def _check_distro(sample: Px4DoctorSample) -> DoctorCheck:
    raw = sample.ros_distro
    norm = raw.strip().lower() if isinstance(raw, str) else ""
    if norm in _ACCEPTED_DISTROS:
        which = "pin" if norm == ROS_DISTRO else f"alternative to {ROS_DISTRO}"
        return DoctorCheck("ros_distro", True, f"ROS_DISTRO={norm} ({which}).", norm)
    if not norm:
        detail = (
            f"ROS_DISTRO is unset. Pin is {ROS_DISTRO}; "
            f"{ROS_DISTRO_ALTERNATIVE} is the alternative. Not assumed {ROS_DISTRO}."
        )
    else:
        detail = f"ROS_DISTRO={raw!r} is not {ROS_DISTRO} or {ROS_DISTRO_ALTERNATIVE}."
    return DoctorCheck("ros_distro", False, detail, norm or None)


def _check_qos(sample: Px4DoctorSample) -> DoctorCheck:
    if sample.qos_best_effort is True:
        detail = sample.qos_detail or "QoS is best-effort"
        return DoctorCheck("qos", True, detail, "best_effort")
    detail = sample.qos_detail or (
        "QoS was not observed as best-effort and is not assumed best-effort."
    )
    value = None if sample.qos_best_effort is None else "not_best_effort"
    return DoctorCheck("qos", False, detail, value)


def _check_armed(sample: Px4DoctorSample) -> DoctorCheck:
    if sample.armed is False:
        detail = sample.armed_detail or "armed=false"
        return DoctorCheck("armed", True, detail, False)
    if sample.armed is True:
        detail = sample.armed_detail or (
            "autopilot arming_state is armed. Read-only doctor expects armed=false."
        )
        return DoctorCheck("armed", False, detail, True)
    detail = sample.armed_detail or (
        "armed was not observed and is not reported as false."
    )
    return DoctorCheck("armed", False, detail, None)


def _check_gps(sample: Px4DoctorSample) -> DoctorCheck:
    if sample.gps_fix in GPS_FIX_NAMES:
        detail = sample.gps_detail or f"gps_fix={sample.gps_fix}"
        return DoctorCheck("gps_fix", True, detail, sample.gps_fix)
    if sample.gps_fix is None:
        detail = sample.gps_detail or "GPS fix was not observed. No fix is reported."
    else:
        detail = sample.gps_detail or (
            f"GPS fix {sample.gps_fix!r} is not a release/1.17 fix name."
        )
    return DoctorCheck("gps_fix", False, detail, None)


def _check_battery(sample: Px4DoctorSample) -> DoctorCheck:
    remaining = sample.battery_remaining
    if _fraction(remaining):
        voltage = sample.battery_voltage_v
        extra = ""
        if _finite(voltage):
            extra = f", voltage_v={float(voltage):.2f}"  # type: ignore[arg-type]
        detail = (
            sample.battery_detail or f"battery remaining={float(remaining):.2f}{extra}"
        )  # type: ignore[arg-type]
        return DoctorCheck("battery", True, detail, float(remaining))  # type: ignore[arg-type]
    detail = sample.battery_detail or (
        "battery remaining was not observed. No fraction is reported."
    )
    return DoctorCheck("battery", False, detail, None)


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _fraction(value: Any) -> bool:
    return _finite(value) and 0.0 <= float(value) <= 1.0


def _ros_distro(env: Mapping[str, str]) -> Optional[str]:
    raw = env.get("ROS_DISTRO")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def _import_module(name: str) -> Any:
    return importlib.import_module(name)


def _find_spec(name: str) -> Any:
    return importlib.util.find_spec(name)


def _rclpy_status() -> tuple[bool, str]:
    if _find_spec("rclpy") is None:
        return False, (
            "rclpy is not importable. Source ROS 2 "
            f"{ROS_DISTRO} ({ROS_DISTRO_ALTERNATIVE} is the alternative). "
            "The [ros2] extra does not pip-install rclpy."
        )
    return True, "rclpy is importable"


def _load_px4_classes() -> tuple[dict[str, Any], Optional[dict[str, str]], str]:
    try:
        module = _import_module("px4_msgs.msg")
    except ImportError as exc:
        detail = (
            "px4_msgs is not importable. Pin is px4_msgs "
            f"{PX4_MSGS_BRANCH} (PX4 v{PX4_VERSION}), agent "
            f"{MICRO_XRCE_AGENT_VERSION}. {exc}"
        )
        return {}, None, detail
    offboard = getattr(module, "OffboardControlMode", None)
    if offboard is None:
        return (
            {},
            None,
            (f"px4_msgs.msg has no OffboardControlMode. Checkout {PX4_MSGS_BRANCH}."),
        )
    try:
        fields = read_offboard_fields(offboard)
    except Px4DdsError as exc:
        fields = None
        detail = str(exc)
    else:
        detail = "read OffboardControlMode field types"
    classes: dict[str, Any] = {"OffboardControlMode": offboard}
    for name in ("VehicleStatus", "BatteryStatus", "SensorGps"):
        cls = getattr(module, name, None)
        if cls is not None:
            classes[name] = cls
    return classes, fields, detail


def _unobserved(
    *,
    ros: Optional[str],
    fields: Optional[Mapping[str, str]],
    px4_detail: str,
    rclpy_detail: str,
    uri: str,
) -> Px4DoctorSample:
    host, port, namespace = parse_px4_dds_uri(uri)
    topic = fmu_topic(namespace, "out", "vehicle_status")
    return Px4DoctorSample(
        agent_reachable=False,
        agent_detail=(
            f"rclpy is not importable, so the Micro XRCE-DDS Agent at "
            f"{host}:{port} was not probed and is not reported reachable. "
            f"{rclpy_detail}"
        ),
        vehicle_status_hz=None,
        vehicle_status_detail=(f"{topic} was not sampled. rclpy is not importable."),
        px4_msgs_fields=None if fields is None else dict(fields),
        px4_msgs_detail=px4_detail,
        ros_distro=ros,
        qos_best_effort=None,
        qos_detail=(
            "QoS was not observed. rclpy is not importable. Not assumed best-effort."
        ),
        armed=None,
        armed_detail=(
            "vehicle_status was not sampled. armed is not reported as false."
        ),
        gps_fix=None,
        gps_detail="sensor_gps was not sampled. No GPS fix is reported.",
        battery_remaining=None,
        battery_voltage_v=None,
        battery_detail="battery_status was not sampled. No fraction is reported.",
    )


def _subscribe(
    create_sub: Callable[..., Any],
    classes: Mapping[str, Any],
    sub_qos: Any,
    status_topic: str,
    battery_topic: str,
    gps_topic: str,
    state: _Seen,
) -> str:
    if "VehicleStatus" not in classes:
        return "px4_msgs is missing VehicleStatus"
    try:
        create_sub(classes["VehicleStatus"], status_topic, _on_status(state), sub_qos)
        if "BatteryStatus" in classes:
            create_sub(
                classes["BatteryStatus"], battery_topic, _on_battery(state), sub_qos
            )
        if "SensorGps" in classes:
            create_sub(classes["SensorGps"], gps_topic, _on_gps(state), sub_qos)
    except Exception as exc:
        return f"create_subscription failed: {exc}"
    return ""


def _on_status(state: _Seen) -> Callable[[Any], None]:
    def callback(msg: Any) -> None:
        state.status_count += 1
        state.armed = _observed_armed(msg)

    return callback


def _on_battery(state: _Seen) -> Callable[[Any], None]:
    def callback(msg: Any) -> None:
        state.battery_seen = True
        state.remaining = _remaining(msg)
        state.voltage = _voltage(msg)

    return callback


def _on_gps(state: _Seen) -> Callable[[Any], None]:
    def callback(msg: Any) -> None:
        state.gps_seen = True
        state.gps = gps_fix_name(getattr(msg, "fix_type", None))

    return callback


def _observed_armed(msg: Any) -> Optional[bool]:
    if msg is None or not hasattr(msg, "arming_state"):
        return None
    value = msg.arming_state
    # bool is an int subclass. True is not arming state 2 and must not pass
    # as disarmed.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return int(value) == ARMING_STATE_ARMED
    except (TypeError, ValueError):
        return None


def _remaining(msg: Any) -> Optional[float]:
    number = _number(getattr(msg, "remaining", None))
    if number is None or number < 0.0 or number > 1.0:
        return None
    return number


def _voltage(msg: Any) -> Optional[float]:
    number = _number(getattr(msg, "voltage_v", None))
    if number is None or number <= 0.0:
        return None
    return number


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _publisher_infos(node: Any, topic: str) -> tuple[list[Any], str]:
    getter = getattr(node, "get_publishers_info_by_topic", None)
    if not callable(getter):
        return [], "node has no get_publishers_info_by_topic"
    try:
        raw = getter(topic)
    except Exception as exc:
        return [], f"get_publishers_info_by_topic failed: {exc}"
    if raw is None:
        return [], ""
    return list(raw), ""


def _agent_from_infos(
    infos: list[Any],
    info_error: str,
    host: str,
    port: int,
    topic: str,
    agent_probe: Optional[AgentProbe],
) -> tuple[bool, str]:
    if infos:
        return True, (
            f"{len(infos)} publisher(s) on {topic}. "
            f"Micro XRCE-DDS Agent is bridging {host}:{port}."
        )
    if agent_probe is not None:
        return agent_probe(host, port)
    extra = f" {info_error}." if info_error else ""
    return False, (
        f"no publisher on {topic} at {host}:{port}.{extra} "
        "The XRCE agent was not probed and is not reported reachable."
    )


def _qos_from_infos(
    infos: list[Any], info_error: str, topic: str, sub_qos: Any
) -> tuple[Optional[bool], str]:
    sub_token = _reliability_token(sub_qos)
    if not infos:
        extra = f" {info_error}." if info_error else ""
        return None, (
            f"no publisher QoS observed on {topic}.{extra} Not assumed best-effort."
        )
    tokens = [_reliability_token(getattr(info, "qos_profile", info)) for info in infos]
    if sub_token != "best_effort":
        return False, (
            f"doctor subscription QoS is {sub_token or 'unknown'}, not best-effort."
        )
    if tokens and all(token == "best_effort" for token in tokens):
        return True, (
            f"subscription and {len(infos)} publisher(s) on {topic} are best-effort."
        )
    shown = ", ".join(token or "unknown" for token in tokens)
    return False, f"publisher QoS on {topic} is {shown}, not best-effort."


def _reliability_token(qos: Any) -> str:
    reliability = getattr(qos, "reliability", qos)
    name = getattr(reliability, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip().lower()
    if isinstance(reliability, int) and not isinstance(reliability, bool):
        return _RELIABILITY_INTS.get(reliability, str(reliability))
    text = str(reliability).strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _armed_from_state(state: _Seen, topic: str) -> tuple[Optional[bool], str]:
    if state.status_count == 0:
        return None, f"no {topic} sample. armed is not reported as false."
    if state.armed is None:
        return None, "vehicle_status arrived without a readable arming_state."
    if state.armed:
        return True, "arming_state is armed (2). Read-only doctor expects armed=false."
    return False, "arming_state is not armed."


def _gps_from_state(state: _Seen, topic: str) -> tuple[Optional[str], str]:
    if not state.gps_seen:
        return None, f"no sample on {topic}. No GPS fix is reported."
    if state.gps is None:
        return None, f"{topic} had no recognised fix_type. No fix is reported."
    return state.gps, f"{topic} fix_type maps to {state.gps}."


def _battery_from_state(
    state: _Seen, topic: str
) -> tuple[Optional[float], Optional[float], str]:
    if not state.battery_seen:
        return None, None, f"no sample on {topic}. No fraction is reported."
    if state.remaining is None:
        return (
            None,
            state.voltage,
            (f"{topic} remaining is missing or outside 0..1. No fraction is reported."),
        )
    voltage = ""
    if state.voltage is not None:
        voltage = f", voltage_v={state.voltage:.2f}"
    return (
        state.remaining,
        state.voltage,
        (f"{topic} remaining={state.remaining:.2f}{voltage}."),
    )


def _default_spin(node: Any, window_s: float) -> None:
    rclpy = _import_module("rclpy")
    deadline = time.monotonic() + window_s
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        rclpy.spin_once(node, timeout_sec=min(0.1, max(remaining, 0.0)))


def _open_rclpy_node() -> tuple[Any, bool]:
    rclpy = _import_module("rclpy")
    started = False
    if not rclpy.ok():
        rclpy.init(args=None)
        started = True
    node = rclpy.create_node(
        "ohho_px4_doctor",
        enable_rosout=False,
        start_parameter_services=False,
    )
    return node, started


__all__ = [
    "GPS_FIX_NAMES",
    "MIN_VEHICLE_STATUS_HZ",
    "PX4_DOCTOR_CHECKS",
    "PX4_DOCTOR_ROBOTS",
    "DoctorCheck",
    "DoctorReport",
    "Px4DoctorSample",
    "evaluate_px4_doctor",
    "live_px4_sample",
    "observe_px4_graph",
    "probe_xrce_agent",
    "sample_px4_graph",
]
