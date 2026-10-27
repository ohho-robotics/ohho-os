"""Hardware self-test suite and report generator.

Table-driven profiles per robot category:
- **Mobile manipulator bench:** serial link, wheel spin + encoders, IMU rate,
  STS3215 arm, camera frames.
- **Legged (Go2):** DDS sport/lowstate rates, 12 leg motors, IMU at rest, BMS
  SOC, error_code, front camera JPEG, optional stand/lie cycle.

Generates dated JSON test reports and surfaces PASS / FAIL / SKIP statuses.
Hardware mode never falls back to simulation.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .adapters.composite import CompositeTransport
from .adapters.sim import SimTransport
from .registry import RobotSpec, get_spec
from .schema import ConnectionState, Velocity
from .transport import Transport


@dataclass
class CheckResult:
    """Individual hardware check outcome."""

    name: str
    status: str  # "PASS", "FAIL", "SKIP"
    reason: str
    measurements: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "measurements": self.measurements,
            "duration_s": round(self.duration_s, 4),
        }


@dataclass
class SelfTestReport:
    """Complete hardware self-test report."""

    timestamp: str
    robot: str
    mode: str  # "sim" or "hardware"
    overall_status: str  # "PASS", "FAIL", "SKIP"
    summary: dict[str, Any]
    checks: dict[str, CheckResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "robot": self.robot,
            "mode": self.mode,
            "overall_status": self.overall_status,
            "summary": self.summary,
            "checks": {name: res.to_dict() for name, res in self.checks.items()},
        }


def default_report_path(now: Optional[datetime.datetime] = None) -> str:
    """Generate default dated JSON report filename: selftest-YYYYMMDD-HHMMSS.json."""
    t = now or datetime.datetime.now()
    return f"selftest-{t.strftime('%Y%m%d-%H%M%S')}.json"


@dataclass(frozen=True)
class SelfTestProfile:
    """Table-driven check list selected from robot category / capabilities."""

    id: str
    check_ids: tuple[str, ...]


MOBILE_MANIPULATOR_BENCH = SelfTestProfile(
    "mobile_manipulator_bench",
    (
        "serial_link",
        "wheel_spin_encoders",
        "imu_rate",
        "sts3215_servos",
        "camera_frames",
    ),
)

LEGGED_GO2_PROFILE = SelfTestProfile(
    "legged_go2",
    (
        "dds_link",
        "leg_motors",
        "imu_rpy_rest",
        "bms_soc",
        "error_code",
        "front_camera",
        "stand_lie_cycle",
    ),
)

# DDS topic rate floors (sport / lowstate). Sim uses ideal rates; hardware allows
# the current Unitree adapter sport poll (~20 Hz) until lowstate subscription lands.
_DDS_SPORT_MIN_HZ_SIM = 50.0
_DDS_LOW_MIN_HZ_SIM = 100.0
_DDS_SPORT_MIN_HZ_HW = 18.0
_DDS_LOW_MIN_HZ_HW = 80.0
_GO2_MOTOR_COUNT = 12
_GO2_MOTOR_MAX_TEMP_C = 80.0
_GO2_BMS_MIN_SOC = 0.30
_GO2_IMU_MAX_ABS_RPY_REST = 0.35


def profile_for_robot(spec: RobotSpec) -> SelfTestProfile:
    if spec.category == "legged":
        return LEGGED_GO2_PROFILE
    return MOBILE_MANIPULATOR_BENCH


def default_hardware_transport_uri(
    spec: RobotSpec,
    *,
    transport_uri: str = "",
    base_port: str = "",
    arm_port: str = "",
) -> str:
    if transport_uri:
        return transport_uri
    if spec.category == "legged":
        iface = os.environ.get("OHHO_GO2_IFACE", "eth0")
        domain = os.environ.get("OHHO_GO2_DOMAIN", "").strip()
        if domain:
            return f"dds://{iface}?domain={domain}"
        return f"dds://{iface}"
    b_port = base_port or "/dev/ttyUSB0"
    a_port = arm_port or "/dev/ttyACM0"
    return f"serial://{b_port},{a_port}"


def _read_legged_diag(transport: Transport) -> dict[str, Any]:
    fn = getattr(transport, "read_legged_diagnostics", None)
    if callable(fn):
        data = fn()
        if isinstance(data, dict):
            return data
    t = transport.read()
    custom = t.custom or {}
    rpy = [custom.get("roll", 0.0), custom.get("pitch", 0.0)]
    if t.odom is not None:
        rpy.append(float(t.odom.theta))
    return {
        "sport_hz": float(custom.get("sport_hz", 0.0)),
        "lowstate_hz": float(custom.get("lowstate_hz", 0.0)),
        "motors": custom.get("motors") or [],
        "imu_rpy": rpy,
        "battery_soc": t.battery,
        "error_code": custom.get("error_code", 0),
        "camera_jpeg_len": custom.get("camera_jpeg_len"),
    }


def _measure_telemetry_rate_hz(transport: Transport, window_s: float) -> float:
    count = 0

    def _cb(_tel: Any) -> None:
        nonlocal count
        count += 1

    unsub = transport.on_telemetry(_cb)
    try:
        time.sleep(window_s)
    finally:
        unsub()
    return count / window_s if window_s > 0 else 0.0


def _skipped_check(name: str, reason: str) -> CheckResult:
    return CheckResult(name=name, status="SKIP", reason=reason)


def _hardware_init_failure_checks(
    profile: SelfTestProfile, err: Exception, fail_name: str, fail_reason: str
) -> dict[str, CheckResult]:
    checks: dict[str, CheckResult] = {}
    for cid in profile.check_ids:
        if cid == fail_name:
            checks[cid] = CheckResult(
                name=cid,
                status="FAIL",
                reason=fail_reason,
                measurements={"error": str(err)},
            )
        else:
            checks[cid] = _skipped_check(cid, "Hardware transport failed to initialize")
    return checks


# ── Individual Checks ─────────────────────────────────────────────────────────


def _port_exists(port: str) -> bool:
    """Check if a serial port is present on the system."""
    if not port:
        return False
    # Strip the Win32 device namespace prefix (\\.\COM3 -> COM3) before deciding,
    # so COM-style names are handled the same way on every OS.
    clean = port.upper().replace("\\\\.\\", "").strip()
    if os.name == "nt" or clean.startswith("COM"):
        try:
            from serial.tools import list_ports  # type: ignore

            devs = [p.device.upper() for p in list_ports.comports()]
            return clean in devs or port.upper() in devs
        except Exception:
            if port.upper().startswith("COM") or "\\\\.\\COM" in port.upper():
                try:
                    import serial  # type: ignore

                    s = serial.Serial(port)
                    s.close()
                    return True
                except Exception:
                    return False
            return os.path.exists(port)
    return os.path.exists(port)


def check_serial_link(
    transport: Transport,
    *,
    sim: bool = False,
    base_port: str = "",
    arm_port: str = "",
) -> CheckResult:
    """Verify serial communication link with the robot."""
    start_t = time.monotonic()
    if sim:
        st = transport.status()
        dur = time.monotonic() - start_t
        return CheckResult(
            name="serial_link",
            status="PASS",
            reason="Simulated communication link active and responding",
            measurements={
                "protocol": transport.protocol,
                "state": st.state.value,
                "latency_ms": st.latency_ms,
            },
            duration_s=dur,
        )

    # Hardware check
    base_p = base_port or getattr(transport, "port", "/dev/ttyUSB0")
    arm_p = arm_port
    if isinstance(transport, CompositeTransport):
        base_p = getattr(transport.base, "port", base_p)
        arm_p = getattr(transport.arm, "port", arm_port or "/dev/ttyACM0")

    # Check pyserial availability
    try:
        import serial  # noqa: F401
    except ImportError:
        dur = time.monotonic() - start_t
        return CheckResult(
            name="serial_link",
            status="SKIP",
            reason="pyserial not installed; install 'ohho-os[serial]' to test real hardware",
            measurements={"pyserial_available": False, "base_port": base_p},
            duration_s=dur,
        )

    st = transport.status()
    if st.state != ConnectionState.CONNECTED:
        dur = time.monotonic() - start_t
        exists = _port_exists(base_p)
        return CheckResult(
            name="serial_link",
            status="SKIP" if not exists else "FAIL",
            reason=f"Serial port {base_p} not connected (state: {st.state.value})",
            measurements={
                "protocol": transport.protocol,
                "state": st.state.value,
                "base_port": base_p,
                "arm_port": arm_p,
                "port_exists": exists,
            },
            duration_s=dur,
        )

    dur = time.monotonic() - start_t
    return CheckResult(
        name="serial_link",
        status="PASS",
        reason=f"Serial link connected ({transport.protocol} on {base_p})",
        measurements={
            "protocol": transport.protocol,
            "state": st.state.value,
            "base_port": base_p,
            "arm_port": arm_p,
            "latency_ms": st.latency_ms,
        },
        duration_s=dur,
    )


def check_wheel_spin_encoders(
    transport: Transport,
    *,
    sim: bool = False,
    allow_spin: bool = False,
    pulse_speed: int = 50,
    pulse_duration: float = 0.25,
) -> CheckResult:
    """Spin each wheel briefly at low speed and verify encoder counts increment in expected direction.

    Bench safety: Must be explicitly enabled with allow_spin=True (or --allow-spin).
    """
    start_t = time.monotonic()
    wheels = ["front_left", "front_right", "rear_left", "rear_right"]

    if not sim and not allow_spin:
        dur = time.monotonic() - start_t
        return CheckResult(
            name="wheel_spin_encoders",
            status="SKIP",
            reason="Wheel spin safety lock: skipped bench spin test (pass --allow-spin to enable)",
            measurements={"safety_locked": True, "wheels": wheels},
            duration_s=dur,
        )

    if not sim:
        st = transport.status()
        if st.state != ConnectionState.CONNECTED:
            dur = time.monotonic() - start_t
            return CheckResult(
                name="wheel_spin_encoders",
                status="SKIP",
                reason="Wheel spin check skipped: transport not connected",
                measurements={
                    "safety_locked": False,
                    "transport_connected": False,
                    "state": st.state.value,
                },
                duration_s=dur,
            )

    get_enc = getattr(transport, "get_wheel_encoders", None)
    if isinstance(transport, CompositeTransport) and not callable(get_enc):
        get_enc = getattr(transport.base, "get_wheel_encoders", None)

    if not sim:
        enc_counts = get_enc() if callable(get_enc) else None
        if not enc_counts or not all(w in enc_counts for w in wheels):
            dur = time.monotonic() - start_t
            return CheckResult(
                name="wheel_spin_encoders",
                status="SKIP",
                reason="encoder counts not available from this adapter",
                measurements={"safety_locked": False, "transport_connected": True},
                duration_s=dur,
            )

    # In sim, or hardware when allow_spin is True and encoders are available
    wheel_results: dict[str, Any] = {}
    failed_wheels: list[str] = []

    # Map wheel name to 4-tuple motor command (fl, fr, rl, rr)
    motor_maps = {
        "front_left": (pulse_speed, 0, 0, 0),
        "front_right": (0, pulse_speed, 0, 0),
        "rear_left": (0, 0, pulse_speed, 0),
        "rear_right": (0, 0, 0, pulse_speed),
    }

    send_m = getattr(transport, "send_motor", None)

    for w in wheels:
        initial_enc = get_enc().get(w, 0) if callable(get_enc) else 0

        fl, fr, rl, rr = motor_maps[w]
        if callable(send_m):
            send_m(fl, fr, rl, rr)
        else:
            # Fallback to base velocity if direct motor command is unavailable
            transport.send_velocity(Velocity(linear_x=0.05))

        if sim:
            step_fn = getattr(transport, "step", None)
            if callable(step_fn):
                step_fn(pulse_duration)
            else:
                time.sleep(pulse_duration)
        else:
            time.sleep(pulse_duration)

        # Stop motor immediately
        if callable(send_m):
            send_m(0, 0, 0, 0)
        else:
            transport.send_velocity(Velocity())

        if not sim:
            time.sleep(0.05)

        final_enc = get_enc().get(w, 0) if callable(get_enc) else 0
        delta = final_enc - initial_enc
        is_ok = delta > 0

        wheel_results[w] = {
            "commanded_speed": pulse_speed,
            "start_ticks": initial_enc,
            "end_ticks": final_enc,
            "delta_ticks": delta,
            "direction": "forward",
            "ok": is_ok,
        }
        if not is_ok:
            failed_wheels.append(w)

    dur = time.monotonic() - start_t
    if failed_wheels:
        return CheckResult(
            name="wheel_spin_encoders",
            status="FAIL",
            reason=f"Wheel encoder check failed for: {', '.join(failed_wheels)} (delta <= 0)",
            measurements={"wheels": wheel_results, "all_ok": False},
            duration_s=dur,
        )

    return CheckResult(
        name="wheel_spin_encoders",
        status="PASS",
        reason=f"All 4 wheels ({', '.join(wheels)}) spun with positive encoder counts in expected direction",
        measurements={"wheels": wheel_results, "all_ok": True},
        duration_s=dur,
    )


def check_imu_rate(
    transport: Transport,
    *,
    sim: bool = False,
    min_rate_hz: float = 10.0,
    sample_window_s: float = 2.0,
) -> CheckResult:
    """Verify IMU is publishing telemetry at the expected rate."""
    start_t = time.monotonic()

    if sim:
        rate = 20.0
        dur = time.monotonic() - start_t
        return CheckResult(
            name="imu_rate",
            status="PASS",
            reason=f"IMU publishing nominally at {rate:.1f} Hz (expected >= {min_rate_hz:.1f} Hz)",
            measurements={
                "rate_hz": rate,
                "min_expected_hz": min_rate_hz,
                "samples_received": int(rate * sample_window_s),
                "window_s": sample_window_s,
            },
            duration_s=dur,
        )

    # Hardware check
    st = transport.status()
    if st.state != ConnectionState.CONNECTED:
        dur = time.monotonic() - start_t
        return CheckResult(
            name="imu_rate",
            status="SKIP",
            reason="IMU check skipped: transport not connected",
            measurements={"rate_hz": 0.0, "state": st.state.value},
            duration_s=dur,
        )

    # Bounded wait for IMU samples: 0.0 Hz from a just-connected adapter is not final until wait expires
    get_rate = getattr(transport, "get_imu_rate", None)
    t_end = time.monotonic() + sample_window_s
    measured_rate = 0.0

    while True:
        if callable(get_rate):
            measured_rate = get_rate()
        else:
            measured_rate = transport.read().custom.get("imu_rate", 0.0)

        if measured_rate >= min_rate_hz:
            break
        if time.monotonic() >= t_end:
            break
        time.sleep(0.05)

    dur = time.monotonic() - start_t

    # Determine sample count within window
    get_samples = getattr(transport, "get_imu_samples", None)
    if callable(get_samples):
        now = time.monotonic()
        recent = [t for t in get_samples() if now - t <= sample_window_s]
        samples_received = len(recent)
    else:
        samples_received = int(round(measured_rate * sample_window_s))

    if measured_rate >= min_rate_hz:
        return CheckResult(
            name="imu_rate",
            status="PASS",
            reason=f"IMU publishing nominally at {measured_rate:.1f} Hz (expected >= {min_rate_hz:.1f} Hz)",
            measurements={
                "rate_hz": round(measured_rate, 2),
                "min_expected_hz": min_rate_hz,
                "samples_received": samples_received,
                "window_s": sample_window_s,
            },
            duration_s=dur,
        )

    if measured_rate == 0.0:
        return CheckResult(
            name="imu_rate",
            status="FAIL",
            reason=f"No IMU packets received ({samples_received} samples received in {sample_window_s:.1f}s)",
            measurements={
                "rate_hz": 0.0,
                "min_expected_hz": min_rate_hz,
                "samples_received": samples_received,
                "window_s": sample_window_s,
            },
            duration_s=dur,
        )

    return CheckResult(
        name="imu_rate",
        status="FAIL",
        reason=f"IMU rate {measured_rate:.1f} Hz is below minimum {min_rate_hz:.1f} Hz ({samples_received} samples received in {sample_window_s:.1f}s)",
        measurements={
            "rate_hz": round(measured_rate, 2),
            "min_expected_hz": min_rate_hz,
            "samples_received": samples_received,
            "window_s": sample_window_s,
        },
        duration_s=dur,
    )


def check_sts3215_servos(
    transport: Transport,
    *,
    sim: bool = False,
    expected_count: int = 6,
    min_voltage_v: float = 6.0,
    max_voltage_v: float = 9.0,
    max_temp_c: float = 70.0,
) -> CheckResult:
    """Verify each STS3215 servo is responding with ID, voltage, temperature, and position."""
    start_t = time.monotonic()

    read_diag = getattr(transport, "read_servo_diagnostics", None)
    if callable(read_diag):
        servos = read_diag()
    else:
        # Try arm property if composite
        arm = getattr(transport, "arm", None)
        read_diag_arm = getattr(arm, "read_servo_diagnostics", None)
        servos = read_diag_arm() if callable(read_diag_arm) else []

    dur = time.monotonic() - start_t
    if not sim:
        st = transport.status()
        if st.state != ConnectionState.CONNECTED and not any(
            s.get("online", False) for s in servos
        ):
            return CheckResult(
                name="sts3215_servos",
                status="SKIP",
                reason="STS3215 servo check skipped: arm bus not connected",
                measurements={"servos": servos, "count": len(servos)},
                duration_s=dur,
            )

    if not servos:
        return CheckResult(
            name="sts3215_servos",
            status="FAIL",
            reason="No STS3215 servos responded on the arm bus",
            measurements={"servos": [], "count": 0},
            duration_s=dur,
        )

    issues: list[str] = []
    for s in servos:
        sid = s.get("id")
        name = s.get("name", f"servo_{sid}")
        v = s.get("voltage")
        temp = s.get("temperature")
        pos = s.get("position")
        online = s.get("online", False)

        if not online:
            issues.append(f"Servo {sid} ({name}) offline")
            continue
        if pos is None:
            issues.append(f"Servo {sid} ({name}) missing position reading")
        if v is None:
            issues.append(f"Servo {sid} ({name}) missing voltage reading")
        elif v < min_voltage_v or v > max_voltage_v:
            issues.append(
                f"Servo {sid} voltage {v:.1f}V out of range [{min_voltage_v}, {max_voltage_v}]V"
            )
        if temp is None:
            issues.append(f"Servo {sid} ({name}) missing temperature reading")
        elif temp > max_temp_c:
            issues.append(
                f"Servo {sid} temperature {temp:.1f}°C exceeds {max_temp_c}°C"
            )

    if len(servos) < expected_count:
        issues.append(f"Expected {expected_count} servos but received {len(servos)}")

    if issues:
        return CheckResult(
            name="sts3215_servos",
            status="FAIL",
            reason=f"STS3215 servo diagnostics issues: {'; '.join(issues)}",
            measurements={
                "servos": servos,
                "count": len(servos),
                "all_responding": False,
            },
            duration_s=dur,
        )

    return CheckResult(
        name="sts3215_servos",
        status="PASS",
        reason=f"All {len(servos)} STS3215 servos responding (id, voltage, temperature, position)",
        measurements={"servos": servos, "count": len(servos), "all_responding": True},
        duration_s=dur,
    )


def check_camera_frames(
    *,
    sim: bool = False,
    camera_idx: int = 0,
    required_frames: int = 5,
) -> CheckResult:
    """Verify camera frames are arriving."""
    start_t = time.monotonic()

    if sim:
        # Simulate synthetic camera frames
        width = 640
        height = 480
        sim_fps = 30.0
        dur = time.monotonic() - start_t
        return CheckResult(
            name="camera_frames",
            status="PASS",
            reason=f"Camera frame stream active ({required_frames} frames received, {width}x{height} at {sim_fps:.1f} fps)",
            measurements={
                "frames_received": required_frames,
                "width": width,
                "height": height,
                "fps": sim_fps,
                "camera_id": "sim_camera",
            },
            duration_s=dur,
        )

    # Hardware check via OpenCV
    try:
        import cv2  # type: ignore
    except Exception:
        dur = time.monotonic() - start_t
        return CheckResult(
            name="camera_frames",
            status="SKIP",
            reason="OpenCV (cv2) not installed; install 'opencv-python' to capture camera frames",
            measurements={"cv2_available": False, "camera_idx": camera_idx},
            duration_s=dur,
        )

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        dur = time.monotonic() - start_t
        if os.name == "nt":
            status = "SKIP"
            reason = f"No camera found at index {camera_idx} via OpenCV probe"
        else:
            dev_node = f"/dev/video{camera_idx}"
            status = "SKIP" if not os.path.exists(dev_node) else "FAIL"
            reason = f"Could not open camera device index {camera_idx}"
        return CheckResult(
            name="camera_frames",
            status=status,
            reason=reason,
            measurements={"camera_idx": camera_idx, "opened": False},
            duration_s=dur,
        )

    try:
        frames_captured = 0
        w, h = 0, 0
        t0 = time.monotonic()
        for _ in range(required_frames):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frames_captured += 1
                h, w = frame.shape[:2]
            time.sleep(0.01)
        t_capture = time.monotonic() - t0
        actual_fps = frames_captured / t_capture if t_capture > 0 else 0.0
    finally:
        cap.release()

    dur = time.monotonic() - start_t
    if frames_captured >= required_frames:
        return CheckResult(
            name="camera_frames",
            status="PASS",
            reason=f"Camera frame stream active ({frames_captured} frames received, {w}x{h} at {actual_fps:.1f} fps)",
            measurements={
                "frames_received": frames_captured,
                "width": w,
                "height": h,
                "fps": round(actual_fps, 2),
                "camera_idx": camera_idx,
            },
            duration_s=dur,
        )

    return CheckResult(
        name="camera_frames",
        status="FAIL",
        reason=f"Captured only {frames_captured}/{required_frames} frames from camera {camera_idx}",
        measurements={
            "frames_received": frames_captured,
            "required_frames": required_frames,
            "camera_idx": camera_idx,
        },
        duration_s=dur,
    )


# ── Legged (Go2) checks ───────────────────────────────────────────────────────


def check_dds_link(
    transport: Transport,
    *,
    sim: bool = False,
    sample_window_s: float = 0.5,
) -> CheckResult:
    start_t = time.monotonic()
    min_sport = _DDS_SPORT_MIN_HZ_SIM if sim else _DDS_SPORT_MIN_HZ_HW
    min_low = _DDS_LOW_MIN_HZ_SIM if sim else _DDS_LOW_MIN_HZ_HW

    if sim:
        diag = _read_legged_diag(transport)
        sport_hz = float(diag.get("sport_hz", _DDS_SPORT_MIN_HZ_SIM))
        low_hz = float(diag.get("lowstate_hz", _DDS_LOW_MIN_HZ_SIM))
    else:
        st = transport.status()
        if st.state != ConnectionState.CONNECTED:
            dur = time.monotonic() - start_t
            return CheckResult(
                name="dds_link",
                status="SKIP",
                reason=f"DDS link check skipped: transport not connected ({st.state.value})",
                duration_s=dur,
            )
        time.sleep(sample_window_s)
        diag = _read_legged_diag(transport)
        sport_hz = float(diag.get("sport_hz") or 0.0)
        low_hz = float(diag.get("lowstate_hz") or 0.0)
        if sport_hz <= 0.0:
            sport_hz = _measure_telemetry_rate_hz(transport, sample_window_s)

    dur = time.monotonic() - start_t
    measurements = {
        "sport_hz": round(sport_hz, 2),
        "lowstate_hz": round(low_hz, 2),
        "min_sport_hz": min_sport,
        "min_lowstate_hz": min_low,
        "topics": ["rt/sportmodestate", "rt/lowstate"],
    }
    issues: list[str] = []
    if sport_hz < min_sport:
        issues.append(f"sportmodestate {sport_hz:.1f} Hz < {min_sport:.1f} Hz")
    if low_hz < min_low:
        if not sim and low_hz <= 0.0:
            issues.append(
                "lowstate stream not available (adapter lowstate subscription pending)"
            )
        else:
            issues.append(f"lowstate {low_hz:.1f} Hz < {min_low:.1f} Hz")
    if issues:
        return CheckResult(
            name="dds_link",
            status="FAIL",
            reason="DDS link check failed: " + "; ".join(issues),
            measurements=measurements,
            duration_s=dur,
        )
    return CheckResult(
        name="dds_link",
        status="PASS",
        reason=(
            f"DDS streams healthy (sportmodestate {sport_hz:.1f} Hz, "
            f"lowstate {low_hz:.1f} Hz)"
        ),
        measurements=measurements,
        duration_s=dur,
    )


def check_leg_motors(
    transport: Transport,
    *,
    sim: bool = False,
    expected_count: int = _GO2_MOTOR_COUNT,
    max_temp_c: float = _GO2_MOTOR_MAX_TEMP_C,
) -> CheckResult:
    start_t = time.monotonic()
    diag = _read_legged_diag(transport)
    motors = diag.get("motors") or []
    dur = time.monotonic() - start_t

    if not sim and not motors:
        return CheckResult(
            name="leg_motors",
            status="SKIP",
            reason="Leg motor telemetry not available (need rt/lowstate)",
            measurements={"motors": [], "count": 0},
            duration_s=dur,
        )

    issues: list[str] = []
    if len(motors) < expected_count:
        issues.append(f"expected {expected_count} motors, got {len(motors)}")
    for m in motors:
        mid = m.get("id", "?")
        lost = m.get("lost", 0)
        temp = m.get("temperature")
        if lost not in (0, 0.0, False, None):
            issues.append(f"motor {mid} lost={lost}")
        if temp is None:
            issues.append(f"motor {mid} missing temperature")
        elif float(temp) > max_temp_c:
            issues.append(f"motor {mid} temperature {temp}°C > {max_temp_c}°C")

    if issues:
        return CheckResult(
            name="leg_motors",
            status="FAIL",
            reason="Leg motor check failed: " + "; ".join(issues),
            measurements={"motors": motors, "count": len(motors)},
            duration_s=dur,
        )
    return CheckResult(
        name="leg_motors",
        status="PASS",
        reason=f"All {len(motors)} leg motors reporting (lost=0, temperature OK)",
        measurements={"motors": motors, "count": len(motors), "all_ok": True},
        duration_s=dur,
    )


def check_imu_rpy_rest(
    transport: Transport,
    *,
    sim: bool = False,
    max_abs_rad: float = _GO2_IMU_MAX_ABS_RPY_REST,
) -> CheckResult:
    start_t = time.monotonic()
    diag = _read_legged_diag(transport)
    rpy = diag.get("imu_rpy") or [0.0, 0.0, 0.0]
    dur = time.monotonic() - start_t
    roll, pitch = float(rpy[0]), float(rpy[1])
    issues = []
    if abs(roll) > max_abs_rad:
        issues.append(f"|roll|={abs(roll):.3f} rad > {max_abs_rad} rad")
    if abs(pitch) > max_abs_rad:
        issues.append(f"|pitch|={abs(pitch):.3f} rad > {max_abs_rad} rad")
    measurements = {
        "imu_rpy": [round(roll, 4), round(pitch, 4), round(float(rpy[2]), 4)],
        "max_abs_roll_pitch_rad": max_abs_rad,
    }
    if issues:
        return CheckResult(
            name="imu_rpy_rest",
            status="FAIL",
            reason="IMU orientation at rest out of range: " + "; ".join(issues),
            measurements=measurements,
            duration_s=dur,
        )
    return CheckResult(
        name="imu_rpy_rest",
        status="PASS",
        reason="IMU roll/pitch sane at rest",
        measurements=measurements,
        duration_s=dur,
    )


def check_bms_soc(
    transport: Transport,
    *,
    sim: bool = False,
    min_soc: float = _GO2_BMS_MIN_SOC,
) -> CheckResult:
    start_t = time.monotonic()
    diag = _read_legged_diag(transport)
    soc = diag.get("battery_soc")
    dur = time.monotonic() - start_t
    if soc is None:
        return CheckResult(
            name="bms_soc",
            status="FAIL" if sim else "SKIP",
            reason="BMS state-of-charge not reported",
            measurements={"battery_soc": None, "min_soc": min_soc},
            duration_s=dur,
        )
    soc_f = float(soc)
    if soc_f < min_soc:
        return CheckResult(
            name="bms_soc",
            status="FAIL",
            reason=f"BMS SOC {soc_f:.0%} below minimum {min_soc:.0%}",
            measurements={"battery_soc": soc_f, "min_soc": min_soc},
            duration_s=dur,
        )
    return CheckResult(
        name="bms_soc",
        status="PASS",
        reason=f"BMS SOC {soc_f:.0%} (>= {min_soc:.0%})",
        measurements={"battery_soc": soc_f, "min_soc": min_soc},
        duration_s=dur,
    )


def check_error_code(transport: Transport, *, sim: bool = False) -> CheckResult:
    start_t = time.monotonic()
    diag = _read_legged_diag(transport)
    code = diag.get("error_code", 0)
    dur = time.monotonic() - start_t
    try:
        code_i = int(code)
    except (TypeError, ValueError):
        code_i = -1
    if code_i != 0:
        return CheckResult(
            name="error_code",
            status="FAIL",
            reason=f"Robot error_code={code_i} (expected 0)",
            measurements={"error_code": code_i},
            duration_s=dur,
        )
    return CheckResult(
        name="error_code",
        status="PASS",
        reason="error_code == 0",
        measurements={"error_code": 0},
        duration_s=dur,
    )


def check_front_camera(
    transport: Transport,
    *,
    sim: bool = False,
) -> CheckResult:
    start_t = time.monotonic()
    if sim:
        diag = _read_legged_diag(transport)
        jpeg_len = int(diag.get("camera_jpeg_len") or 4096)
        dur = time.monotonic() - start_t
        return CheckResult(
            name="front_camera",
            status="PASS",
            reason=f"Front camera JPEG available (sim render, {jpeg_len} bytes)",
            measurements={"jpeg_bytes": jpeg_len, "source": "sim"},
            duration_s=dur,
        )

    capture = getattr(transport, "capture_front_camera_jpeg", None)
    if not callable(capture):
        dur = time.monotonic() - start_t
        return CheckResult(
            name="front_camera",
            status="SKIP",
            reason="Front camera capture not available on this transport",
            measurements={},
            duration_s=dur,
        )
    data = capture()
    dur = time.monotonic() - start_t
    if not data or len(data) < 100:
        return CheckResult(
            name="front_camera",
            status="FAIL",
            reason="VideoClient.GetImageSample() returned no JPEG data",
            measurements={"jpeg_bytes": len(data) if data else 0},
            duration_s=dur,
        )
    return CheckResult(
        name="front_camera",
        status="PASS",
        reason=f"Front camera JPEG received ({len(data)} bytes)",
        measurements={"jpeg_bytes": len(data), "source": "VideoClient"},
        duration_s=dur,
    )


def check_stand_lie_cycle(
    transport: Transport,
    *,
    sim: bool = False,
    allow_stand: bool = False,
    settle_s: float = 2.0,
) -> CheckResult:
    start_t = time.monotonic()
    if not allow_stand:
        dur = time.monotonic() - start_t
        return CheckResult(
            name="stand_lie_cycle",
            status="SKIP",
            reason="Stand/lie cycle skipped (pass --allow-stand to enable)",
            measurements={"safety_locked": True},
            duration_s=dur,
        )
    if sim:
        dur = time.monotonic() - start_t
        return CheckResult(
            name="stand_lie_cycle",
            status="PASS",
            reason="Stand/lie cycle simulated (stand_up → stand_down)",
            measurements={"simulated": True},
            duration_s=dur,
        )

    sport = getattr(transport, "sport_command", None)
    if not callable(sport):
        dur = time.monotonic() - start_t
        return CheckResult(
            name="stand_lie_cycle",
            status="SKIP",
            reason="sport_command not available on transport (TODO: OHH-118 Legged API)",
            measurements={},
            duration_s=dur,
        )
    try:
        sport("stand_up")
        time.sleep(settle_s)
        sport("stand_down")
        time.sleep(settle_s)
    except Exception as e:
        dur = time.monotonic() - start_t
        return CheckResult(
            name="stand_lie_cycle",
            status="FAIL",
            reason=f"Stand/lie cycle failed: {e}",
            measurements={"error": str(e)},
            duration_s=dur,
        )
    dur = time.monotonic() - start_t
    return CheckResult(
        name="stand_lie_cycle",
        status="PASS",
        reason="Stand/lie cycle completed (stand_up → stand_down)",
        measurements={"settle_s": settle_s},
        duration_s=dur,
    )


def _run_profile_checks(
    profile: SelfTestProfile,
    transport: Transport,
    *,
    sim: bool,
    allow_spin: bool,
    allow_stand: bool,
    base_port: str,
    arm_port: str,
    camera_idx: int,
) -> dict[str, CheckResult]:
    results: dict[str, CheckResult] = {}
    for cid in profile.check_ids:
        if cid == "serial_link":
            results[cid] = check_serial_link(
                transport, sim=sim, base_port=base_port, arm_port=arm_port
            )
        elif cid == "wheel_spin_encoders":
            results[cid] = check_wheel_spin_encoders(
                transport, sim=sim, allow_spin=allow_spin
            )
        elif cid == "imu_rate":
            results[cid] = check_imu_rate(transport, sim=sim)
        elif cid == "sts3215_servos":
            results[cid] = check_sts3215_servos(transport, sim=sim)
        elif cid == "camera_frames":
            results[cid] = check_camera_frames(sim=sim, camera_idx=camera_idx)
        elif cid == "dds_link":
            results[cid] = check_dds_link(transport, sim=sim)
        elif cid == "leg_motors":
            results[cid] = check_leg_motors(transport, sim=sim)
        elif cid == "imu_rpy_rest":
            results[cid] = check_imu_rpy_rest(transport, sim=sim)
        elif cid == "bms_soc":
            results[cid] = check_bms_soc(transport, sim=sim)
        elif cid == "error_code":
            results[cid] = check_error_code(transport, sim=sim)
        elif cid == "front_camera":
            results[cid] = check_front_camera(transport, sim=sim)
        elif cid == "stand_lie_cycle":
            results[cid] = check_stand_lie_cycle(
                transport, sim=sim, allow_stand=allow_stand
            )
        else:
            results[cid] = _skipped_check(cid, f"Unknown check id '{cid}'")
    return results


# ── Full Self-Test Runner ─────────────────────────────────────────────────────


def run_selftest(
    robot: str | None = None,
    *,
    sim: bool = False,
    allow_spin: bool = False,
    allow_stand: bool = False,
    out_path: Optional[str] = None,
    base_port: str = "",
    arm_port: str = "",
    camera_idx: int = 0,
    transport_uri: str = "",
    transport: Optional[Transport] = None,
) -> SelfTestReport:
    """Run all hardware self-test checks and return a complete SelfTestReport."""
    if not robot:
        from .plugins import preferred_robot_id

        robot = preferred_robot_id()
    start_time = datetime.datetime.now(datetime.timezone.utc)
    t_start = time.monotonic()
    spec = get_spec(robot)
    profile = profile_for_robot(spec)
    tp = transport
    owned_transport = False
    mode_str = "sim" if sim else "hardware"

    if tp is None:
        owned_transport = True
        if sim:
            tp = SimTransport(spec)
        else:
            from .adapters import resolve_transport

            uri = default_hardware_transport_uri(
                spec,
                transport_uri=transport_uri,
                base_port=base_port,
                arm_port=arm_port,
            )
            try:
                tp = resolve_transport(uri, spec)
            except Exception as e:
                err_msg = f"Failed to initialize hardware transport: {e}"
                fail_name = "dds_link" if profile.id == "legged_go2" else "serial_link"
                checks_map = _hardware_init_failure_checks(
                    profile, e, fail_name, err_msg
                )
                n_fail = sum(1 for c in checks_map.values() if c.status == "FAIL")
                n_skip = sum(1 for c in checks_map.values() if c.status == "SKIP")
                report = SelfTestReport(
                    timestamp=start_time.isoformat(),
                    robot=robot,
                    mode=mode_str,
                    overall_status="FAIL",
                    summary={
                        "total": len(checks_map),
                        "passed": 0,
                        "failed": n_fail,
                        "skipped": n_skip,
                        "duration_s": round(time.monotonic() - t_start, 4),
                        "transport_error": str(e),
                        "profile": profile.id,
                    },
                    checks=checks_map,
                )
                if out_path:
                    save_report_json(report, out_path)
                raise RuntimeError(err_msg) from e

    try:
        connect_fn = getattr(tp, "connect", None)
        if callable(connect_fn):
            try:
                connect_fn()
            except Exception:
                pass

        checks_map = _run_profile_checks(
            profile,
            tp,
            sim=sim,
            allow_spin=allow_spin,
            allow_stand=allow_stand,
            base_port=base_port,
            arm_port=arm_port,
            camera_idx=camera_idx,
        )

    finally:
        if owned_transport and tp is not None:
            disconnect_fn = getattr(tp, "disconnect", None)
            if callable(disconnect_fn):
                try:
                    disconnect_fn()
                except Exception:
                    pass

    total_duration = time.monotonic() - t_start

    n_pass = sum(1 for c in checks_map.values() if c.status == "PASS")
    n_fail = sum(1 for c in checks_map.values() if c.status == "FAIL")
    n_skip = sum(1 for c in checks_map.values() if c.status == "SKIP")

    overall = "FAIL" if n_fail > 0 else ("PASS" if n_pass > 0 else "SKIP")

    summary = {
        "total": len(checks_map),
        "passed": n_pass,
        "failed": n_fail,
        "skipped": n_skip,
        "duration_s": round(total_duration, 4),
        "profile": profile.id,
    }
    if mode_str == "hardware" and n_pass == 0 and n_fail == 0:
        summary["all_skipped"] = True

    report = SelfTestReport(
        timestamp=start_time.isoformat(),
        robot=robot,
        mode=mode_str,
        overall_status=overall,
        summary=summary,
        checks=checks_map,
    )

    if out_path:
        save_report_json(report, out_path)

    return report


def save_report_json(report: SelfTestReport, path: str | Path) -> None:
    """Save self-test report to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)


def format_report_text(report: SelfTestReport) -> str:
    """Format report into human-readable console summary."""
    lines = [
        "=" * 64,
        "OhhO OS Hardware Self-Test Report",
        f"Robot: {report.robot} | Mode: {report.mode} | Overall: {report.overall_status}",
        f"Timestamp: {report.timestamp}",
        "=" * 64,
    ]
    for name, c in report.checks.items():
        lines.append(f"[{c.status:<4}] {name:<22} - {c.reason}")

    lines.append("-" * 64)
    s = report.summary
    lines.append(
        f"Summary: {s['passed']} passed, {s['failed']} failed, {s['skipped']} skipped "
        f"({s['total']} total) in {s['duration_s']:.2f}s"
    )
    if (
        report.mode == "hardware"
        and s.get("passed", 0) == 0
        and s.get("failed", 0) == 0
    ):
        lines.append(
            "WARNING: All checks were SKIPPED in hardware mode (no hardware verified)."
        )
    lines.append("=" * 64)
    return "\n".join(lines)
