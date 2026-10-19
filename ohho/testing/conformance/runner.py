"""Per-class conformance scenarios against a plugin sim backend (OHH-315).

The kit opens a factory from ``RobotPlugin.sim_backends()`` and refuses any
transport whose protocol is not ``simulated``. It does not call
``resolve_transport`` or a hardware scheme factory.

Classes with no sim backend on this branch are registered here with
``sim_backend=None``. Running one reports skipped and a reason. That is not
a pass.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ohho.registry import RobotSpec, UnknownRobot, get_spec, list_specs
from ohho.robot_spec import RobotClass
from ohho.schema import JointReading, Odometry, Telemetry, Velocity
from ohho.transport import Transport

from .vendors import (
    format_violation,
    imported_vendor_sdks,
    load_allowlist,
    scan_repository,
    violations,
)

Status = Literal["pass", "fail", "skipped"]
_SIM_PROTOCOL = "simulated"
_ARM_TOOLS = frozenset({"arm", "release_stop"})


@dataclass(frozen=True)
class ClassScenario:
    """One class in the kit registry.

    ``sim_backend`` is the plugin sim name to open. ``None`` means the class
    is not registered: the kit skips it and does not report a pass.
    """

    robot_class: RobotClass
    name: str
    sim_backend: str | None
    reason: str


def _missing(robot_class: RobotClass, name: str) -> ClassScenario:
    return ClassScenario(
        robot_class,
        name,
        None,
        f"not registered: no {robot_class.value} sim backend on this branch",
    )


CLASS_SCENARIOS: dict[RobotClass, ClassScenario] = {
    RobotClass.MOBILE_BASE: ClassScenario(
        RobotClass.MOBILE_BASE, "drive_square", "kinematic", ""
    ),
    RobotClass.MOBILE_MANIPULATOR: ClassScenario(
        RobotClass.MOBILE_MANIPULATOR, "drive_square", "kinematic", ""
    ),
    RobotClass.ARM: ClassScenario(RobotClass.ARM, "arm_home", "kinematic", ""),
    RobotClass.QUADRUPED: _missing(RobotClass.QUADRUPED, "quadruped_walk"),
    RobotClass.AERIAL: _missing(RobotClass.AERIAL, "aerial_square"),
    RobotClass.HUMANOID: _missing(RobotClass.HUMANOID, "humanoid_walk"),
}


@dataclass(frozen=True)
class Check:
    """One conformance check."""

    name: str
    status: Status
    reason: str = ""


@dataclass(frozen=True)
class RobotResult:
    """Scenario plus plugin checks for one robot, or for a class with no robot."""

    robot_id: str
    robot_class: str
    status: Status
    reason: str
    checks: tuple[Check, ...]


def _overall(checks: Sequence[Check]) -> tuple[Status, str]:
    if any(check.status == "fail" for check in checks):
        failed = [check for check in checks if check.status == "fail"]
        return "fail", "; ".join(f"{check.name}: {check.reason}" for check in failed)
    scenario = next((check for check in checks if check.name == "scenario"), None)
    if scenario is not None and scenario.status == "skipped":
        return "skipped", scenario.reason
    if any(check.status == "pass" for check in checks):
        return "pass", ""
    return "skipped", "no checks ran"


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _telemetry_problems(sample: object) -> list[str]:
    if not isinstance(sample, Telemetry):
        return ["read() did not return Telemetry"]
    problems: list[str] = []
    if sample.odom is not None and not isinstance(sample.odom, Odometry):
        problems.append("odom is not Odometry")
    elif sample.odom is not None:
        for field in ("x", "y", "theta", "vx", "vy", "omega"):
            if not _number(getattr(sample.odom, field)):
                problems.append(f"odom.{field} is not a number")
    if not isinstance(sample.joints, list):
        problems.append("joints is not a list")
    else:
        for joint in sample.joints:
            if not isinstance(joint, JointReading) or not isinstance(joint.name, str):
                problems.append("joints must be JointReading values with string names")
                break
            if not _number(joint.position):
                problems.append(f"joint {joint.name} position is not a number")
                break
    if sample.battery is not None and not _number(sample.battery):
        problems.append("battery is not a number")
    if not isinstance(sample.custom, dict):
        problems.append("custom is not a dict")
    if not _number(sample.timestamp):
        problems.append("timestamp is not a number")
    return problems


def _check_telemetry(transport: Transport) -> Check:
    problems = _telemetry_problems(transport.read())
    if problems:
        return Check("telemetry_schema", "fail", "; ".join(problems))
    return Check("telemetry_schema", "pass")


def _step(transport: Transport, duration: float, dt: float = 0.05) -> None:
    """Advance ``step(dt)`` without writing a new command."""
    step_fn = getattr(transport, "step", None)
    if not callable(step_fn):
        raise RuntimeError("sim backend has no step(dt); refusing to run live")
    elapsed = 0.0
    while elapsed < duration - 1e-12:
        step = min(dt, duration - elapsed)
        step_fn(step)
        elapsed += step


def _integrate(
    transport: Transport, vel: Velocity, duration: float, dt: float = 0.05
) -> None:
    transport.send_velocity(vel)
    _step(transport, duration, dt)


def _pose(transport: Transport) -> tuple[float, float, float]:
    odom = transport.read().odom
    if odom is None:
        raise RuntimeError("sim telemetry has no odometry")
    return odom.x, odom.y, odom.theta


def _drive_square(transport: Transport, spec: RobotSpec) -> Check:
    if spec.max_lin <= 0 or spec.max_ang <= 0:
        return Check("drive_square", "fail", "spec has no linear or yaw speed")
    speed = min(0.25, spec.max_lin)
    yaw = min(1.0, spec.max_ang)
    side = 1.0
    turn = math.pi / 2
    expected = ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0))
    try:
        for index in range(4):
            _integrate(transport, Velocity(linear_x=speed), side / speed)
            x, y, _theta = _pose(transport)
            want_x, want_y = expected[index]
            if abs(x - want_x) > 1e-3 or abs(y - want_y) > 1e-3:
                return Check(
                    "drive_square",
                    "fail",
                    f"corner {index} was ({x:.4f}, {y:.4f}), want ({want_x}, {want_y})",
                )
            _integrate(transport, Velocity(angular_z=yaw), turn / yaw)
        _x, _y, theta = _pose(transport)
    except RuntimeError as exc:
        return Check("drive_square", "fail", str(exc))
    if abs(math.remainder(theta, 2 * math.pi)) > 1e-3:
        return Check("drive_square", "fail", f"heading after the square was {theta}")
    return Check("drive_square", "pass")


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _refuse_prompt(prompt: str) -> str:
    raise AssertionError(f"conformance tried to prompt: {prompt}")


def _fresh(transport: Transport) -> None:
    emit = getattr(transport, "_emit_telemetry", None)
    if callable(emit):
        emit(transport.read())


def _with_gate(transport: Transport, spec: RobotSpec):
    from ohho.safety import AuditLog, SafetyGate

    directory = tempfile.mkdtemp(prefix="ohho-conformance-")
    clock = _Clock()
    gate = SafetyGate(
        transport,
        spec,
        audit=AuditLog(directory),
        clock=clock,
        auto_tick=False,
        environ={},
        isatty=lambda: False,
        input_fn=_refuse_prompt,
    )
    return gate, clock, directory


def _close_gate(gate: object, directory: str) -> None:
    disconnect = getattr(gate, "disconnect", None)
    if callable(disconnect):
        try:
            disconnect()
        except Exception:
            pass
    shutil.rmtree(directory, ignore_errors=True)


def _open_connected(spec: RobotSpec, scenario: ClassScenario) -> Transport:
    transport = open_sim_backend(spec, _backend_name(spec, scenario))
    transport.connect()
    if str(getattr(transport, "protocol", "") or "") != _SIM_PROTOCOL:
        transport.disconnect()
        raise RuntimeError("refusing to command a non-sim transport after connect")
    return transport


def _watchdog_zeroes(spec: RobotSpec, scenario: ClassScenario) -> Check:
    try:
        transport = _open_connected(spec, scenario)
    except RuntimeError as exc:
        return Check("watchdog_zeroes", "fail", str(exc))
    gate, clock, directory = _with_gate(transport, spec)
    try:
        gate.connect()
        if gate.requires_hardware_arming:
            return Check(
                "watchdog_zeroes", "fail", "sim backend required hardware arming"
            )
        clock.t = 0.0
        _fresh(transport)
        if not gate.arm():
            return Check("watchdog_zeroes", "fail", "sim gate did not arm")
        speed = min(0.2, spec.max_lin, gate.profile.max_vx)
        if speed <= 0:
            return Check("watchdog_zeroes", "fail", "no speed inside the safety cap")
        cmd_t = gate.profile.cmd_timeout_s
        state_t = gate.profile.state_timeout_s
        if not cmd_t < state_t:
            return Check(
                "watchdog_zeroes",
                "fail",
                "command timeout is not stricter than the state timeout",
            )
        clock.t = 0.2
        _fresh(transport)
        gate.send_velocity(Velocity(linear_x=speed))
        _step(transport, 0.05)
        moved = _pose(transport)[0]
        if moved <= 1e-4:
            return Check(
                "watchdog_zeroes", "fail", f"base did not start moving ({moved})"
            )
        clock.t = 0.2 + (cmd_t + state_t) / 2
        gate.tick()
        if gate.latched:
            return Check(
                "watchdog_zeroes",
                "fail",
                f"command watchdog latched ({gate.latch_reason})",
            )
        held = _pose(transport)[0]
        _step(transport, 0.5)
        later = _pose(transport)[0]
        if abs(later - held) > 1e-6:
            return Check(
                "watchdog_zeroes",
                "fail",
                f"velocity was not zeroed ({held:.6f} -> {later:.6f})",
            )
        return Check("watchdog_zeroes", "pass")
    except (AssertionError, RuntimeError) as exc:
        return Check("watchdog_zeroes", "fail", str(exc))
    finally:
        _close_gate(gate, directory)


def _estop_latches(spec: RobotSpec, scenario: ClassScenario) -> Check:
    try:
        transport = _open_connected(spec, scenario)
    except RuntimeError as exc:
        return Check("estop_latches", "fail", str(exc))
    gate, clock, directory = _with_gate(transport, spec)
    try:
        gate.connect()
        clock.t = 0.0
        _fresh(transport)
        if not gate.arm():
            return Check("estop_latches", "fail", "sim gate did not arm")
        speed = min(0.2, spec.max_lin, gate.profile.max_vx)
        clock.t = 0.2
        _fresh(transport)
        gate.send_velocity(Velocity(linear_x=speed))
        _step(transport, 0.2)
        if _pose(transport)[0] <= 1e-4:
            return Check("estop_latches", "fail", "base did not start moving")
        gate.emergency_stop()
        if not gate.latched:
            return Check("estop_latches", "fail", "e-stop did not latch")
        held = _pose(transport)[0]
        gate.send_velocity(Velocity(linear_x=speed))
        _step(transport, 0.5)
        later = _pose(transport)[0]
        if abs(later - held) > 1e-6:
            return Check("estop_latches", "fail", "base moved after the latched e-stop")
        if not gate.latched:
            return Check("estop_latches", "fail", "e-stop latch cleared itself")
        return Check("estop_latches", "pass")
    except (AssertionError, RuntimeError) as exc:
        return Check("estop_latches", "fail", str(exc))
    finally:
        _close_gate(gate, directory)


def _joint_position(transport: Transport, name: str) -> float:
    for joint in transport.read().joints:
        if joint.name == name:
            return joint.position
    raise RuntimeError(f"sim telemetry has no joint {name!r}")


def _servo(transport: Transport, name: str, target: float) -> None:
    current = _joint_position(transport, name)
    transport.send_joint_command(name, target)
    duration = abs(target - current) / 2.0 + 0.1
    _step(transport, duration)


def _move_home(transport: Transport, spec: RobotSpec) -> Check:
    names = tuple(spec.joint_names)
    if not names:
        return Check("move_home", "fail", "arm spec has no joints")
    try:
        for name in names:
            _servo(transport, name, 0.4)
        for name in names:
            _servo(transport, name, 0.0)
        for name in names:
            position = _joint_position(transport, name)
            if abs(position) > 0.02:
                return Check("move_home", "fail", f"{name} homed at {position}")
    except RuntimeError as exc:
        return Check("move_home", "fail", str(exc))
    return Check("move_home", "pass")


def _joint_limit_clamp(spec: RobotSpec, scenario: ClassScenario) -> Check:
    try:
        transport = _open_connected(spec, scenario)
    except RuntimeError as exc:
        return Check("joint_limit_clamp", "fail", str(exc))
    gate, clock, directory = _with_gate(transport, spec)
    try:
        gate.connect()
        limits = gate.profile.joint_limits
        if not limits:
            return Check(
                "joint_limit_clamp",
                "fail",
                "safety profile declares no joint limits to clamp",
            )
        name = next(iter(limits))
        _lo, hi = limits[name]
        span = hi - _lo
        approach = hi - min(0.1, span / 2.0)
        _servo(transport, name, approach)
        clock.t = 0.0
        _fresh(transport)
        if not gate.arm():
            return Check("joint_limit_clamp", "fail", "sim gate did not arm")
        requested = hi + 2.0
        gate.send_joint_command(name, requested)
        _step(transport, 0.3)
        after = _joint_position(transport, name)
        if abs(after - hi) > 0.02 or after > hi + 1e-6:
            return Check(
                "joint_limit_clamp",
                "fail",
                f"{name} went to {after:.4f}, limit {hi}, request {requested}",
            )
        return Check("joint_limit_clamp", "pass")
    except (AssertionError, RuntimeError) as exc:
        return Check("joint_limit_clamp", "fail", str(exc))
    finally:
        _close_gate(gate, directory)


def _estop_torque(spec: RobotSpec, scenario: ClassScenario) -> Check:
    try:
        transport = _open_connected(spec, scenario)
    except RuntimeError as exc:
        return Check("estop_torque", "fail", str(exc))
    gate, clock, directory = _with_gate(transport, spec)
    try:
        gate.connect()
        clock.t = 0.0
        _fresh(transport)
        if not gate.arm():
            return Check("estop_torque", "fail", "sim gate did not arm")
        names = tuple(spec.joint_names)
        if not names:
            return Check("estop_torque", "fail", "arm spec has no joints")
        name = names[0]
        gate.emergency_stop()
        if not gate.latched:
            return Check("estop_torque", "fail", "e-stop did not latch")
        enabled = getattr(transport, "torque_enabled", None)
        if enabled is not False:
            return Check("estop_torque", "fail", "e-stop left torque enabled in sim")
        held = _joint_position(transport, name)
        transport.send_joint_command(name, held + 1.0)
        gate.send_joint_command(name, held + 1.0)
        _step(transport, 0.3)
        if abs(_joint_position(transport, name) - held) > 1e-3:
            return Check(
                "estop_torque", "fail", "joints moved after torque was disabled"
            )
        return Check("estop_torque", "pass")
    except (AssertionError, RuntimeError) as exc:
        return Check("estop_torque", "fail", str(exc))
    finally:
        _close_gate(gate, directory)


def open_sim_backend(spec: RobotSpec, backend: str) -> Transport:
    """Build ``backend`` from a plugin sim factory. Never a hardware scheme.

    Raises ``RuntimeError`` when the factory is missing or the result is not
    a simulator. The transport is not connected.
    """
    from ohho.plugins import loaded_plugins

    factory = None
    for info in loaded_plugins():
        if info.error:
            continue
        found = info.sim_factories.get(backend)
        if found is not None:
            factory = found
            break
    if factory is None:
        raise RuntimeError(f"sim backend {backend!r} is not registered by a plugin")
    transport = factory(spec, "", None)
    protocol = str(getattr(transport, "protocol", "") or "")
    if protocol != _SIM_PROTOCOL:
        raise RuntimeError(
            f"sim backend {backend!r} returned protocol {protocol!r}; "
            "refusing to command a non-sim transport"
        )
    return transport


def _backend_name(spec: RobotSpec, scenario: ClassScenario) -> str:
    declared = str(getattr(spec, "sim_backend", "") or "").strip().lower()
    if declared:
        return declared
    return scenario.sim_backend or ""


def _scenario_checks(spec: RobotSpec, scenario: ClassScenario) -> list[Check]:
    if scenario.sim_backend is None:
        return [Check("scenario", "skipped", scenario.reason)]
    backend = _backend_name(spec, scenario)
    try:
        transport = _open_connected(spec, scenario)
    except RuntimeError as exc:
        return [Check("sim_backend", "fail", str(exc))]
    try:
        checks = [Check("sim_backend", "pass", backend), _check_telemetry(transport)]
        if scenario.name == "arm_home":
            checks.append(_move_home(transport, spec))
        else:
            checks.append(_drive_square(transport, spec))
    finally:
        transport.disconnect()
    if scenario.name == "arm_home":
        checks.append(_joint_limit_clamp(spec, scenario))
        checks.append(_estop_torque(spec, scenario))
    else:
        checks.append(_watchdog_zeroes(spec, scenario))
        checks.append(_estop_latches(spec, scenario))
    return checks


def evaluate_mcp(
    tools: Sequence[str] | None, payloads: Sequence[Mapping[str, object]] | None
) -> list[Check]:
    """MCP payload rules. ``None`` means that surface is not on this branch."""
    checks: list[Check] = []
    if tools is None:
        checks.append(Check("mcp_cannot_arm", "skipped", "MCP is not on this branch"))
    else:
        exposed = [name for name in tools if name.casefold() in _ARM_TOOLS]
        if exposed:
            checks.append(Check("mcp_cannot_arm", "fail", f"MCP exposes {exposed}"))
        else:
            checks.append(Check("mcp_cannot_arm", "pass"))
    if payloads is None:
        checks.append(Check("mcp_backend", "skipped", "no MCP payloads on this branch"))
    elif len(payloads) == 0:
        checks.append(Check("mcp_backend", "skipped", "MCP returned no payloads"))
    else:
        missing = [
            index for index, payload in enumerate(payloads) if "backend" not in payload
        ]
        if missing:
            checks.append(
                Check("mcp_backend", "fail", f"payloads missing backend: {missing}")
            )
        else:
            checks.append(Check("mcp_backend", "pass"))
    return checks


def mcp_surface() -> tuple[list[str] | None, list[dict[str, object]] | None]:
    """Tools and payloads from ``ohho.mcp`` when that module exists."""
    if importlib.util.find_spec("ohho.mcp") is None:
        return None, None
    module = importlib.import_module("ohho.mcp")
    tools = list(module.tool_names()) if hasattr(module, "tool_names") else None
    payloads = (
        list(module.sample_payloads()) if hasattr(module, "sample_payloads") else None
    )
    return tools, payloads


def _mcp_checks() -> list[Check]:
    return evaluate_mcp(*mcp_surface())


def _plugin_module(spec: RobotSpec):
    from ohho.plugins import plugin_named

    plugin = plugin_named(str(getattr(spec, "source", "") or ""))
    if plugin is None:
        return None
    return importlib.import_module(type(plugin).__module__)


def _check_vendor_sdk(spec: RobotSpec) -> Check:
    module = _plugin_module(spec)
    if module is None:
        return Check("vendor_sdk_import", "fail", "plugin instance is not loaded")

    def _reload() -> None:
        importlib.reload(module)

    try:
        imported = imported_vendor_sdks(_reload)
    except Exception as exc:
        return Check("vendor_sdk_import", "fail", f"{type(exc).__name__}: {exc}")
    if imported:
        return Check(
            "vendor_sdk_import",
            "fail",
            "vendor SDK imported at plugin load: " + ", ".join(imported),
        )
    return Check("vendor_sdk_import", "pass")


def _check_vendor_names() -> Check:
    try:
        bad = violations(scan_repository(), load_allowlist())
    except (OSError, ValueError) as exc:
        return Check("vendor_names", "fail", str(exc))
    if not bad:
        return Check("vendor_names", "pass")
    shown = ", ".join(format_violation(hit) for hit in bad[:8])
    if len(bad) > 8:
        shown = f"{shown} (+{len(bad) - 8} more)"
    return Check("vendor_names", "fail", shown)


def _plugin_checks(spec: RobotSpec) -> list[Check]:
    try:
        return [_check_vendor_sdk(spec), _check_vendor_names(), *_mcp_checks()]
    except Exception as exc:
        return [Check("plugin_checks", "fail", f"{type(exc).__name__}: {exc}")]


def run_spec(spec: RobotSpec) -> RobotResult:
    """Run the kit for ``spec``, including a spec that is not registered."""
    robot_class = spec.robot_class
    if not isinstance(robot_class, RobotClass):
        robot_class = RobotClass(str(robot_class))
    scenario = CLASS_SCENARIOS.get(robot_class)
    if scenario is None:
        reason = f"not registered: unknown class {robot_class!r}"
        checks = (Check("scenario", "skipped", reason),)
        return RobotResult(spec.id, str(robot_class), "skipped", reason, checks)
    checks_list = [*_scenario_checks(spec, scenario), *_plugin_checks(spec)]
    status, reason = _overall(checks_list)
    return RobotResult(spec.id, robot_class.value, status, reason, tuple(checks_list))


def run_robot(robot_id: str) -> RobotResult:
    """Run the kit for a registered robot id. Raises ``UnknownRobot`` if missing."""
    return run_spec(get_spec(robot_id))


def run_all() -> list[RobotResult]:
    """Every registered robot, then classes that have no robot yet."""
    results = [run_robot(spec.id) for spec in list_specs()]
    present = {result.robot_class for result in results}
    for robot_class, scenario in CLASS_SCENARIOS.items():
        if robot_class.value in present:
            continue
        reason = scenario.reason or "registered scenario, no robot of this class"
        results.append(
            RobotResult(
                "",
                robot_class.value,
                "skipped",
                reason,
                (Check("scenario", "skipped", reason),),
            )
        )
    return results


def format_result(result: RobotResult) -> str:
    """Multi-line report. The first line is ``id (class): status``."""
    label = result.robot_id or result.robot_class
    line = f"{label} ({result.robot_class}): {result.status}"
    if result.reason:
        line = f"{line} — {result.reason}"
    lines = [line]
    for check in result.checks:
        detail = f"  {check.name}: {check.status}"
        if check.reason:
            detail = f"{detail} — {check.reason}"
        lines.append(detail)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m ohho.testing.conformance --robot <id>``."""
    parser = argparse.ArgumentParser(
        prog="python -m ohho.testing.conformance",
        description="Run per-class conformance against a plugin sim backend.",
    )
    parser.add_argument("--robot", help="registered robot id")
    parser.add_argument(
        "--all",
        action="store_true",
        help="every registered robot, plus classes with no robot",
    )
    args = parser.parse_args(argv)
    if args.all == bool(args.robot):
        parser.error("pass exactly one of --robot and --all")
    if args.all:
        results = run_all()
    else:
        try:
            results = [run_robot(args.robot)]
        except UnknownRobot as exc:
            print(str(exc), file=sys.stderr)
            return 2
    for result in results:
        print(format_result(result))
    if any(result.status == "fail" for result in results):
        return 1
    return 0
