"""Runtime manifest schema, class defaults, and the optional services hook."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace

from ohho.plugins import loaded_plugins
from ohho.registry import RobotSpec, builtin_specs, get_spec
from ohho.robot_spec import AerialLimits, RobotClass

from ohho_runtime import entries
from ohho_runtime.manifest import (
    CLASS_DEFAULTS,
    ROLE_AUTOPILOT_HEARTBEAT,
    ROLE_MODE_MONITOR,
    ROLE_POLICY,
    ROLE_SAFETY_WATCHDOG,
    ROLE_TELEMETRY,
    ROLE_TELEOP_GATEWAY,
    ROLE_TRANSPORT,
    ConsoleScript,
    Criticality,
    ManifestError,
    PythonCallable,
    RestartKind,
    RestartPolicy,
    Ros2Launch,
    Ros2Node,
    RuntimeManifest,
    SchedulePolicy,
    SchedulingHint,
    ServicesProvider,
    class_default_manifest,
    load_manifest,
    make_service,
    plugin_for_spec,
    resolve_manifest,
    with_services,
)
from ohho_runtime.qos import QOS_PROFILES, Reliability

_GROUND = (
    "safety_watchdog",
    "transport",
    "telemetry",
    "teleop_gateway",
)


def _noop() -> None:
    return None


def _entry() -> PythonCallable:
    return PythonCallable(_noop)


def _svc(role: str, **kwargs: object) -> object:
    params: dict[str, object] = {"loop_hz": 1.0, "qos": ("state",)}
    params.update(kwargs)
    return make_service(role, _entry(), **params)  # type: ignore[arg-type]


def _spec(robot_class: RobotClass, spec_id: str = "specimen") -> RobotSpec:
    return RobotSpec(
        id=spec_id,
        name="Specimen",
        category=robot_class.value,
        robot_class=robot_class,
    )


def _builtin(robot_class: RobotClass) -> RobotSpec:
    pool = list(builtin_specs())
    if robot_class is RobotClass.MOBILE_MANIPULATOR:
        pool.append(get_spec("omnibot"))
    if robot_class is RobotClass.QUADRUPED:
        pool.append(get_spec("unitree-go2"))
    matches = [spec for spec in pool if spec.robot_class is robot_class]
    if not matches:
        raise AssertionError(f"no built-in spec for {robot_class.value}")
    return matches[0]


class _RecordingPlugin:
    def __init__(self, manifest: RuntimeManifest | None) -> None:
        self.name = "recording"
        self.manifest = manifest
        self.calls = 0

    def services(self, spec: RobotSpec) -> RuntimeManifest | None:
        del spec
        self.calls += 1
        return self.manifest


class _BarePlugin:
    name = "bare"

    def specs(self) -> list[RobotSpec]:
        return []


class ManifestLoadTest(unittest.TestCase):
    def test_unknown_qos_profile_is_rejected(self) -> None:
        service = _svc("telemetry", qos=("not_a_profile",))
        with self.assertRaises(ManifestError) as caught:
            load_manifest([service])  # type: ignore[list-item]
        message = str(caught.exception)
        self.assertIn("unknown QoS profile", message)
        self.assertIn("not_a_profile", message)
        self.assertIn("telemetry", message)

    def test_dependency_cycle_is_rejected(self) -> None:
        alpha = _svc("alpha", depends_on=("beta",))
        beta = _svc("beta", depends_on=("alpha",))
        with self.assertRaises(ManifestError) as caught:
            load_manifest([alpha, beta])  # type: ignore[list-item]
        message = str(caught.exception)
        self.assertIn("dependency cycle", message)
        self.assertIn("alpha", message)
        self.assertIn("beta", message)

    def test_self_cycle_is_rejected(self) -> None:
        service = _svc("loop", depends_on=("loop",))
        with self.assertRaises(ManifestError) as caught:
            load_manifest([service])  # type: ignore[list-item]
        self.assertIn("dependency cycle", str(caught.exception))
        self.assertIn("loop -> loop", str(caught.exception))

    def test_unknown_dependency_is_rejected(self) -> None:
        service = _svc("alpha", depends_on=("missing",))
        with self.assertRaises(ManifestError) as caught:
            load_manifest([service])  # type: ignore[list-item]
        self.assertIn("unknown role", str(caught.exception))
        self.assertIn("missing", str(caught.exception))

    def test_duplicate_role_is_rejected(self) -> None:
        with self.assertRaises(ManifestError) as caught:
            load_manifest([_svc("alpha"), _svc("alpha")])  # type: ignore[list-item]
        self.assertIn("duplicate service role", str(caught.exception))

    def test_restart_never_rejects_restart_fields(self) -> None:
        with self.assertRaises(ManifestError) as caught:
            RestartPolicy(RestartKind.NEVER, max_restarts=1)
        self.assertIn("never", str(caught.exception))

    def test_on_failure_requires_backoff(self) -> None:
        with self.assertRaises(ManifestError) as caught:
            RestartPolicy(
                RestartKind.ON_FAILURE,
                max_restarts=1,
                window_s=10.0,
                backoff_s=0.0,
            )
        self.assertIn("backoff", str(caught.exception))

    def test_watchdog_restart_must_be_never(self) -> None:
        restart = RestartPolicy(
            RestartKind.ON_FAILURE,
            max_restarts=1,
            window_s=10.0,
            backoff_s=0.5,
        )
        with self.assertRaises(ManifestError) as caught:
            _svc(ROLE_SAFETY_WATCHDOG, restart=restart)
        self.assertIn("safety_watchdog", str(caught.exception))
        self.assertIn("never", str(caught.exception))

    def test_transport_restart_is_only_while_disarmed(self) -> None:
        restart = RestartPolicy(
            RestartKind.ON_FAILURE,
            max_restarts=1,
            window_s=10.0,
            backoff_s=0.5,
        )
        with self.assertRaises(ManifestError) as caught:
            _svc(ROLE_TRANSPORT, restart=restart)
        self.assertIn("only_while_disarmed", str(caught.exception))

    def test_required_roles_stay_critical(self) -> None:
        with self.assertRaises(ManifestError) as caught:
            _svc(ROLE_TELEMETRY, criticality=Criticality.OPTIONAL)
        self.assertIn("critical", str(caught.exception))

    def test_entry_kinds_load_without_ros(self) -> None:
        before = "rclpy" in sys.modules
        entries_under_test = (
            _entry(),
            ConsoleScript("ohho-runtime-service"),
            Ros2Launch("bringup", "bringup.launch.py"),
            Ros2Node("bringup", "gateway"),
        )
        for entry in entries_under_test:
            manifest = load_manifest(
                [
                    make_service(
                        "recorder",
                        entry,
                        criticality=Criticality.DEGRADABLE,
                        loop_hz=1.0,
                        qos=("telemetry",),
                    )
                ]
            )
            self.assertEqual(manifest.service("recorder").entry, entry)
        self.assertEqual("rclpy" in sys.modules, before)

    def test_named_qos_profiles(self) -> None:
        self.assertEqual(
            set(QOS_PROFILES),
            {"command", "state", "telemetry", "sensor"},
        )
        command = QOS_PROFILES["command"]
        self.assertEqual(command.reliability, Reliability.RELIABLE)
        self.assertEqual(command.history_depth, 1)
        self.assertIsNotNone(command.deadline_s)
        self.assertIsNone(QOS_PROFILES["sensor"].deadline_s)

    def test_scheduling_hint_rejects_a_non_positive_period(self) -> None:
        with self.assertRaises(ManifestError):
            SchedulingHint(period_s=0)
        with self.assertRaises(ManifestError):
            SchedulingHint(cpu_affinity=(-1,))

    def test_scheduling_defaults_do_not_request_an_os_change(self) -> None:
        hint = SchedulingHint()
        self.assertIs(hint.policy, SchedulePolicy.OTHER)
        self.assertIsNone(hint.nice)
        self.assertIsNone(hint.priority)
        self.assertEqual(hint.cpu_affinity, ())
        self.assertFalse(hint.mlockall)
        self.assertFalse(hint.requests_os_change())
        self.assertFalse(SchedulingHint(period_s=0.05).requests_os_change())

    def test_realtime_policy_keeps_the_declared_priority(self) -> None:
        with self.assertRaises(ManifestError):
            SchedulingHint(policy="fifo")
        with self.assertRaises(ManifestError):
            SchedulingHint(policy=SchedulePolicy.RR, priority=0)
        with self.assertRaises(ManifestError):
            SchedulingHint(policy="fifo", priority=100)
        with self.assertRaises(ManifestError):
            SchedulingHint(priority=1)
        with self.assertRaises(ManifestError):
            SchedulingHint(policy="batch", priority=1)
        with self.assertRaises(ManifestError):
            SchedulingHint(nice=20)
        with self.assertRaises(ManifestError):
            SchedulingHint(nice=True)  # type: ignore[arg-type]
        with self.assertRaises(ManifestError):
            SchedulingHint(mlockall=1)  # type: ignore[arg-type]
        hint = SchedulingHint(
            policy="rr",
            priority=2,
            nice=-20,
            mlockall=True,
            cpu_affinity=(0, 1),
        )
        self.assertIs(hint.policy, SchedulePolicy.RR)
        self.assertEqual(hint.priority, 2)
        self.assertTrue(hint.requests_os_change())


class ClassDefaultTest(unittest.TestCase):
    def test_every_robot_class_has_a_default(self) -> None:
        self.assertEqual(set(CLASS_DEFAULTS), set(RobotClass))

    def test_ground_services_and_class_extras(self) -> None:
        for robot_class in RobotClass:
            with self.subTest(robot_class=robot_class):
                manifest = class_default_manifest(robot_class)
                self.assertEqual(class_default_manifest(robot_class.value), manifest)
                roles = manifest.roles()
                self.assertEqual(roles[:4], _GROUND)
                watchdog = manifest.service(ROLE_SAFETY_WATCHDOG)
                self.assertEqual(watchdog.depends_on, ())
                self.assertEqual(watchdog.restart.kind, RestartKind.NEVER)
                self.assertEqual(watchdog.loop_hz, 20.0)
                self.assertIsNotNone(watchdog.scheduling)
                assert watchdog.scheduling is not None
                self.assertIs(watchdog.scheduling.policy, SchedulePolicy.OTHER)
                self.assertFalse(watchdog.scheduling.requests_os_change())
                self.assertFalse(watchdog.scheduling.mlockall)
                transport = manifest.service(ROLE_TRANSPORT)
                self.assertEqual(transport.depends_on, (ROLE_SAFETY_WATCHDOG,))
                self.assertTrue(transport.commands_motion)
                self.assertTrue(transport.restart.only_while_disarmed)
                self.assertTrue(manifest.service(ROLE_TELEOP_GATEWAY).commands_motion)
                self.assertFalse(watchdog.do_not_operate)
                if robot_class is RobotClass.AERIAL:
                    self.assertEqual(roles[-1], ROLE_AUTOPILOT_HEARTBEAT)
                    self.assertNotIn(ROLE_MODE_MONITOR, roles)
                elif robot_class is RobotClass.HUMANOID:
                    self.assertEqual(roles[-1], ROLE_MODE_MONITOR)
                    self.assertNotIn(ROLE_AUTOPILOT_HEARTBEAT, roles)
                else:
                    self.assertEqual(roles, _GROUND)

    def test_default_entries_do_not_command_a_robot(self) -> None:
        manifest = class_default_manifest(RobotClass.HUMANOID)
        for service in manifest.services:
            self.assertIsInstance(service.entry, PythonCallable)
            self.assertIsNone(service.entry.target())

    def test_default_entries_are_the_declared_targets(self) -> None:
        manifest = class_default_manifest(RobotClass.AERIAL)
        self.assertIs(manifest.service(ROLE_TRANSPORT).entry.target, entries.transport)
        self.assertIs(
            manifest.service(ROLE_AUTOPILOT_HEARTBEAT).entry.target,
            entries.autopilot_heartbeat,
        )

    def test_unknown_robot_class_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            class_default_manifest("submarine")


class ResolveTest(unittest.TestCase):
    def test_every_builtin_spec_uses_its_class_default(self) -> None:
        specs = list(builtin_specs())
        specs.append(get_spec("omnibot"))
        specs.append(get_spec("unitree-go2"))
        self.assertGreaterEqual(len(specs), 1)
        seen: set[RobotClass] = set()
        for spec in specs:
            self.assertIsInstance(spec.robot_class, RobotClass)
            seen.add(spec.robot_class)
            plugin = plugin_for_spec(spec)
            self.assertIsNotNone(plugin)
            self.assertIsNone(getattr(plugin, "services", None))
            self.assertNotIsInstance(plugin, ServicesProvider)
            resolved = resolve_manifest(spec)
            default = class_default_manifest(spec.robot_class)
            if spec.robot_class is RobotClass.AERIAL:
                # Built-in aircraft stay experimental and not flight_ready.
                # They still use the aerial class default; the gate only
                # sets do_not_operate on motion services.
                self.assertTrue(spec.experimental)
                self.assertFalse(spec.flight_ready)
                self.assertEqual(resolved.roles(), default.roles())
                self.assertEqual(resolved.log_level, default.log_level)
                self.assertEqual(resolved.log_format, default.log_format)
                self.assertEqual(resolved.state_dir, default.state_dir)
                motion_roles = {ROLE_TRANSPORT, ROLE_TELEOP_GATEWAY, ROLE_POLICY}
                for role in default.roles():
                    got = resolved.service(role)
                    base = default.service(role)
                    if role in motion_roles or base.commands_motion:
                        self.assertEqual(replace(got, do_not_operate=False), base)
                        self.assertTrue(got.do_not_operate)
                    else:
                        self.assertEqual(got, base)
                        self.assertFalse(got.do_not_operate)
            else:
                self.assertEqual(resolved, default)
                self.assertFalse(resolved.service(ROLE_TRANSPORT).do_not_operate)
        self.assertIn(RobotClass.MOBILE_BASE, seen)
        self.assertIn(RobotClass.MOBILE_MANIPULATOR, seen)
        self.assertIn(RobotClass.QUADRUPED, seen)
        self.assertIn(RobotClass.AERIAL, seen)

    def test_builtin_plugin_without_services_still_loads(self) -> None:
        infos = loaded_plugins()
        self.assertTrue(any(info.ok and info.name == "builtin" for info in infos))
        spec = _builtin(RobotClass.MOBILE_BASE)
        resolved = resolve_manifest(spec, _BarePlugin())
        self.assertEqual(resolved, class_default_manifest(RobotClass.MOBILE_BASE))

    def test_plugin_services_override_is_used(self) -> None:
        spec = _builtin(RobotClass.MOBILE_BASE)
        base = class_default_manifest(spec.robot_class)
        recorder = make_service(
            "recorder",
            _entry(),
            depends_on=(ROLE_TELEMETRY,),
            criticality=Criticality.DEGRADABLE,
            loop_hz=2.0,
            qos=("telemetry",),
        )
        custom = with_services(base, recorder)
        plugin = _RecordingPlugin(custom)
        self.assertEqual(plugin.calls, 0)
        class_default_manifest(RobotClass.ARM)
        self.assertEqual(plugin.calls, 0)
        resolved = resolve_manifest(spec, plugin)
        self.assertEqual(plugin.calls, 1)
        self.assertEqual(resolved, custom)
        self.assertEqual(resolved.service("recorder").loop_hz, 2.0)
        self.assertNotIn("recorder", base.roles())

    def test_services_none_keeps_the_class_default(self) -> None:
        spec = _builtin(RobotClass.QUADRUPED)
        plugin = _RecordingPlugin(None)
        resolved = resolve_manifest(spec, plugin)
        self.assertEqual(plugin.calls, 1)
        self.assertEqual(resolved, class_default_manifest(RobotClass.QUADRUPED))

    def test_plugin_cannot_drop_the_watchdog(self) -> None:
        spec = _builtin(RobotClass.MOBILE_MANIPULATOR)
        partial = load_manifest(
            [
                make_service(
                    ROLE_TRANSPORT,
                    _entry(),
                    loop_hz=1.0,
                    qos=("command",),
                    restart=RestartPolicy(RestartKind.NEVER),
                ),
                make_service(ROLE_TELEMETRY, _entry(), loop_hz=1.0, qos=("state",)),
            ]
        )
        with self.assertRaises(ManifestError) as caught:
            resolve_manifest(spec, _RecordingPlugin(partial))
        message = str(caught.exception)
        self.assertIn("safety_watchdog", message)
        self.assertIn("cannot be removed", message)

    def test_motion_service_must_depend_on_the_watchdog(self) -> None:
        spec = _spec(RobotClass.ARM, "arm-specimen")
        manifest = load_manifest(
            [
                make_service(
                    ROLE_SAFETY_WATCHDOG,
                    _entry(),
                    loop_hz=1.0,
                    qos=("state",),
                ),
                make_service(
                    ROLE_TRANSPORT,
                    _entry(),
                    loop_hz=1.0,
                    qos=("command",),
                    restart=RestartPolicy(RestartKind.NEVER),
                ),
                make_service(ROLE_TELEMETRY, _entry(), loop_hz=1.0, qos=("state",)),
            ]
        )
        with self.assertRaises(ManifestError) as caught:
            resolve_manifest(spec, _RecordingPlugin(manifest))
        self.assertIn("transport must depend on safety_watchdog", str(caught.exception))

    def test_services_return_type_is_checked(self) -> None:
        spec = _builtin(RobotClass.MOBILE_BASE)

        class Bad:
            name = "bad"

            def services(self, robot: RobotSpec) -> dict[str, str]:
                del robot
                return {"role": "transport"}

        with self.assertRaises(ManifestError) as caught:
            resolve_manifest(spec, Bad())
        self.assertIn("RuntimeManifest", str(caught.exception))

    def test_non_callable_services_attribute_is_rejected(self) -> None:
        spec = _builtin(RobotClass.MOBILE_BASE)

        class Bad:
            name = "bad"
            services = "not-a-hook"

        with self.assertRaises(ManifestError) as caught:
            resolve_manifest(spec, Bad())
        self.assertIn("not callable", str(caught.exception))

    def test_default_aerial_spec_blocks_motion_services(self) -> None:
        spec = _spec(RobotClass.AERIAL, "flyer")
        self.assertFalse(spec.flight_ready)
        self.assertFalse(spec.experimental)
        default = class_default_manifest(RobotClass.AERIAL)
        resolved = resolve_manifest(spec)
        self.assertEqual(resolved.roles(), default.roles())
        self.assertFalse(default.service(ROLE_TRANSPORT).do_not_operate)
        self.assertTrue(resolved.service(ROLE_TRANSPORT).do_not_operate)
        self.assertTrue(resolved.service(ROLE_TELEOP_GATEWAY).do_not_operate)
        self.assertFalse(resolved.service(ROLE_SAFETY_WATCHDOG).do_not_operate)
        self.assertFalse(resolved.service(ROLE_TELEMETRY).do_not_operate)
        self.assertFalse(resolved.service(ROLE_AUTOPILOT_HEARTBEAT).do_not_operate)

    def test_experimental_aerial_stays_do_not_operate(self) -> None:
        spec = RobotSpec(
            id="flyer-experimental",
            name="Flyer",
            category="aerial",
            limits=AerialLimits(flight_ready=True, experimental=True),
        )
        resolved = resolve_manifest(spec)
        self.assertTrue(resolved.service(ROLE_TRANSPORT).do_not_operate)

    def test_flight_ready_aerial_is_not_forced_do_not_operate(self) -> None:
        spec = RobotSpec(
            id="flyer-ready",
            name="Flyer",
            category="aerial",
            limits=AerialLimits(flight_ready=True, experimental=False),
        )
        resolved = resolve_manifest(spec)
        self.assertEqual(resolved, class_default_manifest(RobotClass.AERIAL))
        self.assertFalse(resolved.service(ROLE_TRANSPORT).do_not_operate)

    def test_aerial_plugin_override_is_used_and_still_gated(self) -> None:
        spec = _spec(RobotClass.AERIAL, "flyer")
        base = class_default_manifest(RobotClass.AERIAL)
        recorder = make_service(
            "recorder",
            _entry(),
            depends_on=(ROLE_TELEMETRY,),
            criticality=Criticality.DEGRADABLE,
            loop_hz=1.0,
            qos=("telemetry",),
        )
        custom = with_services(base, recorder)
        resolved = resolve_manifest(spec, _RecordingPlugin(custom))
        self.assertIn("recorder", resolved.roles())
        self.assertFalse(custom.service(ROLE_TRANSPORT).do_not_operate)
        self.assertTrue(resolved.service(ROLE_TRANSPORT).do_not_operate)
        self.assertFalse(resolved.service("recorder").do_not_operate)


if __name__ == "__main__":
    unittest.main()
