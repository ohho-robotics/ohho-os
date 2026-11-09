"""Safety-preserving degradation (OHH-358).

Per-class cases run on the kinematic sim backend. Aerial hold or land uses
the in-process drone sim, because that backend is what implements flight
commands. The conformance scenario for aerial and humanoid has no class
backend registered here. Humanoid behaviour is checked against the public
``SafetyPolicy``. ``HumanoidPolicy`` is not on this branch.
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from ohho.adapters.drone_sim import DroneSimTransport
from ohho.capabilities import (
    AERIAL,
    BASE_DRIVE,
    BASE_HOLONOMIC,
    HUMANOID_LOCOMOTION,
    HUMANOID_POSTURE,
    LEGGED_POSE,
    MANIPULATION,
)
from ohho.registry import RobotSpec
from ohho.robot_spec import RobotClass
from ohho.safety import AuditLog, SafetyGate
from ohho.schema import Telemetry, Velocity
from ohho.testing.conformance import open_sim_backend

from ohho_runtime.degradation import (
    Degradation,
    HoldCommand,
    InvalidOutput,
    OutputSlot,
    PolicyRunner,
)
from ohho_runtime.manifest import (
    ROLE_PERCEPTION,
    ROLE_POLICY,
    ROLE_RECORDER,
    ROLE_SAFETY_WATCHDOG,
    ROLE_TELEMETRY,
    ROLE_TELEOP_GATEWAY,
    ROLE_TRANSPORT,
    Criticality,
    PythonCallable,
    RestartKind,
    RestartPolicy,
    RuntimeManifest,
    ServiceSpec,
    load_manifest,
    make_service,
)
from ohho_runtime.service import LifecycleState, ManualClock, ServiceBase
from ohho_runtime.supervisor import Supervisor

_CAPS = {
    RobotClass.MOBILE_BASE: (BASE_DRIVE, BASE_HOLONOMIC),
    RobotClass.MOBILE_MANIPULATOR: (BASE_DRIVE, BASE_HOLONOMIC, MANIPULATION),
    RobotClass.ARM: (MANIPULATION,),
    RobotClass.QUADRUPED: (BASE_DRIVE, LEGGED_POSE),
    RobotClass.AERIAL: (AERIAL, BASE_DRIVE),
    RobotClass.HUMANOID: (HUMANOID_POSTURE, HUMANOID_LOCOMOTION, BASE_DRIVE),
}


def _noop() -> None:
    return None


def _spec(robot_class: RobotClass) -> RobotSpec:
    return RobotSpec(
        id=f"degradation-{robot_class.value}",
        name="degradation sim",
        category=robot_class.value,
        robot_class=robot_class,
        capabilities=_CAPS[robot_class],
    )


def _upright() -> Telemetry:
    """Standing sample. Above the lying-height threshold, level, no fault."""
    return Telemetry(
        custom={
            "roll": 0.0,
            "pitch": 0.0,
            "body_height": 0.4,
            "humanoid": {"fall_state": "upright", "mode": "walk"},
        }
    )


def _entry() -> PythonCallable:
    return PythonCallable(_noop)


def _never(role: str, *, depends_on: tuple[str, ...] = ()) -> ServiceSpec:
    return make_service(
        role,
        _entry(),
        depends_on=depends_on,
        readiness_timeout_s=1000.0,
        liveness_timeout_s=1000.0,
        restart=RestartPolicy(RestartKind.NEVER),
        loop_hz=1.0,
        qos=("state",),
    )


def _on_failure(
    role: str,
    *,
    depends_on: tuple[str, ...] = (),
    criticality: Criticality = Criticality.CRITICAL,
    commands_motion: bool = False,
    only_while_disarmed: bool = False,
    max_restarts: int = 5,
    loop_hz: float = 10.0,
    liveness_timeout_s: float = 1000.0,
) -> ServiceSpec:
    return make_service(
        role,
        _entry(),
        depends_on=depends_on,
        criticality=criticality,
        readiness_timeout_s=1000.0,
        liveness_timeout_s=liveness_timeout_s,
        restart=RestartPolicy(
            RestartKind.ON_FAILURE,
            max_restarts=max_restarts,
            window_s=60.0,
            backoff_s=0.5,
            only_while_disarmed=only_while_disarmed,
        ),
        loop_hz=loop_hz,
        qos=("state",),
        commands_motion=commands_motion,
    )


def _manifest(policy_restarts: int = 5) -> RuntimeManifest:
    return load_manifest(
        [
            _never(ROLE_SAFETY_WATCHDOG),
            _on_failure(
                ROLE_TRANSPORT,
                depends_on=(ROLE_SAFETY_WATCHDOG,),
                commands_motion=True,
                only_while_disarmed=True,
                max_restarts=3,
            ),
            _never(ROLE_TELEMETRY, depends_on=(ROLE_TRANSPORT,)),
            _on_failure(
                ROLE_TELEOP_GATEWAY,
                depends_on=(ROLE_SAFETY_WATCHDOG, ROLE_TRANSPORT, ROLE_TELEMETRY),
                commands_motion=True,
            ),
            _on_failure(
                ROLE_POLICY,
                depends_on=(ROLE_SAFETY_WATCHDOG, ROLE_TELEMETRY),
                commands_motion=True,
                max_restarts=policy_restarts,
                loop_hz=10.0,
            ),
            _on_failure(
                ROLE_PERCEPTION,
                depends_on=(ROLE_TELEMETRY,),
                criticality=Criticality.DEGRADABLE,
            ),
            _on_failure(
                ROLE_RECORDER,
                depends_on=(ROLE_TELEMETRY,),
                criticality=Criticality.OPTIONAL,
                commands_motion=False,
            ),
        ]
    )


class _HoldService(ServiceBase):
    def __init__(
        self,
        clock: ManualClock,
        *,
        readiness_timeout_s: float,
        liveness_timeout_s: float,
        grace_s: float,
    ) -> None:
        super().__init__(
            clock=clock,
            readiness_timeout_s=readiness_timeout_s,
            liveness_timeout_s=liveness_timeout_s,
            grace_s=grace_s,
        )
        self._clock_obj = clock

    def on_start(self) -> None:
        self.mark_ready()
        self.heartbeat()

    def crash(self) -> None:
        self._fail("killed")


class _HoldFactory:
    def __init__(self, clock: ManualClock) -> None:
        self._clock = clock

    def __call__(
        self,
        spec: ServiceSpec,
        *,
        clock: Callable[[], float],
        grace_s: float,
    ) -> _HoldService:
        del spec
        if not isinstance(clock, ManualClock):
            raise AssertionError("expected ManualClock")
        return _HoldService(
            self._clock,
            readiness_timeout_s=1000.0,
            liveness_timeout_s=1000.0,
            grace_s=grace_s,
        )


class _World:
    """One gated sim, one supervisor, one degradation layer."""

    def __init__(
        self,
        test: unittest.TestCase,
        robot_class: RobotClass,
        *,
        transport: object | None = None,
        policy_restarts: int = 5,
    ) -> None:
        self.test = test
        self.clock = ManualClock()
        self.spec = _spec(robot_class)
        self.transport = (
            open_sim_backend(self.spec, "kinematic") if transport is None else transport
        )
        self.sport: list[str] = []
        self.motors: list[str] = []
        self.arm_calls: list[str] = []
        self.disarm_calls: list[str] = []
        self._spy()
        self.tmp = tempfile.TemporaryDirectory()
        test.addCleanup(self.tmp.cleanup)
        self.gate = SafetyGate(
            self.transport,
            self.spec,
            audit=AuditLog(Path(self.tmp.name)),
            clock=self.clock,
            auto_tick=False,
            environ={},
            input_fn=self._refuse_prompt,
        )
        test.addCleanup(self.gate.disconnect)
        self.degradation = Degradation(self.spec, self.gate, clock=self.clock)
        self.supervisor = Supervisor(
            _manifest(policy_restarts),
            robot_class=robot_class,
            clock=self.clock,
            grace_s=1.0,
            service_factory=_HoldFactory(self.clock),
            degradation=self.degradation,
        )

    def _refuse_prompt(self, prompt: str) -> str:
        raise AssertionError(prompt)

    def _spy(self) -> None:
        transport = self.transport
        original_sport = getattr(transport, "sport_command", None)

        def sport(name: str, *args: float) -> None:
            self.sport.append(name)
            if callable(original_sport):
                original_sport(name, *args)

        transport.sport_command = sport  # type: ignore[attr-defined]
        transport.motor_cut = lambda: self.motors.append("motor_cut")  # type: ignore[attr-defined]

    def connect(self) -> None:
        self.gate.connect()
        self._emit(_upright())

    def _emit(self, sample: Telemetry) -> None:
        emit = getattr(self.transport, "_emit_telemetry")
        emit(sample)

    def start(self) -> None:
        self.supervisor.start()

    def crash(self, role: str) -> None:
        service = self.supervisor.service(role)
        if not isinstance(service, _HoldService):
            raise AssertionError(type(service).__name__)
        service.crash()
        self.supervisor.poll()

    def restart_due(self) -> None:
        self.clock.advance(0.5)
        self._emit(_upright())
        self.supervisor.poll()

    def arm_gate(self) -> None:
        self._emit(_upright())
        original = self.gate.arm

        def wrapped() -> bool:
            self.arm_calls.append("arm")
            return original()

        self.gate.arm = wrapped  # type: ignore[method-assign]
        self.test.assertTrue(self.gate.arm())
        self.arm_calls.clear()

    def spy_disarm(self) -> None:
        original = self.gate.disarm

        def wrapped(reason: str = "operator") -> None:
            self.disarm_calls.append(reason)
            original(reason)

        self.gate.disarm = wrapped  # type: ignore[method-assign]


class OutputTests(unittest.TestCase):
    def test_invalid_marker_replaces_the_last_value_and_exposes_age(self) -> None:
        slot = OutputSlot("perception")
        value = {"objects": ["chair"]}
        slot.publish(value, 1.0)
        self.assertEqual(slot.read(1.25), value)
        self.assertAlmostEqual(slot.age_s(1.25), 0.25)
        slot.invalidate(2.0, "killed")
        marker = slot.read(2.5)
        self.assertIsInstance(marker, InvalidOutput)
        self.assertNotEqual(marker, value)
        self.assertEqual(marker.role, "perception")
        self.assertEqual(marker.reason, "killed")
        self.assertAlmostEqual(marker.age_s, 1.5)
        self.assertAlmostEqual(slot.age_s(2.5), 1.5)

    def test_policy_runner_zeroes_within_one_period_and_holds(self) -> None:
        slot = OutputSlot(ROLE_POLICY)
        runner = PolicyRunner(slot, period_s=0.05)
        command = (0.4, 0.1, 0.2)
        runner.publish(command, 0.0)
        runner.fail(0.01, "killed")
        self.assertEqual(runner.mode, "hold")
        self.assertEqual(runner.applied, HoldCommand())
        self.assertEqual(runner.applied.linear_x, 0.0)
        self.assertEqual(runner.applied.linear_y, 0.0)
        self.assertEqual(runner.applied.angular_z, 0.0)
        self.assertIsNotNone(runner.zeroed_after_s)
        self.assertLessEqual(runner.zeroed_after_s, runner.period_s)
        seen = runner.read(0.02)
        self.assertIsInstance(seen, InvalidOutput)
        self.assertNotEqual(seen, command)
        runner.publish(command, 0.03)
        self.assertIsInstance(runner.read(0.03), InvalidOutput)
        self.assertEqual(runner.applied, HoldCommand())


class DegradationTests(unittest.TestCase):
    def _world(
        self,
        robot_class: RobotClass,
        *,
        transport: object | None = None,
        policy_restarts: int = 5,
    ) -> _World:
        return _World(
            self,
            robot_class,
            transport=transport,
            policy_restarts=policy_restarts,
        )

    def test_transport_loss_latches_every_class_and_does_not_damp_upright(self) -> None:
        for robot_class in RobotClass:
            with self.subTest(robot_class=robot_class):
                world = self._world(robot_class)
                world.connect()
                world.arm_gate()
                world.start()
                self.assertTrue(world.gate.armed)
                world.crash(ROLE_TRANSPORT)
                self.assertTrue(world.gate.latched)
                self.assertFalse(world.gate.armed)
                self.assertFalse(world.degradation.arm_restored)
                self.assertEqual(world.arm_calls, [])
                self.assertEqual(world.motors, [])
                self.assertNotIn("damp", world.sport)
                self.assertNotIn("sit", world.sport)
                hooks = [(event.kind, event.role) for event in world.degradation.events]
                self.assertIn(("on_estop", ROLE_TRANSPORT), hooks)
                marker = world.degradation.output(ROLE_TRANSPORT).read(world.clock())
                # No sample was published; the failure still invalidates the slot.
                self.assertIsInstance(marker, InvalidOutput)

    def test_policy_kill_zeroes_output_and_falls_back_to_hold(self) -> None:
        world = self._world(RobotClass.MOBILE_BASE)
        world.connect()
        world.arm_gate()
        world.start()
        command = (0.3, 0.0, 0.2)
        world.degradation.publish(ROLE_POLICY, command, world.clock())
        world.clock.advance(0.1)
        world._emit(_upright())
        world.gate.send_velocity(Velocity(0.2, 0.0, 0.0))
        world.transport.step(0.1)
        moved = world.transport.read().odom.x
        self.assertGreater(moved, 0.0)
        world.crash(ROLE_POLICY)
        runner = world.degradation.policy_runner
        self.assertEqual(runner.mode, "hold")
        self.assertEqual(runner.applied, HoldCommand())
        self.assertLessEqual(runner.zeroed_after_s, runner.period_s)
        seen = runner.read(world.clock())
        self.assertIsInstance(seen, InvalidOutput)
        self.assertNotEqual(seen, command)
        self.assertGreaterEqual(seen.age_s, 0.0)
        world.transport.step(0.1)
        self.assertAlmostEqual(world.transport.read().odom.x, moved)
        self.assertFalse(world.gate.latched)
        self.assertEqual(world.motors, [])
        self.assertNotIn("damp", world.sport)
        # The dead runner must not accept the old command back.
        world.degradation.publish(ROLE_POLICY, command, world.clock())
        self.assertIsInstance(runner.read(world.clock()), InvalidOutput)
        self.assertEqual(runner.applied, HoldCommand())

    def test_perception_failure_leaves_teleop_and_disables_the_agent(self) -> None:
        world = self._world(RobotClass.MOBILE_BASE)
        world.connect()
        world.arm_gate()
        world.start()
        world.degradation.publish(
            ROLE_PERCEPTION, {"objects": ["chair"]}, world.clock()
        )
        world.degradation.publish(ROLE_TELEOP_GATEWAY, {"vx": 0.1}, world.clock())
        world.crash(ROLE_PERCEPTION)
        self.assertFalse(world.degradation.agent_enabled)
        self.assertFalse(world.degradation.accepts_agent())
        self.assertTrue(world.degradation.teleop_enabled())
        self.assertTrue(world.degradation.accepts_teleop())
        self.assertFalse(world.gate.latched)
        self.assertTrue(world.gate.armed)
        self.assertIs(
            world.supervisor.service(ROLE_TELEOP_GATEWAY).state,
            LifecycleState.READY,
        )
        self.assertIs(
            world.supervisor.service(ROLE_TRANSPORT).state,
            LifecycleState.READY,
        )
        perception = world.degradation.output(ROLE_PERCEPTION).read(world.clock())
        self.assertIsInstance(perception, InvalidOutput)
        self.assertNotEqual(perception, {"objects": ["chair"]})
        self.assertEqual(
            world.degradation.output(ROLE_TELEOP_GATEWAY).read(world.clock()),
            {"vx": 0.1},
        )
        world.clock.advance(0.1)
        world._emit(_upright())
        before = world.transport.read().odom.x
        world.gate.send_velocity(Velocity(0.2, 0.0, 0.0))
        world.transport.step(0.2)
        self.assertGreater(world.transport.read().odom.x, before)
        self.assertFalse(world.supervisor.do_not_operate)
        self.assertEqual(world.motors, [])

    def test_restart_disarms_and_requires_arm(self) -> None:
        world = self._world(RobotClass.MOBILE_MANIPULATOR)
        world.connect()
        world.spy_disarm()
        world.arm_gate()
        world.start()
        self.assertTrue(world.gate.armed)
        self.assertEqual(world.disarm_calls, [])
        world.crash(ROLE_POLICY)
        # Backoff has not elapsed, so this failure has not restarted yet.
        self.assertTrue(world.gate.armed)
        self.assertEqual(world.arm_calls, [])
        world.restart_due()
        self.assertGreaterEqual(world.degradation.restarts, 1)
        self.assertFalse(world.gate.armed)
        self.assertFalse(world.gate.latched)
        self.assertFalse(world.degradation.arm_restored)
        self.assertEqual(world.arm_calls, [])
        self.assertTrue(world.disarm_calls)
        before = world.transport.read().odom.x
        world.clock.advance(0.1)
        world._emit(_upright())
        world.gate.send_velocity(Velocity(0.2, 0.0, 0.0))
        world.transport.step(0.2)
        self.assertAlmostEqual(world.transport.read().odom.x, before)
        self.assertTrue(world.gate.arm())
        world.clock.advance(0.1)
        world._emit(_upright())
        world.gate.send_velocity(Velocity(0.2, 0.0, 0.0))
        world.transport.step(0.2)
        self.assertGreater(world.transport.read().odom.x, before)

    def test_transport_restart_stays_disarmed_until_arm(self) -> None:
        world = self._world(RobotClass.QUADRUPED)
        world.connect()
        world.arm_gate()
        world.start()
        world.crash(ROLE_TRANSPORT)
        self.assertTrue(world.gate.latched)
        self.assertFalse(world.gate.armed)
        world.restart_due()
        self.assertFalse(world.gate.armed)
        self.assertTrue(world.gate.latched)
        self.assertEqual(world.arm_calls, [])
        self.assertFalse(world.gate.arm())
        self.assertNotIn("damp", world.sport)
        self.assertNotIn("sit", world.sport)
        self.assertTrue(world.gate.release_stop())
        self.assertFalse(world.gate.armed)
        world._emit(_upright())
        self.assertTrue(world.gate.arm())

    def test_repeated_critical_failure_latches_do_not_operate(self) -> None:
        world = self._world(RobotClass.ARM, policy_restarts=5)
        world.connect()
        world.arm_gate()
        world.start()
        world.crash(ROLE_POLICY)
        self.assertFalse(world.supervisor.do_not_operate)
        world.restart_due()
        self.assertFalse(world.gate.armed)
        world.crash(ROLE_POLICY)
        self.assertTrue(world.supervisor.do_not_operate)
        self.assertTrue(world.degradation.do_not_operate())
        self.assertFalse(world.degradation.accepts_agent())
        self.assertFalse(world.degradation.accepts_teleop())
        self.assertFalse(world.degradation.arm_restored)
        self.assertEqual(world.arm_calls, [])
        self.assertGreaterEqual(world.degradation.failures(ROLE_POLICY), 2)
        self.assertEqual(world.degradation.policy_runner.mode, "hold")

    def test_watchdog_failure_is_do_not_operate_and_does_not_damp(self) -> None:
        world = self._world(RobotClass.HUMANOID)
        world.connect()
        world.arm_gate()
        world.start()
        world.crash(ROLE_SAFETY_WATCHDOG)
        self.assertTrue(world.supervisor.do_not_operate)
        self.assertFalse(world.gate.armed)
        self.assertNotIn("damp", world.sport)
        self.assertEqual(world.motors, [])
        self.assertEqual(world.arm_calls, [])
        hooks = [(event.kind, event.role) for event in world.degradation.events]
        self.assertIn(("on_watchdog", ROLE_SAFETY_WATCHDOG), hooks)

    def test_telemetry_failure_invalidates_and_latches_do_not_operate(self) -> None:
        world = self._world(RobotClass.MOBILE_BASE)
        world.connect()
        world.start()
        sample = {"odom": 1}
        world.degradation.publish(ROLE_TELEMETRY, sample, 0.0)
        world.crash(ROLE_TELEMETRY)
        seen = world.degradation.output(ROLE_TELEMETRY).read(world.clock())
        self.assertIsInstance(seen, InvalidOutput)
        self.assertNotEqual(seen, sample)
        self.assertGreaterEqual(seen.age_s, 0.0)
        self.assertTrue(world.supervisor.do_not_operate)
        self.assertFalse(world.gate.latched)
        hooks = [(event.kind, event.role) for event in world.degradation.events]
        self.assertIn(("on_watchdog", ROLE_TELEMETRY), hooks)

    def test_stale_output_is_invalid_and_transport_staleness_latches(self) -> None:
        world = self._world(RobotClass.MOBILE_BASE)
        world.connect()
        world.arm_gate()
        stale = make_service(
            ROLE_TRANSPORT,
            _entry(),
            depends_on=(ROLE_SAFETY_WATCHDOG,),
            restart=RestartPolicy(
                RestartKind.ON_FAILURE,
                max_restarts=1,
                window_s=60.0,
                backoff_s=0.5,
                only_while_disarmed=True,
            ),
            loop_hz=10.0,
            qos=("command",),
            liveness_timeout_s=0.2,
            commands_motion=True,
        )
        world.degradation.watch(stale)
        value = {"vx": 0.4}
        world.degradation.publish(ROLE_TRANSPORT, value, 0.0)
        world.degradation.observe(0.2, {ROLE_TRANSPORT: LifecycleState.READY})
        marker = world.degradation.output(ROLE_TRANSPORT).read(0.35)
        self.assertIsInstance(marker, InvalidOutput)
        self.assertEqual(marker.reason, "stale")
        self.assertNotEqual(marker, value)
        self.assertAlmostEqual(marker.age_s, 0.35)
        self.assertTrue(world.gate.latched)
        self.assertEqual(world.motors, [])

    def test_recorder_failure_leaves_the_gate_unchanged(self) -> None:
        world = self._world(RobotClass.ARM)
        world.connect()
        world.arm_gate()
        world.start()
        world.degradation.publish(ROLE_RECORDER, {"episode": 1}, world.clock())
        world.crash(ROLE_RECORDER)
        self.assertTrue(world.gate.armed)
        self.assertFalse(world.gate.latched)
        self.assertFalse(world.supervisor.do_not_operate)
        self.assertTrue(world.degradation.agent_enabled)
        seen = world.degradation.output(ROLE_RECORDER).read(world.clock())
        self.assertIsInstance(seen, InvalidOutput)
        self.assertNotEqual(seen, {"episode": 1})

    def test_aerial_failure_holds_or_lands_and_never_kills_motors(self) -> None:
        flights: list[str] = []

        def run(landed: str) -> list[str]:
            spec = _spec(RobotClass.AERIAL)
            drone = DroneSimTransport(spec)
            original = drone.send_flight

            def send_flight(action: str, **fields: object) -> None:
                flights.append(action)
                original(action, **fields)

            drone.send_flight = send_flight  # type: ignore[method-assign]
            world = self._world(RobotClass.AERIAL, transport=drone)
            world.connect()
            world._emit(
                Telemetry(
                    custom={
                        "roll": 0.0,
                        "pitch": 0.0,
                        "drone": {"landed_state": landed},
                    }
                )
            )
            world.start()
            before = len(flights)
            world.crash(ROLE_TRANSPORT)
            self.assertEqual(world.motors, [])
            self.assertTrue(world.gate.latched)
            self.assertFalse(world.gate.armed)
            return flights[before:]

        in_air = run("in_air")
        self.assertEqual(in_air, ["land"])
        unknown = run("unknown")
        self.assertEqual(unknown, ["hold", "land"])
        self.assertTrue(set(in_air + unknown) <= {"hold", "land"})

    def test_aerial_policy_failure_does_not_kill_motors(self) -> None:
        spec = _spec(RobotClass.AERIAL)
        drone = DroneSimTransport(spec)
        calls: list[str] = []
        original = drone.send_flight

        def send_flight(action: str, **fields: object) -> None:
            calls.append(action)
            original(action, **fields)

        drone.send_flight = send_flight  # type: ignore[method-assign]
        world = self._world(RobotClass.AERIAL, transport=drone)
        world.connect()
        world._emit(
            Telemetry(
                custom={
                    "drone": {"landed_state": "in_air"},
                    "roll": 0.0,
                    "pitch": 0.0,
                }
            )
        )
        world.start()
        world.crash(ROLE_POLICY)
        self.assertEqual(world.degradation.policy_runner.mode, "hold")
        self.assertEqual(world.motors, [])
        self.assertTrue(set(calls) <= {"hold", "land"})
        self.assertNotIn("damp", world.sport)

    def test_humanoid_policy_from_pr_78_is_deferred(self) -> None:
        import ohho.safety_policy as policies

        if not hasattr(policies, "HumanoidPolicy"):
            self.skipTest(
                "HumanoidPolicy is not on main (PR 78 is still open). "
                "Upright service failure is covered against the public "
                "SafetyPolicy; the class-specific hold test needs that policy."
            )


if __name__ == "__main__":
    unittest.main()
