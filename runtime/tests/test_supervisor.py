"""Supervisor: ordered start, restart limits, and reverse shutdown.

Dev mode is in-process. These tests share one clock and do not sleep, so
they stay deterministic on Linux, macOS, and Windows.
"""

from __future__ import annotations

import unittest
from collections.abc import Callable
from pathlib import Path

from ohho.registry import RobotSpec, list_specs
from ohho.robot_spec import CI_SCENARIOS, CiSim, RobotClass

from ohho_runtime.backoff import MAX_BACKOFF_S, ExponentialBackoff
from ohho_runtime.manifest import (
    ROLE_SAFETY_WATCHDOG,
    ROLE_TELEMETRY,
    ROLE_TRANSPORT,
    ConsoleScript,
    Criticality,
    PythonCallable,
    RestartKind,
    RestartPolicy,
    ServiceSpec,
    class_default_manifest,
    load_manifest,
    make_service,
    resolve_manifest,
)
from ohho_runtime.service import LifecycleState, ManualClock, ServiceBase
from ohho_runtime.supervisor import (
    ClassSafeAction,
    Supervisor,
    SupervisorError,
    SupervisorState,
    build_dependency_graph,
    class_safe_action,
)

_RUNTIME_SRC = Path(__file__).resolve().parents[1] / "src" / "ohho_runtime"


def _noop() -> None:
    return None


def _entry() -> PythonCallable:
    return PythonCallable(_noop)


def _ci_spec(robot_class: RobotClass) -> RobotSpec:
    return RobotSpec(
        id=f"ci-sim-{robot_class.value}",
        name="CI sim",
        category=robot_class.value,
        robot_class=robot_class,
        ci_sim=CiSim(enabled=True, scenario=CI_SCENARIOS[robot_class]),
    )


def _ci_sim_specs() -> list[RobotSpec]:
    """One enabled ci_sim spec per class, plus any registered opt-in."""
    specs = [_ci_spec(robot_class) for robot_class in RobotClass]
    known = {spec.id for spec in specs}
    for spec in list_specs():
        if spec.ci_sim.enabled and spec.id not in known:
            specs.append(spec)
            known.add(spec.id)
    return specs


def _watchdog() -> ServiceSpec:
    return make_service(
        ROLE_SAFETY_WATCHDOG,
        _entry(),
        readiness_timeout_s=1000.0,
        liveness_timeout_s=1000.0,
        restart=RestartPolicy(RestartKind.NEVER),
        loop_hz=1.0,
        qos=("state",),
    )


def _telemetry() -> ServiceSpec:
    return make_service(
        ROLE_TELEMETRY,
        _entry(),
        depends_on=(ROLE_SAFETY_WATCHDOG,),
        readiness_timeout_s=1000.0,
        liveness_timeout_s=1000.0,
        restart=RestartPolicy(RestartKind.NEVER),
        loop_hz=1.0,
        qos=("telemetry",),
    )


def _restartable(
    role: str,
    *,
    criticality: Criticality,
    depends_on: tuple[str, ...],
    commands_motion: bool = False,
    only_while_disarmed: bool = False,
    max_restarts: int = 3,
) -> ServiceSpec:
    return make_service(
        role,
        _entry(),
        depends_on=depends_on,
        criticality=criticality,
        readiness_timeout_s=1000.0,
        liveness_timeout_s=1000.0,
        restart=RestartPolicy(
            RestartKind.ON_FAILURE,
            max_restarts=max_restarts,
            window_s=60.0,
            backoff_s=0.5,
            only_while_disarmed=only_while_disarmed,
        ),
        loop_hz=1.0,
        qos=("state",),
        commands_motion=commands_motion,
    )


class _HoldService(ServiceBase):
    """Synchronous stand-in. Readiness is explicit. No thread and no sleep."""

    def __init__(
        self,
        role: str,
        clock: ManualClock,
        *,
        auto_ready: bool,
        stops: list[str] | None,
        advance_on_stop: float,
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
        self.role = role
        self._clock_obj = clock
        self._auto_ready = auto_ready
        self._stops = stops
        self._advance_on_stop = advance_on_stop

    def on_start(self) -> None:
        if self._auto_ready:
            self.mark_ready()
            self.heartbeat()

    def on_stop(self) -> None:
        if self._stops is not None:
            self._stops.append(self.role)
        if self._advance_on_stop:
            self._clock_obj.advance(self._advance_on_stop)

    def crash(self) -> None:
        self._fail("crashed")


class _HoldFactory:
    def __init__(
        self,
        *,
        auto_ready: bool,
        stops: list[str] | None = None,
        advance: float = 0.0,
    ) -> None:
        self._auto_ready = auto_ready
        self._stops = stops
        self._advance = advance

    def __call__(
        self,
        spec: ServiceSpec,
        *,
        clock: Callable[[], float],
        grace_s: float,
    ) -> _HoldService:
        if not isinstance(clock, ManualClock):
            raise AssertionError("hold factory requires ManualClock")
        return _HoldService(
            spec.role,
            clock,
            auto_ready=self._auto_ready,
            stops=self._stops,
            advance_on_stop=self._advance,
            readiness_timeout_s=spec.readiness_timeout_s,
            liveness_timeout_s=spec.liveness_timeout_s,
            grace_s=grace_s,
        )


def _bring_up(sup: Supervisor) -> None:
    """Mark each starting service ready until the supervisor is running.

    A service that appears in ``started_order`` before a dependency is
    ready fails the assertion. The clock does not move.
    """
    expected = list(sup.graph.start_order())
    released: set[str] = set()
    for _ in range(len(expected) * 4):
        for role in sup.started_order:
            if role in released:
                continue
            for dep in sup.graph.dependencies(role):
                if sup.service(dep).state is not LifecycleState.READY:
                    raise AssertionError(f"{role} started before {dep} was ready")
            service = sup.service(role)
            if service.state is LifecycleState.STARTING:
                service.mark_ready()
                service.heartbeat()
                released.add(role)
        sup.poll()
        ready = all(
            sup.service(role).state is LifecycleState.READY for role in expected
        )
        if list(sup.started_order) == expected and ready:
            if sup.state is not SupervisorState.RUNNING:
                raise AssertionError(f"ready services left state {sup.state}")
            return
    raise AssertionError(
        f"did not become ready: started={sup.started_order} state={sup.state}"
    )


class BackoffTest(unittest.TestCase):
    def test_delay_doubles_and_caps_at_the_link_ceiling(self) -> None:
        backoff = ExponentialBackoff(0.5)
        self.assertEqual(backoff.ceiling_s, MAX_BACKOFF_S)
        delays = [backoff.next_delay() for _ in range(7)]
        self.assertEqual(delays, [0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 10.0])
        backoff.reset()
        self.assertEqual(backoff.next_delay(), 0.5)

    def test_interval_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            ExponentialBackoff(0)


class GraphTest(unittest.TestCase):
    def test_start_order_is_dependencies_first(self) -> None:
        for robot_class in RobotClass:
            with self.subTest(robot_class=robot_class):
                manifest = class_default_manifest(robot_class)
                graph = build_dependency_graph(manifest)
                order = graph.start_order()
                self.assertEqual(order, manifest.roles())
                self.assertEqual(graph.stop_order(), tuple(reversed(order)))
                self.assertLess(
                    order.index(ROLE_SAFETY_WATCHDOG),
                    order.index(ROLE_TRANSPORT),
                )
                self.assertEqual(
                    graph.dependencies(ROLE_TRANSPORT),
                    (ROLE_SAFETY_WATCHDOG,),
                )
                self.assertIn(
                    ROLE_TRANSPORT,
                    graph.dependents(ROLE_SAFETY_WATCHDOG),
                )

    def test_unknown_role_is_rejected(self) -> None:
        graph = build_dependency_graph(class_default_manifest(RobotClass.ARM))
        with self.assertRaises(SupervisorError):
            graph.dependencies("missing")


class CiSimBringUpTest(unittest.TestCase):
    def test_each_ci_sim_manifest_starts_in_order_without_sleep(self) -> None:
        specs = _ci_sim_specs()
        self.assertGreaterEqual(len(specs), len(RobotClass))
        seen = {spec.robot_class for spec in specs}
        self.assertEqual(seen, set(RobotClass))
        for spec in specs:
            robot_class = spec.robot_class
            assert isinstance(robot_class, RobotClass)
            with self.subTest(robot=spec.id):
                self.assertTrue(spec.ci_sim.enabled)
                if spec.id.startswith("ci-sim-"):
                    self.assertEqual(spec.ci_sim.scenario, CI_SCENARIOS[robot_class])
                manifest = resolve_manifest(spec)
                clock = ManualClock()
                sup = Supervisor(
                    manifest,
                    robot_class=robot_class,
                    clock=clock,
                    grace_s=1.0,
                    mode="dev",
                    service_factory=_HoldFactory(auto_ready=False),
                )
                sup.start()
                self.assertFalse(sup.allows_motion())
                self.assertIn(ROLE_SAFETY_WATCHDOG, sup.started_order)
                self.assertNotIn(ROLE_TRANSPORT, sup.started_order)
                for role in sup.started_order:
                    self.assertEqual(sup.graph.dependencies(role), ())
                _bring_up(sup)
                self.assertEqual(clock.now, 0.0)
                self.assertEqual(sup.started_order, manifest.roles())
                starts = [event.role for event in sup.events.of("service_start")]
                self.assertEqual(starts, list(manifest.roles()))
                self.assertLess(
                    sup.started_order.index(ROLE_SAFETY_WATCHDOG),
                    sup.started_order.index(ROLE_TRANSPORT),
                )
                blocked = any(
                    service.commands_motion and service.do_not_operate
                    for service in manifest.services
                )
                self.assertEqual(sup.allows_motion(), not blocked)
                sup.shutdown()
                self.assertEqual(
                    sup.stopped_order,
                    tuple(reversed(sup.started_order)),
                )
                self.assertEqual(sup.stopped_order[-1], ROLE_SAFETY_WATCHDOG)
                self.assertEqual(sup.shutdown_elapsed_s, 0.0)
                self.assertTrue(sup.shutdown_within_grace)
                self.assertIs(sup.state, SupervisorState.STOPPED)
                for role in sup.stopped_order:
                    self.assertIs(sup.service(role).state, LifecycleState.STOPPED)

    def test_dev_factory_brings_a_manifest_up_on_start(self) -> None:
        for robot_class in (RobotClass.MOBILE_BASE, RobotClass.AERIAL):
            with self.subTest(robot_class=robot_class):
                spec = _ci_spec(robot_class)
                clock = ManualClock()
                sup = Supervisor.for_spec(spec, clock=clock, grace_s=1.0)
                manifest = resolve_manifest(spec)
                sup.start()
                self.assertEqual(clock.now, 0.0)
                self.assertEqual(sup.mode, "dev")
                self.assertIs(sup.state, SupervisorState.RUNNING)
                self.assertEqual(sup.started_order, manifest.roles())
                blocked = any(
                    service.commands_motion and service.do_not_operate
                    for service in manifest.services
                )
                self.assertEqual(sup.allows_motion(), not blocked)


class RestartPolicyTest(unittest.TestCase):
    def _degradable(self, clock: ManualClock) -> Supervisor:
        manifest = load_manifest(
            [
                _watchdog(),
                _telemetry(),
                _restartable(
                    "recorder",
                    criticality=Criticality.DEGRADABLE,
                    depends_on=(ROLE_TELEMETRY,),
                ),
            ]
        )
        return Supervisor(
            manifest,
            robot_class=RobotClass.MOBILE_BASE,
            clock=clock,
            grace_s=1.0,
            service_factory=_HoldFactory(auto_ready=True),
        )

    def _crash_until_limit(
        self, sup: Supervisor, role: str, clock: ManualClock
    ) -> None:
        for index in range(3):
            service = sup.service(role)
            assert isinstance(service, _HoldService)
            service.crash()
            sup.poll()
            if index == 2:
                break
            scheduled = sup.events.of("restart_scheduled")
            self.assertEqual(len(scheduled), index + 1)
            self.assertIs(service.state, LifecycleState.FAILED)
            clock.advance(float(scheduled[-1].detail))
            sup.poll()
            self.assertIs(sup.service(role).state, LifecycleState.READY)

    def test_degradable_stops_after_three_failures_in_60s(self) -> None:
        clock = ManualClock()
        sup = self._degradable(clock)
        sup.start()
        self.assertIs(sup.state, SupervisorState.RUNNING)
        self.assertTrue(sup.allows_motion())
        self._crash_until_limit(sup, "recorder", clock)
        delays = [event.detail for event in sup.events.of("restart_scheduled")]
        self.assertEqual(delays, ["0.5", "1.0"])
        self.assertEqual(len(sup.events.of("restart")), 2)
        self.assertEqual(len(sup.events.of("restart_limit")), 1)
        self.assertIn("3 failures in 60.0s", sup.events.of("restart_limit")[0].detail)
        self.assertTrue(sup.restart_exhausted("recorder"))
        self.assertIs(sup.service("recorder").state, LifecycleState.FAILED)
        clock.advance(30.0)
        sup.poll()
        self.assertEqual(sup.restart_count("recorder"), 2)
        self.assertIs(sup.service("recorder").state, LifecycleState.FAILED)
        self.assertIs(sup.service(ROLE_SAFETY_WATCHDOG).state, LifecycleState.READY)
        self.assertIs(sup.service(ROLE_TELEMETRY).state, LifecycleState.READY)
        self.assertIs(sup.state, SupervisorState.RUNNING)
        self.assertFalse(sup.do_not_operate)
        self.assertFalse(sup.holding_safe_state)
        self.assertTrue(sup.allows_motion())

    def test_failures_outside_the_window_do_not_stack(self) -> None:
        clock = ManualClock()
        sup = self._degradable(clock)
        sup.start()
        service = sup.service("recorder")
        assert isinstance(service, _HoldService)
        service.crash()
        sup.poll()
        self.assertEqual(sup.events.of("restart_scheduled")[-1].detail, "0.5")
        clock.advance(0.5)
        sup.poll()
        self.assertIs(sup.service("recorder").state, LifecycleState.READY)
        clock.advance(60.5)
        again = sup.service("recorder")
        assert isinstance(again, _HoldService)
        again.crash()
        sup.poll()
        self.assertEqual(sup.events.of("restart_scheduled")[-1].detail, "0.5")
        self.assertFalse(sup.restart_exhausted("recorder"))
        self.assertFalse(sup.do_not_operate)

    def test_critical_limit_latches_do_not_operate_for_every_class(self) -> None:
        for robot_class in RobotClass:
            with self.subTest(robot_class=robot_class):
                clock = ManualClock()
                notices: list[object] = []
                manifest = load_manifest(
                    [
                        _watchdog(),
                        _telemetry(),
                        _restartable(
                            "policy",
                            criticality=Criticality.CRITICAL,
                            depends_on=(ROLE_SAFETY_WATCHDOG,),
                            commands_motion=True,
                        ),
                    ]
                )
                sup = Supervisor(
                    manifest,
                    robot_class=robot_class,
                    clock=clock,
                    grace_s=1.0,
                    service_factory=_HoldFactory(auto_ready=True),
                    on_do_not_operate=notices.append,
                )
                sup.start()
                self.assertTrue(sup.allows_motion())
                self.assertIs(sup.safe_action, class_safe_action(robot_class))
                self.assertFalse(sup.holding_safe_state)
                self._crash_until_limit(sup, "policy", clock)
                self.assertIs(sup.state, SupervisorState.DO_NOT_OPERATE)
                self.assertTrue(sup.do_not_operate)
                self.assertTrue(sup.holding_safe_state)
                self.assertFalse(sup.allows_motion())
                self.assertEqual(len(notices), 1)
                notice = notices[0]
                self.assertEqual(notice.robot_class, robot_class)  # type: ignore[attr-defined]
                self.assertEqual(notice.safe_action, class_safe_action(robot_class))  # type: ignore[attr-defined]
                self.assertEqual(notice.role, "policy")  # type: ignore[attr-defined]
                self.assertFalse(notice.allows_motion)  # type: ignore[attr-defined]
                self.assertEqual(
                    sup.events.of("safe_hold")[-1].detail,
                    class_safe_action(robot_class).value,
                )
                self.assertIs(
                    sup.service(ROLE_SAFETY_WATCHDOG).state,
                    LifecycleState.READY,
                )
                self.assertIs(sup.service(ROLE_TELEMETRY).state, LifecycleState.READY)
                self.assertIs(sup.service("policy").state, LifecycleState.FAILED)
                clock.advance(10.0)
                sup.poll()
                self.assertTrue(sup.do_not_operate)
                self.assertEqual(len(notices), 1)
                self.assertEqual(sup.restart_count("policy"), 2)

    def test_human_clear_does_not_restart_or_allow_motion(self) -> None:
        clock = ManualClock()
        notices: list[object] = []
        manifest = load_manifest(
            [
                _watchdog(),
                _restartable(
                    "policy",
                    criticality=Criticality.CRITICAL,
                    depends_on=(ROLE_SAFETY_WATCHDOG,),
                    commands_motion=True,
                ),
            ]
        )
        sup = Supervisor(
            manifest,
            robot_class=RobotClass.QUADRUPED,
            clock=clock,
            grace_s=1.0,
            service_factory=_HoldFactory(auto_ready=True),
            on_do_not_operate=notices.append,
        )
        sup.start()
        self._crash_until_limit(sup, "policy", clock)
        self.assertIs(sup.safe_action, ClassSafeAction.STOP)
        clock.advance(30.0)
        sup.poll()
        self.assertTrue(sup.do_not_operate)
        self.assertEqual(len(notices), 1)
        sup.clear_do_not_operate()
        self.assertFalse(sup.do_not_operate)
        self.assertFalse(sup.holding_safe_state)
        self.assertIs(sup.state, SupervisorState.RUNNING)
        self.assertFalse(sup.allows_motion())
        self.assertIs(sup.service("policy").state, LifecycleState.FAILED)
        self.assertEqual(sup.restart_count("policy"), 2)
        self.assertTrue(sup.events.of("human_clear"))
        with self.assertRaises(SupervisorError):
            sup.clear_do_not_operate()

    def test_critical_restart_is_refused_while_not_disarmed(self) -> None:
        clock = ManualClock()
        manifest = load_manifest(
            [
                _watchdog(),
                _restartable(
                    ROLE_TRANSPORT,
                    criticality=Criticality.CRITICAL,
                    depends_on=(ROLE_SAFETY_WATCHDOG,),
                    commands_motion=True,
                    only_while_disarmed=True,
                ),
            ]
        )
        sup = Supervisor(
            manifest,
            robot_class=RobotClass.MOBILE_BASE,
            clock=clock,
            grace_s=1.0,
            service_factory=_HoldFactory(auto_ready=True),
        )
        sup.start()
        self.assertTrue(sup.disarmed)
        self.assertTrue(sup.allows_motion())
        sup.set_disarmed(False)
        service = sup.service(ROLE_TRANSPORT)
        assert isinstance(service, _HoldService)
        service.crash()
        sup.poll()
        self.assertEqual(sup.events.of("restart_scheduled"), ())
        self.assertEqual(sup.restart_count(ROLE_TRANSPORT), 0)
        self.assertTrue(sup.restart_exhausted(ROLE_TRANSPORT))
        self.assertTrue(sup.do_not_operate)
        self.assertTrue(sup.holding_safe_state)
        self.assertIs(sup.safe_action, ClassSafeAction.STOP)
        self.assertFalse(sup.allows_motion())
        self.assertIs(sup.service(ROLE_SAFETY_WATCHDOG).state, LifecycleState.READY)

    def test_watchdog_never_restarts_and_blocks_dependents(self) -> None:
        clock = ManualClock()
        manifest = class_default_manifest(RobotClass.MOBILE_BASE)
        sup = Supervisor(
            manifest,
            robot_class=RobotClass.MOBILE_BASE,
            clock=clock,
            grace_s=1.0,
            service_factory=_HoldFactory(auto_ready=False),
        )
        sup.start()
        self.assertEqual(sup.started_order, (ROLE_SAFETY_WATCHDOG,))
        service = sup.service(ROLE_SAFETY_WATCHDOG)
        assert isinstance(service, _HoldService)
        service.crash()
        sup.poll()
        self.assertTrue(sup.restart_exhausted(ROLE_SAFETY_WATCHDOG))
        self.assertEqual(sup.restart_count(ROLE_SAFETY_WATCHDOG), 0)
        self.assertNotIn(ROLE_TRANSPORT, sup.started_order)
        self.assertTrue(sup.events.of("blocked"))
        self.assertFalse(sup.do_not_operate)
        self.assertFalse(sup.allows_motion())
        clock.advance(5.0)
        sup.poll()
        self.assertEqual(sup.restart_count(ROLE_SAFETY_WATCHDOG), 0)
        self.assertIs(sup.state, SupervisorState.STARTING)


class ShutdownTest(unittest.TestCase):
    def test_shutdown_is_reverse_and_within_grace(self) -> None:
        clock = ManualClock()
        stops: list[str] = []
        manifest = class_default_manifest(RobotClass.MOBILE_BASE)
        sup = Supervisor(
            manifest,
            robot_class=RobotClass.MOBILE_BASE,
            clock=clock,
            grace_s=1.0,
            service_factory=_HoldFactory(auto_ready=True, stops=stops, advance=0.1),
        )
        sup.start()
        self.assertIs(sup.state, SupervisorState.RUNNING)
        sup.shutdown()
        self.assertEqual(stops, list(reversed(sup.started_order)))
        self.assertEqual(sup.stopped_order, tuple(reversed(sup.started_order)))
        self.assertEqual(sup.stopped_order[-1], ROLE_SAFETY_WATCHDOG)
        self.assertAlmostEqual(sup.shutdown_elapsed_s, 0.4)
        self.assertLessEqual(sup.shutdown_elapsed_s, sup.grace_s)
        self.assertTrue(sup.shutdown_within_grace)
        self.assertEqual(sup.events.of("grace_exceeded"), ())
        self.assertEqual(sup.events.of("shutdown_done")[-1].detail, "within_grace")
        self.assertIs(sup.state, SupervisorState.STOPPED)

    def test_shutdown_past_grace_still_finishes_in_order(self) -> None:
        clock = ManualClock()
        manifest = class_default_manifest(RobotClass.MOBILE_MANIPULATOR)
        sup = Supervisor(
            manifest,
            robot_class=RobotClass.MOBILE_MANIPULATOR,
            clock=clock,
            grace_s=0.3,
            service_factory=_HoldFactory(auto_ready=True, advance=0.4),
        )
        sup.start()
        sup.shutdown()
        self.assertEqual(sup.stopped_order, tuple(reversed(sup.started_order)))
        self.assertEqual(sup.stopped_order[-1], ROLE_SAFETY_WATCHDOG)
        self.assertGreater(sup.shutdown_elapsed_s, sup.grace_s)
        self.assertFalse(sup.shutdown_within_grace)
        self.assertTrue(sup.events.of("grace_exceeded"))
        self.assertIs(sup.state, SupervisorState.STOPPED)

    def test_shutdown_stops_only_services_that_started(self) -> None:
        clock = ManualClock()
        manifest = class_default_manifest(RobotClass.HUMANOID)
        sup = Supervisor(
            manifest,
            robot_class=RobotClass.HUMANOID,
            clock=clock,
            grace_s=1.0,
            service_factory=_HoldFactory(auto_ready=False),
        )
        sup.start()
        sup.shutdown()
        self.assertEqual(sup.stopped_order, (ROLE_SAFETY_WATCHDOG,))
        self.assertIs(sup.service(ROLE_TRANSPORT).state, LifecycleState.CREATED)
        self.assertEqual(clock.now, 0.0)
        self.assertTrue(sup.shutdown_within_grace)


class DevModeTest(unittest.TestCase):
    def test_dev_mode_rejects_a_process_entry_and_unknown_modes(self) -> None:
        manifest = load_manifest(
            [
                make_service(
                    ROLE_SAFETY_WATCHDOG,
                    ConsoleScript("ohho-runtime-service"),
                    loop_hz=1.0,
                    qos=("state",),
                )
            ]
        )
        with self.assertRaises(SupervisorError) as caught:
            Supervisor(
                manifest,
                robot_class=RobotClass.MOBILE_BASE,
                clock=ManualClock(),
            )
        self.assertIn("dev mode", str(caught.exception))
        with self.assertRaises(SupervisorError):
            Supervisor(
                class_default_manifest(RobotClass.MOBILE_BASE),
                robot_class=RobotClass.MOBILE_BASE,
                clock=ManualClock(),
                mode="subprocess",
            )

    def test_sources_do_not_sleep_or_use_posix_process_control(self) -> None:
        banned = ("sleep(", "subprocess", "AF_UNIX", "SIGTERM", "os.kill", "fork(")
        for name in ("supervisor.py", "backoff.py"):
            text = (_RUNTIME_SRC / name).read_text(encoding="utf-8")
            for token in banned:
                self.assertNotIn(token, text, f"{name} contains {token}")

    def test_class_safe_actions_match_the_failure_table(self) -> None:
        self.assertIs(class_safe_action(RobotClass.MOBILE_BASE), ClassSafeAction.STOP)
        self.assertIs(
            class_safe_action(RobotClass.MOBILE_MANIPULATOR),
            ClassSafeAction.STOP_BASE_HOLD_ARM,
        )
        self.assertIs(class_safe_action(RobotClass.ARM), ClassSafeAction.HOLD)
        self.assertIs(class_safe_action(RobotClass.QUADRUPED), ClassSafeAction.STOP)
        self.assertIs(
            class_safe_action(RobotClass.AERIAL),
            ClassSafeAction.DO_NOT_OPERATE,
        )
        self.assertIs(class_safe_action(RobotClass.HUMANOID), ClassSafeAction.STOP)


if __name__ == "__main__":
    unittest.main()
