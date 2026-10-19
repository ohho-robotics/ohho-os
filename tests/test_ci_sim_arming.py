"""Sim-only GitHub Actions arming exception (OHH-288).

Fake transports and an injected env mapping. One case constructs the real
``Px4DdsTransport`` and does not connect. No ``os.environ`` mutation, no
sleeps, and no hardware.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

from ohho.adapters.px4_dds import Px4DdsTransport
from ohho.aerial import Frame
from ohho.registry import RobotSpec, get_spec
from ohho.safety import (
    ARM_ENV,
    CI_SIM_ARMING_ENV,
    CI_SITL_ENV,
    DRONE_PROFILE,
    GO2_PROFILE,
    AuditLog,
    SafetyGate,
    ci_sim_arming_reason,
)
from ohho.schema import ConnectionState, Telemetry, TransportStatus, Velocity
from ohho.transport import BaseTransport

_MISSING = object()


def setUpModule() -> None:
    logging.getLogger("ohho.safety").addHandler(logging.NullHandler())


class FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FlightLink(BaseTransport):
    """Records flight requests. Does not open a socket."""

    def __init__(
        self,
        *,
        backend_id: str | None = "drone-sim",
        protocol: str = "simulated",
    ) -> None:
        super().__init__()
        if backend_id is not None:
            self.backend_id = backend_id
        self.protocol = protocol
        self.state = ConnectionState.CONNECTED
        self.flight: list[tuple[str, dict[str, object]]] = []
        self.velocities: list[tuple[float, float, float]] = []
        self.latest = Telemetry()

    def connect(self) -> TransportStatus:
        self.state = ConnectionState.CONNECTED
        return self.status()

    def disconnect(self) -> None:
        self.state = ConnectionState.DISCONNECTED

    def status(self) -> TransportStatus:
        return TransportStatus(protocol=self.protocol, state=self.state, label="ci")

    def read(self) -> Telemetry:
        return self.latest

    def send_velocity(self, vel: Velocity) -> None:
        self.velocities.append((vel.linear_x, vel.linear_y, vel.angular_z))

    def send_joint_command(self, name: str, position: float) -> None:
        return None

    def emergency_stop(self) -> None:
        self.velocities.append((0.0, 0.0, 0.0))

    def release_stop(self) -> None:
        return None

    def simulated_descent(self) -> None:
        return None

    def send_flight(self, action: str, **fields: object) -> None:
        self.flight.append((action, dict(fields)))

    def emit(self, tele: Telemetry) -> None:
        self.latest = tele
        self._emit_telemetry(tele)

    def actions(self) -> list[str]:
        return [name for name, _fields in self.flight]


class Px4Link(FlightLink):
    def __init__(
        self,
        uri: str,
        *,
        sim_target: bool = True,
        hardware: bool = False,
        protocol: str = "simulated",
    ) -> None:
        super().__init__(backend_id="px4-dds", protocol=protocol)
        self.uri = uri
        self._sim_target = sim_target
        self._hardware = hardware

    @property
    def sim_target(self) -> bool:
        return self._sim_target

    @property
    def requires_hardware_arming(self) -> bool:
        return self._hardware


class MavlinkLink(FlightLink):
    def __init__(
        self,
        uri: str,
        *,
        simulation_target: bool = True,
        hardware: bool = False,
        protocol: str = "simulated",
    ) -> None:
        super().__init__(backend_id="mavlink", protocol=protocol)
        self.uri = uri
        self._simulation_target = simulation_target
        self._hardware = hardware

    @property
    def simulation_target(self) -> bool:
        return self._simulation_target

    @property
    def requires_hardware_arming(self) -> bool:
        return self._hardware


class BoomFlag(FlightLink):
    def __init__(self) -> None:
        super().__init__(backend_id="px4-dds", protocol="simulated")
        self.uri = "px4+dds://127.0.0.1:8888"

    @property
    def sim_target(self) -> bool:
        raise RuntimeError("inspect")


def _spec() -> RobotSpec:
    return RobotSpec(
        id="sitl-ci",
        name="SITL CI",
        category="aerial",
        capabilities=("aerial",),
    )


def _github(**overrides: str | None) -> dict[str, str]:
    env: dict[str, str] = {
        "CI": "true",
        "GITHUB_ACTIONS": "true",
        CI_SIM_ARMING_ENV: "1",
    }
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _ready(*, battery: float | None = 0.9, link_ok: bool | None = True) -> Telemetry:
    drone: dict[str, object] = {"landed_state": "on_ground"}
    if link_ok is not None:
        drone["link_ok"] = link_ok
    if battery is not None:
        drone["battery"] = battery
    return Telemetry(battery=battery, custom={"drone": drone})


class CiCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="ohho-ci-sim-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clock = FakeClock()

    def _input(self, prompt: str) -> str:
        raise AssertionError(prompt)

    def make(
        self,
        link: FlightLink,
        env: dict[str, str],
        *,
        tty: bool = False,
        arm: bool = True,
        tele: Telemetry | None = None,
        profile: object = DRONE_PROFILE,
        spec: RobotSpec | None = None,
        audit: AuditLog | None = None,
    ) -> SafetyGate:
        gate = SafetyGate(
            link,
            spec if spec is not None else _spec(),
            profile=profile,  # type: ignore[arg-type]
            clock=self.clock,
            environ=env,
            isatty=lambda: tty,
            input_fn=self._input,
            audit=audit or AuditLog(self.tmp),
            auto_tick=False,
        )
        gate.connect()
        link.emit(tele if tele is not None else _ready())
        if arm:
            self.assertTrue(gate.arm())
        return gate

    def events(self, name: str | None = None) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for path in sorted(Path(self.tmp).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if name is None or rec["event"] == name:
                    out.append(rec)
        return out

    def assert_flies(
        self, gate: SafetyGate, link: FlightLink, *, backend: str, host: str, uri: str
    ) -> None:
        self.assertTrue(
            gate.command("offboard_velocity", 0.2, 0.0, 0.0, 0.0, Frame.BODY_FRD)
        )
        before = len(self.events("ci_sim_arming"))
        with self.assertLogs("ohho.safety", level="WARNING") as logs:
            self.assertTrue(gate.command("arm_flight"))
            self.assertTrue(gate.command("takeoff", 2.0))
        text = "\n".join(logs.output)
        self.assertEqual(text.count("CI sim-only arming exception used"), 2)
        self.assertIn("arm_flight", text)
        self.assertIn("takeoff", text)
        self.assertIn(gate.robot_id, text)
        self.assertIn(f"backend={backend}", text)
        self.assertIn(f"on {backend if host == '' else host}", text)
        if uri:
            self.assertIn(uri, text)
        self.assertIn("arm", link.actions())
        self.assertEqual(link.actions()[-1], "takeoff")
        new = self.events("ci_sim_arming")[before:]
        self.assertEqual(
            [(rec["action"], rec["backend"], rec["host"]) for rec in new],
            [("arm_flight", backend, host), ("takeoff", backend, host)],
        )

    def assert_refused(self, gate: SafetyGate, link: FlightLink, reason: str) -> None:
        before = len(self.events("ci_sim_arming"))
        with self.assertLogs("ohho.safety", level="WARNING") as logs:
            self.assertFalse(gate.command("takeoff", 2.0))
        self.assertEqual(gate.last_error, reason)
        self.assertNotIn("takeoff", link.actions())
        self.assertEqual(len(self.events("ci_sim_arming")), before)
        self.assertFalse(
            any("CI sim-only arming exception used" in line for line in logs.output)
        )


class TestCiSimAllows(CiCase):
    def test_drone_sim_arm_and_takeoff_logs_and_audits(self) -> None:
        link = FlightLink(backend_id="drone-sim")
        gate = self.make(link, _github())
        self.assert_flies(gate, link, backend="drone-sim", host="", uri="")

    def test_simulated_protocol_without_a_backend_id_is_drone_sim(self) -> None:
        link = FlightLink(backend_id=None, protocol="simulated")
        gate = self.make(link, _github())
        self.assert_flies(gate, link, backend="drone-sim", host="", uri="")

    def test_github_tokens_are_case_insensitive(self) -> None:
        env = _github(CI="True", GITHUB_ACTIONS=" TRUE ")
        link = FlightLink(backend_id="drone-sim")
        gate = self.make(link, env)
        self.assertTrue(gate.command("takeoff", 2.0))
        self.assertEqual(link.actions(), ["takeoff"])

    def test_px4_and_mavlink_loopback_with_sitl_marker(self) -> None:
        cases = (
            ("px4+dds://127.0.0.1:8888", "127.0.0.1", "px4-dds"),
            ("px4+dds://localhost:8888", "localhost", "px4-dds"),
            ("px4+dds://[::1]:8888", "::1", "px4-dds"),
            ("px4+dds://127.0.0.2:8888", "127.0.0.2", "px4-dds"),
            ("px4+dds://127.0.0.1:8888?ns=uav", "127.0.0.1", "px4-dds"),
            ("mavlink+udp://127.0.0.1:14540", "127.0.0.1", "mavlink"),
        )
        env = _github(**{CI_SITL_ENV: "1"})
        for uri, host, backend in cases:
            with self.subTest(uri=uri):
                link: FlightLink
                if backend == "px4-dds":
                    link = Px4Link(uri, sim_target=True, hardware=False)
                else:
                    link = MavlinkLink(uri, simulation_target=True, hardware=False)
                gate = self.make(link, env)
                self.assert_flies(gate, link, backend=backend, host=host, uri=uri)


class TestCiSimRefuses(CiCase):
    def test_loopback_sitl_without_the_marker(self) -> None:
        env = _github()
        for link in (
            Px4Link("px4+dds://127.0.0.1:8888"),
            MavlinkLink("mavlink+udp://127.0.0.1:14540"),
        ):
            with self.subTest(link=link.uri):
                gate = self.make(link, env)
                self.assert_refused(gate, link, "ci")

    def test_serial_radio_and_non_loopback_targets(self) -> None:
        env = _github(**{CI_SITL_ENV: "1"})
        cases = (
            Px4Link("px4+dds://192.168.1.10:8888", sim_target=True),
            Px4Link("px4+dds://[::ffff:127.0.0.1]:8888", sim_target=True),
            Px4Link("px4+dds://[::2]:8888", sim_target=True),
            MavlinkLink("mavlink+udp://10.0.0.5:14540", simulation_target=True),
            MavlinkLink("mavlink+udp://px4-sitl:14540", simulation_target=True),
            MavlinkLink("mavlink+udp://px4-sitl:14540?sitl=1", simulation_target=True),
            MavlinkLink("mavlink+serial:///dev/ttyACM0", simulation_target=True),
            MavlinkLink("mavlink+serial://COM3", simulation_target=True),
            MavlinkLink("serial://COM3", simulation_target=True),
            FlightLink(backend_id="crazyflie", protocol="simulated"),
            FlightLink(backend_id="crazyflie-sim", protocol="simulated"),
            FlightLink(backend_id="sim", protocol="simulated"),
            FlightLink(backend_id="px4-sitl", protocol="simulated"),
            FlightLink(backend_id="not-a-drone", protocol="simulated"),
        )
        crazy = cases[-5]
        crazy.uri = "radio://0/80/2M"  # type: ignore[attr-defined]
        for link in cases:
            with self.subTest(
                backend=getattr(link, "backend_id", ""), uri=getattr(link, "uri", "")
            ):
                gate = self.make(link, env)
                self.assert_refused(gate, link, "ci")

    def test_opt_in_must_be_exactly_one(self) -> None:
        samples = (
            _github(**{CI_SIM_ARMING_ENV: None}),
            _github(**{CI_SIM_ARMING_ENV: "true"}),
            _github(**{CI_SIM_ARMING_ENV: "1 "}),
            _github(**{CI_SIM_ARMING_ENV: "yes"}),
        )
        for env in samples:
            with self.subTest(opt=env.get(CI_SIM_ARMING_ENV)):
                link = FlightLink()
                gate = self.make(link, env)
                self.assert_refused(gate, link, "ci")

    def test_only_github_actions_qualifies(self) -> None:
        samples = (
            {"CI": "true", CI_SIM_ARMING_ENV: "1"},
            {"GITHUB_ACTIONS": "true", CI_SIM_ARMING_ENV: "1"},
            {"GITLAB_CI": "true", CI_SIM_ARMING_ENV: "1", CI_SITL_ENV: "1"},
            {"CI": "yes", "GITHUB_ACTIONS": "true", CI_SIM_ARMING_ENV: "1"},
        )
        for env in samples:
            with self.subTest(env=sorted(env)):
                link = FlightLink()
                gate = self.make(link, env)
                self.assert_refused(gate, link, "ci")

    def test_arm_hardware_env_any_value_denies(self) -> None:
        for value in ("1", "0", ""):
            with self.subTest(value=value):
                link = FlightLink()
                gate = self.make(link, _github(**{ARM_ENV: value}))
                self.assert_refused(gate, link, "ci")

    def test_hardware_gate_denies_even_if_the_transport_says_sim(self) -> None:
        # ``px4-dds`` with ``sim_target`` True is not a hardware gate (OHH-290).
        # A sim flag on any other protocol still is, and the CI exception must
        # not override that classification.
        link = Px4Link(
            "px4+dds://127.0.0.1:8888",
            sim_target=True,
            hardware=False,
            protocol="serial",
        )
        gate = self.make(link, _github(**{CI_SITL_ENV: "1"}), arm=False)
        self.assertTrue(gate.requires_hardware_arming)
        self.assertFalse(gate.arm())
        gate._armed = True
        self.assert_refused(gate, link, "ci")

    def test_inner_hardware_flag_denies_a_simulated_protocol(self) -> None:
        link = Px4Link(
            "px4+dds://127.0.0.1:8888",
            sim_target=True,
            hardware=True,
            protocol="simulated",
        )
        gate = self.make(link, _github(**{CI_SITL_ENV: "1"}))
        self.assertFalse(gate.requires_hardware_arming)
        self.assert_refused(gate, link, "ci")

    def test_missing_or_false_sim_flag_and_inspection_errors(self) -> None:
        env = _github(**{CI_SITL_ENV: "1"})
        bare = FlightLink(backend_id="px4-dds", protocol="simulated")
        bare.uri = "px4+dds://127.0.0.1:8888"
        false_flag = Px4Link("px4+dds://127.0.0.1:8888", sim_target=False)
        for link in (bare, false_flag, BoomFlag()):
            with self.subTest(link=type(link).__name__):
                gate = self.make(link, env)
                self.assert_refused(gate, link, "ci")

    def test_non_ci_headless_with_opt_in_is_still_no_tty(self) -> None:
        env = {CI_SIM_ARMING_ENV: "1", CI_SITL_ENV: "1"}
        link = FlightLink()
        gate = self.make(link, env, tty=False)
        self.assert_refused(gate, link, "no_tty")

    def test_non_ci_tty_does_not_use_the_exception(self) -> None:
        env = {CI_SIM_ARMING_ENV: "1", CI_SITL_ENV: "1"}
        link = FlightLink()
        gate = self.make(link, env, tty=True)
        before = len(self.events("ci_sim_arming"))
        self.assertTrue(gate.command("takeoff", 2.0))
        self.assertEqual(link.actions(), ["takeoff"])
        self.assertEqual(len(self.events("ci_sim_arming")), before)


class TestCiSimStillEnforces(CiCase):
    def test_later_checks_still_refuse_under_the_exception(self) -> None:
        env = _github()
        gate = self.make(FlightLink(), env)
        self.assertTrue(gate.command("takeoff", 3.0))
        link_clear = FlightLink()
        gate = self.make(link_clear, env)
        self.assertFalse(gate.command("takeoff", 3.01))
        self.assertEqual(gate.last_error, "geofence")
        self.assertEqual(link_clear.actions(), ["hold"])
        self.assertFalse(
            gate.command("offboard_position", 12.0, 0.0, 0.0, 0.0, Frame.LOCAL_ENU)
        )
        self.assertEqual(gate.last_error, "geofence")

        low = FlightLink()
        gate = self.make(low, env, tele=_ready(battery=0.1))
        self.assertFalse(gate.command("takeoff", 2.0))
        self.assertEqual(gate.last_error, "battery")
        self.assertNotIn("takeoff", low.actions())

        dark = FlightLink()
        gate = self.make(dark, env, tele=_ready(link_ok=False))
        self.assertFalse(gate.command("takeoff", 2.0))
        self.assertEqual(gate.last_error, "link")
        self.assertNotIn("takeoff", dark.actions())

        quiet = FlightLink()
        gate = self.make(quiet, env)
        self.assertFalse(gate.command("arm_flight"))
        self.assertEqual(gate.last_error, "no_setpoint_stream")
        self.assertNotIn("arm", quiet.actions())

        listed = FlightLink()
        gate = self.make(listed, env)
        self.assertFalse(gate.command("orbit"))
        self.assertEqual(gate.last_error, "not_allowlisted")
        self.assertFalse(gate.command("flip"))
        self.assertEqual(gate.last_error, "denied")

        capped = FlightLink()
        gate = self.make(capped, env)
        self.assertTrue(
            gate.command("offboard_velocity", 5.0, 0.0, 0.0, 0.0, Frame.BODY_FRD)
        )
        self.assertAlmostEqual(float(capped.flight[-1][1]["vx"]), 1.5)

        before = len(self.events("ci_sim_arming"))
        bad = FlightLink()
        gate = self.make(bad, env)
        self.assertFalse(gate.command("takeoff", -1.0))
        self.assertEqual(gate.last_error, "bad_arguments")
        self.assertEqual(len(self.events("ci_sim_arming")), before)

        latched = FlightLink()
        gate = self.make(latched, env)
        gate.emergency_stop()
        before = len(self.events("ci_sim_arming"))
        self.assertFalse(gate.command("takeoff", 2.0))
        self.assertEqual(gate.last_error, "estop_latched")
        self.assertEqual(len(self.events("ci_sim_arming")), before)
        self.assertNotIn("takeoff", latched.actions())

    def test_failed_audit_does_not_forward(self) -> None:
        class FailCi(AuditLog):
            def write(self, event: str, **fields: object) -> bool:
                if event == "ci_sim_arming":
                    return False
                return super().write(event, **fields)

        link = FlightLink()
        gate = self.make(link, _github(), audit=FailCi(self.tmp))
        with self.assertNoLogs("ohho.safety", level="WARNING"):
            self.assertFalse(gate.command("takeoff", 2.0))
        self.assertNotIn("takeoff", link.actions())
        self.assertFalse(gate.armed)


class TestCiSimDoesNotWidenOtherPaths(CiCase):
    def test_ground_arm_stays_refused_in_ci(self) -> None:
        spec = RobotSpec(
            id="unitree-go2",
            name="Go2",
            category="legged",
            capabilities=("base.drive", "legged.pose"),
        )
        link = FlightLink(backend_id="unitree", protocol="serial")
        gate = self.make(link, _github(), arm=False, profile=GO2_PROFILE, spec=spec)
        self.assertFalse(gate.arm())
        self.assertEqual(self.events("arm_refused")[-1]["reason"], "env")

        link = FlightLink(backend_id="unitree", protocol="serial")
        gate = self.make(
            link,
            _github(**{ARM_ENV: "1"}),
            arm=False,
            profile=GO2_PROFILE,
            spec=spec,
        )
        self.assertFalse(gate.arm())
        self.assertEqual(self.events("arm_refused")[-1]["reason"], "ci")
        self.assertFalse(gate.armed)

    def test_hardware_release_stays_refused_in_ci(self) -> None:
        link = Px4Link(
            "px4+dds://192.168.1.10:8888",
            sim_target=False,
            hardware=True,
            protocol="serial",
        )
        gate = self.make(link, _github(**{CI_SITL_ENV: "1"}), arm=False)
        gate.emergency_stop()
        self.assertFalse(gate.release_stop())
        self.assertTrue(gate.latched)
        self.assertEqual(self.events("release_refused")[-1]["reason"], "ci")

    def test_sim_release_is_unchanged(self) -> None:
        link = FlightLink()
        gate = self.make(link, _github())
        gate.emergency_stop()
        self.assertTrue(gate.release_stop())
        self.assertFalse(gate.latched)
        self.assertFalse(gate.armed)


class TestCiSimHelper(unittest.TestCase):
    def test_constants_are_exported(self) -> None:
        import ohho.safety as safety

        self.assertEqual(CI_SIM_ARMING_ENV, "OHHO_CI_SIM_ARMING")
        self.assertEqual(CI_SITL_ENV, "OHHO_CI_SITL")
        self.assertIn("CI_SIM_ARMING_ENV", safety.__all__)
        self.assertIn("CI_SITL_ENV", safety.__all__)
        self.assertIn("ci_sim_arming_reason", safety.__all__)

    def test_pure_helper_edges(self) -> None:
        good = _github(**{CI_SITL_ENV: "1"})

        class Ns:
            backend_id = "px4-dds"
            protocol = "px4-dds"
            uri = "px4+dds://127.0.0.2:8888"
            sim_target = True
            requires_hardware_arming = False

        self.assertEqual(ci_sim_arming_reason(Ns(), good, hardware=False), "")
        self.assertNotEqual(ci_sim_arming_reason(Ns(), good, hardware=True), "")

        class Mapped(Ns):
            uri = "px4+dds://[::ffff:127.0.0.1]:8888"

        self.assertNotEqual(ci_sim_arming_reason(Mapped(), good, hardware=False), "")

        class NoHost(Ns):
            uri = "px4+dds://"

        self.assertNotEqual(ci_sim_arming_reason(NoHost(), good, hardware=False), "")

        class NotString(Ns):
            uri = 8888

        self.assertNotEqual(ci_sim_arming_reason(NotString(), good, hardware=False), "")

        class Angry(dict[str, str]):
            def get(self, key: str, default: str | None = None) -> str:
                raise RuntimeError(key)

        self.assertEqual(
            ci_sim_arming_reason(Ns(), Angry(), hardware=False), "inspect_error"
        )

        bool_env: dict[str, object] = {
            "CI": True,
            "GITHUB_ACTIONS": "true",
            CI_SIM_ARMING_ENV: "1",
        }
        self.assertNotEqual(
            ci_sim_arming_reason(Ns(), bool_env, hardware=False),  # type: ignore[arg-type]
            "",
        )

        class Slotted:
            """No ``__dict__`` and no sim flag. Inspection must deny, not raise."""

            __slots__ = ("backend_id", "protocol", "uri")

            def __init__(self) -> None:
                self.backend_id = "px4-dds"
                self.protocol = "px4-dds"
                self.uri = "px4+dds://127.0.0.1:8888"

        self.assertNotEqual(ci_sim_arming_reason(Slotted(), good, hardware=False), "")

        class WrongFlag(Ns):
            simulation_target = True

            @property
            def sim_target(self) -> bool:
                raise AttributeError("no sim_target")

        # sim_target is still on the class via the property, so inspection
        # raises and the helper denies rather than trusting simulation_target.
        self.assertEqual(
            ci_sim_arming_reason(WrongFlag(), good, hardware=False), "inspect_error"
        )


class TestRealPx4DdsCiSim(unittest.TestCase):
    """``Px4DdsTransport`` itself, wrapped in ``SafetyGate`` and not connected.

    No ROS node, no agent, and no network. The constructor only parses the URI.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="ohho-ci-px4-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _prompt(self, prompt: str) -> str:
        raise AssertionError(prompt)

    def _gate(self, transport: Px4DdsTransport, env: dict[str, str]) -> SafetyGate:
        return SafetyGate(
            transport,
            get_spec("sim"),
            clock=FakeClock(),
            environ=env,
            isatty=lambda: False,
            input_fn=self._prompt,
            audit=AuditLog(self.tmp),
            auto_tick=False,
        )

    def test_loopback_is_eligible_lan_and_missing_sitl_are_refused(self) -> None:
        env = _github(**{CI_SITL_ENV: "1"})
        self.assertNotIn(ARM_ENV, env)
        loop = Px4DdsTransport(get_spec("sim"), "px4+dds://127.0.0.1:8888")
        self.assertIs(loop.status().state, ConnectionState.IDLE)
        self.assertTrue(loop.sim_target)
        gate = self._gate(loop, env)
        self.assertFalse(gate.requires_hardware_arming)
        self.assertEqual(gate.protocol, "px4-dds")
        self.assertEqual(
            ci_sim_arming_reason(loop, env, hardware=gate.requires_hardware_arming),
            "",
        )

        lan = Px4DdsTransport(get_spec("sim"), "px4+dds://192.168.1.10:8888")
        self.assertIs(lan.status().state, ConnectionState.IDLE)
        self.assertFalse(lan.sim_target)
        lan_gate = self._gate(lan, env)
        self.assertTrue(lan_gate.requires_hardware_arming)
        self.assertEqual(
            ci_sim_arming_reason(lan, env, hardware=lan_gate.requires_hardware_arming),
            "hardware_gate",
        )

        no_sitl = _github()
        self.assertNotIn(CI_SITL_ENV, no_sitl)
        self.assertEqual(
            ci_sim_arming_reason(loop, no_sitl, hardware=False),
            "sitl_opt_in",
        )


if __name__ == "__main__":
    unittest.main()
