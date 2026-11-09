"""OHH-285 e-stop decision, force-disarm ban, and simulation-target refusal."""

from __future__ import annotations

import math
import socket
import unittest
from collections.abc import Callable

from ohho.drone import LandedState, command_is_forbidden, estop_commands
from ohho.drone.estop import (
    DISARM_ALTITUDE_M,
    DISARM_COMMAND,
    HOLD_COMMAND,
    LAND_COMMAND,
    MAV_CMD_DO_FLIGHTTERMINATION,
    MAV_FORCE_DISARM_MAGIC,
    PX4_MODE_AUTO_LOITER,
    EstopCommand,
    disarm_allowed,
    is_disarm_command,
)
from ohho.drone.mock_link import (
    MockCommandLog,
    simulation_refusal_reason,
    simulation_target_allowed,
)
from ohho.drone.telemetry import DronePose, DroneTelemetry

_STATES: tuple[object, ...] = (
    LandedState.ON_GROUND,
    LandedState.IN_AIR,
    LandedState.TAKEOFF,
    LandedState.LANDING,
    LandedState.UNKNOWN,
    "on_ground",
    "in_air",
    "takeoff",
    "landing",
    "unknown",
    "IN_AIR",
    " on_ground ",
    "flying",
    "2",
    None,
    0,
    1,
    2,
    3,
    4,
    9,
    True,
    False,
)

_ALTITUDES: tuple[object, ...] = (
    None,
    0,
    0.0,
    0.2,
    0.25,
    math.nextafter(0.25, math.inf),
    0.2500001,
    0.26,
    1,
    2.5,
    10.0,
    100.0,
    -0.01,
    -5.0,
    False,
    True,
    "0.0",
    "high",
    float("nan"),
    float("inf"),
    float("-inf"),
)

_HOSTS: tuple[tuple[str, bool, bool], ...] = (
    ("127.0.0.1", False, True),
    ("127.0.0.1", True, True),
    ("127.8.8.8", False, True),
    ("::1", False, True),
    ("[::1]", False, True),
    ("::ffff:127.0.0.1", False, True),
    ("localhost", False, True),
    ("LOCALHOST", False, True),
    (" localhost ", False, True),
    ("::ffff:8.8.8.8", False, False),
    ("::", False, False),
    ("0.0.0.0", False, False),
    ("192.0.2.10", False, False),
    ("10.1.2.3", True, False),
    ("8.8.8.8", True, False),
    ("198.51.100.20", True, False),
    ("192.168.1.20", True, False),
    ("203.0.113.9", True, False),
    ("example.com", False, False),
    ("example.com", True, False),
    ("", False, False),
    ("host.docker.internal", False, False),
    ("host.docker.internal", True, True),
    ("Host.Docker.Internal", True, True),
    ("gateway.docker.internal", True, True),
    ("host.containers.internal", True, True),
    ("foo.docker.internal", True, True),
    ("foo.docker.internal", False, False),
    ("foo.containers.internal", True, True),
    ("host.docker.internal.evil.test", True, False),
)

_NAMES = {"land": 21, "hold": 176, "disarm": 400}


def _spec_landed_state(value: object) -> str:
    """Independent reading of the landed-state table. Not the library helper."""
    if isinstance(value, LandedState):
        return value.value
    if value is None or isinstance(value, bool):
        return "unknown"
    if isinstance(value, int):
        return {
            0: "unknown",
            1: "on_ground",
            2: "in_air",
            3: "takeoff",
            4: "landing",
        }.get(value, "unknown")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"on_ground", "in_air", "takeoff", "landing", "unknown"}:
            return text
        return "unknown"
    return "unknown"


def _spec_altitude_allows_disarm(altitude_m: object) -> bool:
    """Finite metres at or below 0.25. Missing and booleans are not zero."""
    if isinstance(altitude_m, bool) or not isinstance(altitude_m, (int, float)):
        return False
    if not math.isfinite(altitude_m):
        return False
    return float(altitude_m) <= 0.25


def _expected_names(landed_state: object, altitude_m: object) -> list[str]:
    kind = _spec_landed_state(landed_state)
    if kind in {"in_air", "takeoff", "landing"}:
        return ["land"]
    if kind == "on_ground" and _spec_altitude_allows_disarm(altitude_m):
        return ["disarm"]
    if kind == "on_ground":
        return ["land"]
    return ["hold", "land"]


class TestDroneEstop(unittest.TestCase):
    def test_constants_match_the_mavlink_ids(self) -> None:
        self.assertEqual(DISARM_ALTITUDE_M, 0.25)
        self.assertEqual(MAV_FORCE_DISARM_MAGIC, 21196)
        self.assertEqual(MAV_CMD_DO_FLIGHTTERMINATION, 185)
        self.assertEqual(PX4_MODE_AUTO_LOITER, (3 << 24) | (4 << 16))
        self.assertNotEqual(PX4_MODE_AUTO_LOITER, 21196)
        self.assertFalse(command_is_forbidden(LAND_COMMAND))
        self.assertFalse(command_is_forbidden(HOLD_COMMAND))
        self.assertFalse(command_is_forbidden(DISARM_COMMAND))

    def test_in_air_takeoff_and_landing_land_without_disarm(self) -> None:
        for landed in (LandedState.IN_AIR, LandedState.TAKEOFF, LandedState.LANDING):
            for altitude in (None, 0.0, 0.25, 3.0):
                commands = estop_commands(landed, altitude)
                self.assertEqual([command.name for command in commands], ["land"])
                self.assertIs(commands[0], LAND_COMMAND)
                self.assertFalse(
                    any(is_disarm_command(command) for command in commands)
                )

    def test_unknown_holds_then_lands_at_any_altitude(self) -> None:
        for altitude in (None, 0.0, 0.25, 10.0):
            commands = estop_commands(LandedState.UNKNOWN, altitude)
            self.assertEqual([command.name for command in commands], ["hold", "land"])
            self.assertIs(commands[0], HOLD_COMMAND)
            self.assertIs(commands[1], LAND_COMMAND)
            self.assertEqual(commands[0].mav_command, 176)
            self.assertEqual(commands[0].param1, 1.0)
            self.assertEqual(int(commands[0].param2), PX4_MODE_AUTO_LOITER)
            self.assertEqual(commands[0].param3, 3.0)
            self.assertNotEqual(int(commands[0].param2), 21196)

    def test_on_ground_disarms_only_at_or_below_the_threshold(self) -> None:
        at = estop_commands(LandedState.ON_GROUND, 0.25)
        self.assertEqual([command.name for command in at], ["disarm"])
        self.assertIs(at[0], DISARM_COMMAND)
        self.assertEqual(at[0].mav_command, 400)
        self.assertEqual(at[0].param1, 0.0)
        self.assertEqual(at[0].param2, 0.0)
        self.assertNotEqual(at[0].param2, 21196)
        above = estop_commands(LandedState.ON_GROUND, math.nextafter(0.25, math.inf))
        self.assertEqual([command.name for command in above], ["land"])
        self.assertIs(above[0], LAND_COMMAND)
        missing = estop_commands(LandedState.ON_GROUND)
        self.assertEqual([command.name for command in missing], ["land"])
        for altitude in (None, float("nan"), float("inf"), False, "0.0"):
            commands = estop_commands(LandedState.ON_GROUND, altitude)
            self.assertEqual([command.name for command in commands], ["land"])
            self.assertFalse(any(command.name == "disarm" for command in commands))

    def test_ned_and_enu_height_gate_the_disarm(self) -> None:
        high = DroneTelemetry(
            pose_frame="LOCAL_NED",
            velocity_frame="LOCAL_NED",
            landed_state=LandedState.ON_GROUND,
            pose=DronePose(z=-2.0),
        )
        self.assertEqual(high.local_altitude_up_m(), 2.0)
        self.assertEqual(
            [
                command.name
                for command in estop_commands(
                    high.landed_state, high.local_altitude_up_m()
                )
            ],
            ["land"],
        )
        low = DroneTelemetry(
            pose_frame="LOCAL_NED",
            velocity_frame="LOCAL_NED",
            landed_state="on_ground",
            pose=DronePose(z=0.0),
        )
        low_commands = estop_commands(low.landed_state, low.local_altitude_up_m())
        self.assertEqual([command.name for command in low_commands], ["disarm"])
        self.assertEqual(low_commands[0].param2, 0.0)
        enu = DroneTelemetry(
            pose_frame="LOCAL_ENU",
            velocity_frame="LOCAL_ENU",
            landed_state=LandedState.ON_GROUND,
            pose=DronePose(z=0.2),
        )
        self.assertEqual(enu.local_altitude_up_m(), 0.2)
        self.assertEqual(
            [
                command.name
                for command in estop_commands(
                    enu.landed_state, enu.local_altitude_up_m()
                )
            ],
            ["disarm"],
        )

    def test_property_no_disarm_no_force_magic_no_termination(self) -> None:
        log = MockCommandLog(host="127.0.0.1")
        self.assertTrue(log.connect().allowed)
        for landed_state in _STATES:
            for altitude_m in _ALTITUDES:
                with self.subTest(landed_state=landed_state, altitude_m=altitude_m):
                    commands = estop_commands(landed_state, altitude_m)
                    names = [command.name for command in commands]
                    self.assertEqual(names, _expected_names(landed_state, altitude_m))
                    self.assertGreaterEqual(len(commands), 1)
                    self.assertEqual(
                        disarm_allowed(landed_state, altitude_m),
                        _spec_landed_state(landed_state) == "on_ground"
                        and _spec_altitude_allows_disarm(altitude_m),
                    )
                    for command in commands:
                        self._assert_command_safe(command)
                        self.assertEqual(
                            is_disarm_command(command), command.name == "disarm"
                        )
                        if command.name == "disarm" or command.mav_command == 400:
                            self.assertEqual(
                                _spec_landed_state(landed_state), "on_ground"
                            )
                            self.assertTrue(_spec_altitude_allows_disarm(altitude_m))
                            self.assertLessEqual(float(altitude_m), 0.25)  # type: ignore[arg-type]
                    if "disarm" in names:
                        self.assertEqual(names, ["disarm"])
                    if "hold" in names:
                        self.assertEqual(names, ["hold", "land"])
                    before = len(log.received_commands())
                    emitted = log.apply_estop(landed_state, altitude_m)
                    self.assertEqual(emitted, commands)
                    self.assertEqual(log.received_commands()[before:], commands)
                    for recorded in log.received_tuples()[before:]:
                        self.assertNotEqual(recorded[1], 185)
                        self.assertNotEqual(recorded[3], 21196)

    def test_decision_does_not_write_the_log_or_the_network(self) -> None:
        log = MockCommandLog(host="127.0.0.1")
        self.assertTrue(log.connect().allowed)
        estop_commands(LandedState.IN_AIR, 3.0)
        self.assertEqual(log.received_commands(), ())
        self._call_under_broken_socket(
            lambda: [estop_commands(state, 1.0) for state in LandedState]
        )

    def test_injected_force_disarm_and_termination_are_dropped(self) -> None:
        log = MockCommandLog(host="127.0.0.1")
        self.assertTrue(log.connect().allowed)
        injected = (
            EstopCommand("disarm", 400, 0.0, 21196.0),
            EstopCommand("disarm", 400, 0.0, float(MAV_FORCE_DISARM_MAGIC)),
            EstopCommand("flight_termination", 185, 1.0, 0.0),
            EstopCommand("land", 185, 0.0, 0.0),
            EstopCommand("kill", 21, 0.0, 0.0),
            EstopCommand("land", 21, 0.0, 21196.0),
            EstopCommand("force-disarm", 400, 0.0, 0.0),
            EstopCommand("terminate", 21, 0.0, 0.0),
        )
        for command in injected:
            self.assertTrue(command_is_forbidden(command))
            self.assertFalse(is_disarm_command(command))
            self.assertFalse(log.record(command))
        self.assertEqual(log.received_commands(), ())
        self.assertEqual(log.dropped_forbidden, len(injected))
        self.assertTrue(log.record(DISARM_COMMAND))
        self.assertEqual(log.received_commands(), (DISARM_COMMAND,))
        name, mav_command, param1, param2 = log.received_tuples()[-1]
        self.assertEqual((name, mav_command, param1, param2), ("disarm", 400, 0.0, 0.0))
        self.assertNotEqual(param2, 21196)

    def test_refused_target_does_not_emit_disarm(self) -> None:
        log = MockCommandLog(host="192.0.2.10", sitl=True)
        decision = log.connect()
        self.assertFalse(decision.allowed)
        assert decision.reason is not None
        self.assertIn("Refusing connection", decision.reason)
        self.assertIn("docker-local", decision.reason)
        self.assertEqual(log.apply_estop(LandedState.ON_GROUND, 0.0), ())
        self.assertEqual(log.received_commands(), ())
        self.assertEqual(
            [command.name for command in estop_commands(LandedState.ON_GROUND, 0.0)],
            ["disarm"],
        )

    def test_loopback_allow_list_and_sitl_docker_names(self) -> None:
        plain = simulation_refusal_reason("192.0.2.10", sitl=False)
        self.assertEqual(
            plain,
            "Refusing connection to '192.0.2.10': this transport is simulation-only and "
            "only accepts loopback hosts (127.0.0.0/8, ::1, localhost). "
            "Pass sitl=True to also allow docker-local hosts. Other hosts are refused.",
        )
        flagged = simulation_refusal_reason("198.51.100.20", sitl=True)
        self.assertEqual(
            flagged,
            "Refusing connection to '198.51.100.20': SITL mode is simulation-only and "
            "still requires a loopback host (127.0.0.0/8, ::1, localhost) or a "
            "docker-local host such as host.docker.internal. Other hosts are refused.",
        )
        for host, sitl, allowed in _HOSTS:
            with self.subTest(host=host, sitl=sitl):
                reason = simulation_refusal_reason(host, sitl=sitl)
                self.assertEqual(simulation_target_allowed(host, sitl=sitl), allowed)
                self.assertEqual(reason is None, allowed)
                log = MockCommandLog(host=host, sitl=sitl)
                decision = log.connect()
                self.assertEqual(decision.allowed, allowed)
                if allowed:
                    self.assertIsNone(decision.reason)
                    continue
                assert reason is not None
                self.assertIn("Refusing connection", reason)
                if sitl:
                    self.assertIn("docker-local", reason)
                else:
                    self.assertIn("loopback", reason)
                    self.assertIn("sitl=True", reason)
                self.assertEqual(log.apply_estop(LandedState.ON_GROUND, 0.0), ())
                self.assertEqual(log.received_commands(), ())

    def test_connect_does_not_touch_the_network(self) -> None:
        def check() -> None:
            for host, sitl, _allowed in _HOSTS:
                MockCommandLog(host=host, sitl=sitl).connect()

        self._call_under_broken_socket(check)

    def _assert_command_safe(self, command: EstopCommand) -> None:
        self.assertIn(command.name, _NAMES)
        self.assertEqual(command.mav_command, _NAMES[command.name])
        self.assertNotEqual(command.mav_command, 185)
        self.assertNotIn(
            command.name,
            {"flight_termination", "terminate", "kill", "force_disarm"},
        )
        self.assertFalse(command_is_forbidden(command))
        for value in (
            command.param1,
            command.param2,
            command.param3,
            float(command.mav_command),
        ):
            self.assertFalse(math.isclose(value, 21196.0, abs_tol=1e-6))
        if command.name == "disarm":
            self.assertEqual(command.param1, 0.0)
            self.assertEqual(command.param2, 0.0)

    def _call_under_broken_socket(self, fn: Callable[..., object]) -> None:
        def boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("network call")

        original_socket = socket.socket
        original_getaddrinfo = socket.getaddrinfo
        try:
            setattr(socket, "socket", boom)
            setattr(socket, "getaddrinfo", boom)
            fn()
        finally:
            setattr(socket, "socket", original_socket)
            setattr(socket, "getaddrinfo", original_getaddrinfo)


if __name__ == "__main__":
    unittest.main()
