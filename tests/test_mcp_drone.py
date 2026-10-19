"""OHH-293 drone MCP tools. Sim-only unless --allow-hardware. Never arms."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from ohho.aerial import Frame
from ohho.mcp_drone import DRONE_TOOL_NAMES, build_drone_tools
from ohho.mcp_server import handle_request
from ohho.mcp_tools import (
    McpTool,
    RobotSession,
    _finalize_tools,
    assert_mcp_tool_name_allowed,
)
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime
from ohho.safety import DRONE_PROFILE, AuditLog, SafetyGate, drone_mcp_allowed
from ohho.schema import ConnectionState, Telemetry, TransportStatus, Velocity
from ohho.transport import BaseTransport

_DENIED = (
    "drone_arm",
    "drone_takeoff",
    "drone_rtl",
    "release_stop",
    "drone_disarm",
    "takeoff",
    "rtl",
    "arm_flight",
    "set_fence",
    "fence_change",
    "changeFence",
    "setGeofence",
)


def _call(session: RobotSession, name: str, arguments: dict, req_id: int = 1) -> dict:
    res = handle_request(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        session=session,
    )
    if res.get("error"):
        raise AssertionError(res["error"])
    body = res["result"]
    if body.get("isError"):
        raise AssertionError(body["content"][0]["text"])
    text = body["content"][0]["text"]
    return json.loads(text)


def _error_body(session: RobotSession, name: str, arguments: dict) -> dict:
    res = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        session=session,
    )
    if res.get("error"):
        raise AssertionError(res["error"])
    body = res["result"]
    if not body.get("isError"):
        raise AssertionError(body)
    text = body["content"][0]["text"]
    return json.loads(text)


def _assert_backend(payload: dict) -> None:
    assert payload["backend"] in ("sim", "hardware")


def _airborne(bot: Robot, alt: float) -> dict:
    """Put the kinematic sim in the air. The MCP tools do not do this."""
    bot.arm_flight()
    bot.takeoff(alt)
    last = None
    for _ in range(400):
        last = bot.transport.step(0.05)
        drone = last.custom["drone"]
        if drone["flight_mode"] == "hold" and drone["landed_state"] == "in_air":
            return drone
    raise AssertionError(last.custom["drone"] if last is not None else "no step")


def _altitude(bot: Robot) -> float:
    pose = bot.telemetry().custom["drone"]["pose"]
    return -float(pose["z"])


@contextmanager
def _forbid_arming():
    """Explode if an MCP handler arms, takes off, returns, or releases."""
    with (
        patch.object(Robot, "arm", side_effect=AssertionError("MCP called arm")),
        patch.object(
            Robot, "arm_flight", side_effect=AssertionError("MCP called arm_flight")
        ),
        patch.object(
            Robot, "takeoff", side_effect=AssertionError("MCP called takeoff")
        ),
        patch.object(Robot, "rtl", side_effect=AssertionError("MCP called rtl")),
        patch.object(
            Robot,
            "release_stop",
            side_effect=AssertionError("MCP called release_stop"),
        ),
    ):
        yield


def _no_prompt(_text: str) -> str:
    raise AssertionError("MCP prompted on stdin")


class _FlightLink(BaseTransport):
    """In-memory aircraft link. Never opens a radio or a flight controller."""

    protocol = "px4-dds"
    backend_id = "px4-dds"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list = []
        self.state = ConnectionState.IDLE
        self.emit_on_connect = True
        self.drone: dict = {
            "armed": False,
            "flight_mode": "disarmed",
            "landed_state": "on_ground",
            "pose": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            "pose_frame": "LOCAL_NED",
            "velocity": {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0},
            "link_ok": True,
            "battery": 1.0,
            "experimental": True,
            "flight_ready": False,
            "gps_fix": "3d",
            "failsafe": [],
        }

    def connect(self) -> TransportStatus:
        self.state = ConnectionState.CONNECTED
        if self.emit_on_connect:
            self.push()
        return self.status()

    def disconnect(self) -> None:
        self.calls.append(("disconnect",))
        self.state = ConnectionState.DISCONNECTED

    def status(self) -> TransportStatus:
        return TransportStatus(protocol=self.protocol, state=self.state, label="fake")

    def read(self) -> Telemetry:
        return Telemetry(custom={"drone": self.drone}, battery=1.0, timestamp=0.0)

    def push(self) -> None:
        self._emit_telemetry(self.read())

    def send_velocity(self, vel: Velocity) -> None:
        self.calls.append(("vel", vel.linear_x, vel.linear_y, vel.angular_z))

    def send_joint_command(self, name: str, position: float) -> None:
        self.calls.append(("joint", name, position))

    def send_flight(self, action: str, **fields: object) -> None:
        self.calls.append(("flight", action, dict(fields)))

    def emergency_stop(self) -> None:
        self.calls.append(("estop",))

    def release_stop(self) -> None:
        self.calls.append(("release",))

    def flight_actions(self) -> list:
        return [call[1] for call in self.calls if call[0] == "flight"]


def _gated(
    tmp: str,
    inner: _FlightLink | None = None,
    profile=None,
):
    spec = get_spec("px4-sitl")
    link = inner or _FlightLink()
    gate = SafetyGate(
        link,
        spec,
        profile=profile if profile is not None else DRONE_PROFILE,
        audit=AuditLog(tmp),
        auto_tick=False,
        isatty=lambda: False,
        input_fn=_no_prompt,
        environ={"OHHO_ARM_HARDWARE": "1"},
    )
    gate.connect()
    runtime = NativeRuntime()
    runtime.start()
    return Robot(spec, gate, runtime), link, gate


class TestDroneMcpRegistration(unittest.TestCase):
    def test_tools_list_omits_arm_takeoff_rtl_and_release(self) -> None:
        session = RobotSession()
        listed = handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, session=session
        )
        names = {tool["name"] for tool in listed["result"]["tools"]}
        for name in DRONE_TOOL_NAMES:
            self.assertIn(name, names)
        self.assertIn("emergency_stop", names)
        for name in ("drone_arm", "drone_takeoff", "drone_rtl", "release_stop"):
            self.assertNotIn(name, names)
            res = handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": {}},
                },
                session=session,
            )
            self.assertEqual(res["error"]["code"], -32601, name)

    def test_registration_denylist(self) -> None:
        for name in _DENIED:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    assert_mcp_tool_name_allowed(name)
                tool = McpTool(
                    name,
                    "forbidden",
                    {"type": "object", "properties": {}},
                    lambda _params: {},
                    "pilot",
                )
                with self.assertRaises(ValueError):
                    _finalize_tools([tool])
        for name in (*DRONE_TOOL_NAMES, "emergency_stop", "drive", "look"):
            assert_mcp_tool_name_allowed(name)

    def test_bound_action_is_checked_with_drone_mcp_allowed(self) -> None:
        real = drone_mcp_allowed

        def fake(action: object) -> bool:
            if action == "land":
                return False
            return real(action)

        with patch("ohho.mcp_drone.drone_mcp_allowed", side_effect=fake):
            with self.assertRaises(ValueError) as raised:
                build_drone_tools(RobotSession())
        self.assertIn("land", str(raised.exception))


class TestDroneMcpSim(unittest.TestCase):
    def test_connect_telemetry_and_clamp_without_arming(self) -> None:
        session = RobotSession()
        env = {"OHHO_ARM_HARDWARE": "1", "OHHO_ALLOW_HARDWARE": "1"}
        try:
            with patch.dict(os.environ, env):
                with _forbid_arming():
                    out = _call(session, "drone_connect", {})
                    _assert_backend(out)
                    self.assertEqual(out["backend"], "sim")
                    self.assertEqual(out["robot"], "px4-sitl")
                    self.assertEqual(out["transport"], "sim://")
                    self.assertFalse(out["gate_armed"])
                    self.assertFalse(out["autopilot_armed"])
                    self.assertFalse(out["drone"]["flight_ready"])
                    self.assertFalse(session.allow_hardware)
                    telem = _call(session, "drone_telemetry", {})
                    _assert_backend(telem)
                    self.assertEqual(telem["backend"], "sim")
                    self.assertIn("drone", telem)
                    self.assertFalse(telem["drone"]["armed"])
                    vel = _call(
                        session,
                        "drone_velocity",
                        {"vx": 3, "vy": 4, "yaw_rate": 5, "frame": "LOCAL_NED"},
                    )
                    _assert_backend(vel)
                    self.assertEqual(vel["backend"], "sim")
                    self.assertTrue(vel["clamped"])
                    self.assertEqual(vel["requested"]["vx"], 3)
                    self.assertAlmostEqual(
                        math.hypot(vel["applied"]["vx"], vel["applied"]["vy"]), 1.5
                    )
                    self.assertAlmostEqual(vel["applied"]["yaw_rate"], 0.8)
                    self.assertAlmostEqual(vel["cap"]["horizontal"], 1.5)
                    self.assertAlmostEqual(vel["cap"]["vertical"], 1.0)
                    self.assertAlmostEqual(vel["cap"]["altitude_m"], 3.0)
                    self.assertTrue(vel["sent"])
                    missing = _error_body(session, "drone_velocity", {"vx": 1})
                    _assert_backend(missing)
                    self.assertEqual(missing["error"], "bad_arguments")
                    self.assertEqual(missing["backend"], "sim")
        finally:
            session.disconnect()

    def test_three_metres_per_second_is_clamped_before_the_sim(self) -> None:
        session = RobotSession()
        try:
            _call(
                session, "drone_connect", {"robot": "px4-sitl", "transport": "sim://"}
            )
            bot = session.robot
            assert bot is not None
            _airborne(bot, 1.0)
            with _forbid_arming():
                out = _call(
                    session,
                    "drone_velocity",
                    {"vx": 3, "yaw_rate": 5, "frame": "LOCAL_NED"},
                )
            _assert_backend(out)
            self.assertTrue(out["clamped"])
            self.assertAlmostEqual(out["applied"]["vx"], 1.5)
            self.assertAlmostEqual(out["applied"]["yaw_rate"], 0.8)
            self.assertEqual(out["drone"]["flight_mode"], "offboard")
            bot.transport.step(2.0)
            velocity = bot.telemetry().custom["drone"]["velocity"]
            speed = math.hypot(velocity["vx"], velocity["vy"])
            self.assertAlmostEqual(speed, 1.5, delta=0.05)
            self.assertLess(speed, 1.7)
        finally:
            session.disconnect()

    def test_fence_refuses_a_climb_that_would_exit_and_holds(self) -> None:
        session = RobotSession()
        try:
            _call(session, "drone_connect", {})
            bot = session.robot
            assert bot is not None
            _airborne(bot, 2.9)
            body = _error_body(
                session,
                "drone_velocity",
                {"vz": -5, "frame": "LOCAL_NED"},
            )
            _assert_backend(body)
            self.assertEqual(body["error"], "geofence")
            self.assertEqual(body["reason"], "altitude")
            self.assertTrue(body["projected"])
            self.assertFalse(body["sent"])
            self.assertTrue(body["hold"])
            self.assertTrue(body["hold_sent"])
            self.assertEqual(body["backend"], "sim")
            self.assertAlmostEqual(body["applied"]["vz"], -1.0)
            self.assertEqual(body["drone"]["flight_mode"], "hold")
            held = _altitude(bot)
            for _ in range(10):
                bot.transport.step(0.05)
            self.assertAlmostEqual(_altitude(bot), held, delta=0.2)
            self.assertLess(_altitude(bot), 3.05)
            enu = _error_body(
                session,
                "drone_velocity",
                {"vz": 1, "frame": "LOCAL_ENU"},
            )
            self.assertEqual(enu["error"], "geofence")
            self.assertEqual(enu["backend"], "sim")
            self.assertFalse(enu["sent"])
        finally:
            session.disconnect()

    def test_pose_outside_the_fence_holds_instead_of_clipping(self) -> None:
        session = RobotSession()
        try:
            _call(session, "drone_connect", {})
            bot = session.robot
            assert bot is not None
            _airborne(bot, 1.0)
            bot.offboard_position(0.0, 0.0, -3.6, 0.0, Frame.LOCAL_NED)
            for _ in range(200):
                bot.transport.step(0.05)
                if _altitude(bot) > 3.2:
                    break
            self.assertGreater(_altitude(bot), 3.0)
            body = _error_body(
                session,
                "drone_velocity",
                {"vx": 0.2, "frame": "LOCAL_NED"},
            )
            self.assertEqual(body["error"], "geofence")
            self.assertEqual(body["reason"], "altitude")
            self.assertFalse(body["projected"])
            self.assertFalse(body["sent"])
            self.assertEqual(body["backend"], "sim")
            self.assertEqual(body["drone"]["flight_mode"], "hold")
        finally:
            session.disconnect()

    def test_horizontal_velocity_that_would_exit_holds(self) -> None:
        session = RobotSession()
        try:
            _call(session, "drone_connect", {})
            bot = session.robot
            assert bot is not None
            _airborne(bot, 1.0)
            bot.offboard_position(9.8, 0.0, -1.0, 0.0, Frame.LOCAL_NED)
            for _ in range(400):
                bot.transport.step(0.05)
                pose = bot.telemetry().custom["drone"]["pose"]
                if pose["x"] > 9.75 and abs(-pose["z"] - 1.0) < 0.25:
                    break
            self.assertGreater(bot.telemetry().custom["drone"]["pose"]["x"], 9.75)
            x_before = bot.telemetry().custom["drone"]["pose"]["x"]
            body = _error_body(
                session,
                "drone_velocity",
                {"vx": 3, "frame": "LOCAL_NED"},
            )
            self.assertEqual(body["error"], "geofence")
            self.assertEqual(body["reason"], "radius")
            self.assertTrue(body["projected"])
            self.assertFalse(body["sent"])
            self.assertEqual(body["backend"], "sim")
            self.assertAlmostEqual(body["applied"]["vx"], 1.5)
            self.assertEqual(body["drone"]["flight_mode"], "hold")
            bot.transport.step(0.5)
            self.assertAlmostEqual(
                bot.telemetry().custom["drone"]["pose"]["x"], x_before, delta=0.05
            )
        finally:
            session.disconnect()

    def test_hold_land_and_emergency_stop(self) -> None:
        session = RobotSession()
        try:
            _call(session, "drone_connect", {"robot": "crazyflie-sim"})
            bot = session.robot
            assert bot is not None
            self.assertEqual(bot.spec.id, "crazyflie-sim")
            _airborne(bot, 2.0)
            bot.offboard_velocity(0.2, 0.0, 0.0, 0.0, Frame.LOCAL_NED)
            self.assertEqual(
                bot.transport.read().custom["drone"]["flight_mode"], "offboard"
            )
            held = _call(session, "drone_hold", {})
            _assert_backend(held)
            self.assertTrue(held["sent"])
            self.assertEqual(held["backend"], "sim")
            self.assertEqual(held["drone"]["flight_mode"], "hold")
            with _forbid_arming():
                landed = _call(session, "drone_land", {})
            _assert_backend(landed)
            self.assertTrue(landed["sent"])
            self.assertEqual(landed["action"], "land")
            self.assertEqual(landed["backend"], "sim")
            self.assertFalse(landed["latched"])
            self.assertEqual(landed["drone"]["flight_mode"], "land")
            for _ in range(400):
                sample = bot.transport.step(0.05)
                if sample.custom["drone"]["landed_state"] == "on_ground":
                    break
            drone = bot.telemetry().custom["drone"]
            self.assertEqual(drone["landed_state"], "on_ground")
            self.assertFalse(drone["armed"])
        finally:
            session.disconnect()

    def test_emergency_stop_latches_and_blocks_velocity(self) -> None:
        session = RobotSession()
        try:
            _call(session, "drone_connect", {})
            bot = session.robot
            assert bot is not None
            _airborne(bot, 2.0)
            with _forbid_arming():
                stopped = _call(session, "emergency_stop", {})
            _assert_backend(stopped)
            self.assertTrue(stopped["estop"])
            self.assertTrue(stopped["latched"])
            self.assertEqual(stopped["backend"], "sim")
            self.assertTrue(bot.transport._estop_latched)
            drone = bot.transport.read().custom["drone"]
            self.assertEqual(drone["flight_mode"], "land")
            self.assertEqual(drone["landed_state"], "landing")
            self.assertTrue(drone["armed"])
            body = _error_body(
                session,
                "drone_velocity",
                {"vx": 1, "frame": "LOCAL_NED"},
            )
            self.assertEqual(body["reason"], "estop_latched")
            self.assertEqual(body["backend"], "sim")
            self.assertFalse(body["sent"])
            landed = _call(session, "drone_land", {})
            self.assertTrue(landed["sent"])
            self.assertTrue(landed["latched"])
            self.assertEqual(landed["backend"], "sim")
            self.assertTrue(bot.transport._estop_latched)
        finally:
            session.disconnect()

    def test_hardware_uri_without_the_flag_errors(self) -> None:
        session = RobotSession()
        with patch.dict(
            os.environ, {"OHHO_ALLOW_HARDWARE": "1", "OHHO_ARM_HARDWARE": "1"}
        ):
            with patch(
                "ohho.mcp_tools.Robot.connect", side_effect=AssertionError("opened")
            ):
                body = _error_body(
                    session,
                    "drone_connect",
                    {
                        "robot": "px4-sitl",
                        "transport": "px4+dds://192.0.2.1:8888",
                        "allow_hardware": True,
                        "allowHardware": True,
                    },
                )
        _assert_backend(body)
        self.assertEqual(body["error"], "hardware_refused")
        self.assertEqual(body["reason"], "sim_only")
        self.assertEqual(body["backend"], "hardware")
        self.assertIn("--allow-hardware", body["message"])
        self.assertFalse(session.allow_hardware)
        self.assertIsNone(session.robot)

    def test_non_aerial_robot_is_refused(self) -> None:
        session = RobotSession()
        try:
            body = _error_body(
                session,
                "drone_connect",
                {"robot": "omnibot", "transport": "sim://"},
            )
        finally:
            session.disconnect()
        self.assertEqual(body["error"], "not_aerial")
        self.assertEqual(body["backend"], "sim")
        self.assertIsNone(session.robot)

    def test_not_connected_payloads_have_a_backend(self) -> None:
        session = RobotSession()
        for name in ("drone_telemetry", "drone_velocity", "drone_hold", "drone_land"):
            args = {"frame": "LOCAL_NED"} if name == "drone_velocity" else {}
            body = _error_body(session, name, args)
            self.assertEqual(body["error"], "not_connected", name)
            self.assertEqual(body["backend"], "sim")


class TestDroneMcpHardware(unittest.TestCase):
    def test_allow_hardware_connect_does_not_arm(self) -> None:
        inner = _FlightLink()

        def resolve(uri, spec, runtime):
            self.assertEqual(uri, "px4+dds://192.0.2.10:8888")
            self.assertEqual(spec.id, "px4-sitl")
            return inner

        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "OHHO_ARM_HARDWARE": "1",
                "OHHO_SAFETY_DIR": tmp,
                "OHHO_SAFETY_CONFIG": "",
            }
            with patch.dict(os.environ, env):
                with patch("ohho.robot.resolve_transport", resolve):
                    with _forbid_arming():
                        session = RobotSession(allow_hardware=True)
                        try:
                            out = _call(
                                session,
                                "drone_connect",
                                {
                                    "robot": "px4-sitl",
                                    "transport": "px4+dds://192.0.2.10:8888",
                                },
                            )
                            _assert_backend(out)
                            self.assertEqual(out["backend"], "hardware")
                            self.assertFalse(out["gate_armed"])
                            self.assertFalse(out["autopilot_armed"])
                            gate = session.robot.transport
                            self.assertIsInstance(gate, SafetyGate)
                            self.assertFalse(gate.armed)
                            body = _error_body(
                                session,
                                "drone_velocity",
                                {"vx": 3, "frame": "BODY_FRD"},
                            )
                            self.assertEqual(body["error"], "motion_refused")
                            self.assertEqual(body["reason"], "not_armed")
                            self.assertEqual(body["backend"], "hardware")
                            self.assertFalse(body["sent"])
                            self.assertTrue(body["clamped"])
                            self.assertAlmostEqual(body["applied"]["vx"], 1.5)
                            self.assertAlmostEqual(body["cap"]["altitude_m"], 3.0)
                            self.assertNotIn(
                                "offboard_velocity", inner.flight_actions()
                            )
                            self.assertNotIn("arm", inner.flight_actions())
                            self.assertFalse(gate.armed)
                        finally:
                            session.disconnect()

    def test_hardware_geofence_ceiling_and_disarmed_land(self) -> None:
        tight = dataclasses.replace(
            DRONE_PROFILE, fence_max_alt_m=2.0, takeoff_alt_m=2.0
        )
        inner = _FlightLink()
        inner.drone["pose"]["z"] = -2.5
        inner.drone["landed_state"] = "in_air"
        with tempfile.TemporaryDirectory() as tmp:
            bot, link, gate = _gated(tmp, inner, tight)
            session = RobotSession(allow_hardware=True, robot_id="px4-sitl")
            session._robot = bot
            try:
                self.assertFalse(gate.arm())
                self.assertFalse(gate.armed)
                body = _error_body(
                    session,
                    "drone_velocity",
                    {"vx": 0.2, "frame": "LOCAL_NED"},
                )
                self.assertEqual(body["error"], "geofence")
                self.assertEqual(body["reason"], "altitude")
                self.assertEqual(body["backend"], "hardware")
                self.assertAlmostEqual(body["cap"]["altitude_m"], 2.0)
                self.assertFalse(body["sent"])
                self.assertTrue(body["hold"])
                self.assertNotIn("offboard_velocity", link.flight_actions())
                link.drone["pose"]["z"] = 0.0
                link.drone["landed_state"] = "on_ground"
                link.push()
                link.calls.clear()
                inside = _error_body(
                    session,
                    "drone_velocity",
                    {"vx": 3, "frame": "LOCAL_NED"},
                )
                self.assertEqual(inside["reason"], "not_armed")
                self.assertEqual(inside["backend"], "hardware")
                self.assertAlmostEqual(inside["applied"]["vx"], 1.5)
                self.assertAlmostEqual(inside["cap"]["altitude_m"], 2.0)
                self.assertNotIn("offboard_velocity", link.flight_actions())
                session.allow_hardware = False
                refused = _error_body(
                    session,
                    "drone_velocity",
                    {"vx": 0.1, "frame": "LOCAL_NED"},
                )
                self.assertEqual(refused["error"], "hardware_refused")
                self.assertEqual(refused["backend"], "hardware")
                landed = _error_body(session, "drone_land", {})
                self.assertEqual(landed["backend"], "hardware")
                self.assertNotEqual(landed["error"], "hardware_refused")
                self.assertEqual(landed["reason"], "not_armed")
                self.assertFalse(landed["sent"])
                self.assertFalse(gate.armed)
            finally:
                session.disconnect()

    def test_emergency_stop_maps_land_in_the_air_and_latches(self) -> None:
        inner = _FlightLink()
        inner.drone["landed_state"] = "in_air"
        inner.drone["armed"] = True
        inner.drone["pose"]["z"] = -2.0
        inner.drone["flight_mode"] = "offboard"
        with tempfile.TemporaryDirectory() as tmp:
            bot, link, gate = _gated(tmp, inner)
            session = RobotSession(allow_hardware=False, robot_id="px4-sitl")
            session._robot = bot
            try:
                stopped = _call(session, "emergency_stop", {})
                self.assertTrue(stopped["estop"])
                self.assertTrue(stopped["latched"])
                self.assertEqual(stopped["backend"], "hardware")
                self.assertTrue(gate.latched)
                self.assertFalse(gate.armed)
                self.assertEqual(link.flight_actions(), ["land"])
                self.assertNotIn(("release",), link.calls)
                self.assertNotIn("disarm", link.flight_actions())
            finally:
                session.disconnect()

    def test_unknown_landed_state_holds_then_lands(self) -> None:
        inner = _FlightLink()
        inner.emit_on_connect = False
        with tempfile.TemporaryDirectory() as tmp:
            bot, link, gate = _gated(tmp, inner)
            session = RobotSession(allow_hardware=True, robot_id="px4-sitl")
            session._robot = bot
            try:
                stopped = _call(session, "emergency_stop", {})
                self.assertTrue(stopped["latched"])
                self.assertEqual(stopped["backend"], "hardware")
                self.assertEqual(link.flight_actions(), ["hold", "land"])
                self.assertNotIn("disarm", link.flight_actions())
                self.assertTrue(gate.latched)
            finally:
                session.disconnect()


if __name__ == "__main__":
    unittest.main()
