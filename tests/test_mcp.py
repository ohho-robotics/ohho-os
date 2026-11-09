import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from ohho.adapters.sim import SimTransport
from ohho.cli import main
from ohho.mcp_connect import SERVER_ARGS
from ohho.mcp_server import handle_request
from ohho.mcp_split import ON_ROBOT_MCP_PRODUCTS
from ohho.mcp_tools import (
    SIM_TOOL_NAMES,
    McpTool,
    RobotSession,
    _finalize_tools,
    assert_mcp_tool_name_allowed,
    build_tools,
)
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime
from ohho.safety import AuditLog, SafetyGate
from ohho.schema import ConnectionState, Telemetry, TransportStatus, Velocity
from ohho.transport import BaseTransport


class TestMcpServer(unittest.TestCase):
    def test_tools_are_on_robot_products(self):
        session = RobotSession()
        products = {t.product for t in build_tools(session)}
        self.assertTrue(products.issubset(ON_ROBOT_MCP_PRODUCTS))
        self.assertGreaterEqual(len(products), 6)

    def test_initialize(self):
        res = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(res["id"], 1)
        self.assertIn("protocolVersion", res["result"])

    def test_list_and_call_connect(self):
        session = RobotSession(robot_id="omnibot", transport="sim://")
        listed = handle_request(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, session=session
        )
        names = {t["name"] for t in listed["result"]["tools"]}
        self.assertIn("connect.listRobots", names)
        self.assertIn("bridge.listAdapters", names)

        connect_res = handle_request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "connect.connect",
                    "arguments": {"robotId": "omnibot", "transport": "sim://"},
                },
            },
            session=session,
        )
        self.assertFalse(connect_res.get("error"))
        self.assertFalse(connect_res["result"]["isError"])

        telem = handle_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "connect.getTelemetry", "arguments": {}},
            },
            session=session,
        )
        self.assertFalse(telem["result"]["isError"])
        session.disconnect()

    def test_hosted_tool_rejected(self):
        res = handle_request(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "garage.listRobots", "arguments": {}},
            }
        )
        self.assertEqual(res["error"]["code"], -32601)


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
    return json.loads(text) if text[:1] in "{[" else {"text": text}


def _detached_sim() -> Robot:
    """Sim robot whose physics advance only when tests call ``step``."""
    spec = get_spec("omnibot")
    transport = SimTransport(spec)
    transport.connect()
    runtime = NativeRuntime()
    runtime.start()
    return Robot(spec, transport, runtime)


class TestRealSimTools(unittest.TestCase):
    def test_listed_with_on_robot_catalog(self):
        session = RobotSession()
        listed = handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, session=session
        )
        names = {t["name"] for t in listed["result"]["tools"]}
        self.assertTrue(set(SIM_TOOL_NAMES).issubset(names))
        products = {t.product for t in build_tools(session)}
        self.assertTrue(products.issubset(ON_ROBOT_MCP_PRODUCTS))

    def test_connect_rejects_hardware(self):
        session = RobotSession()
        for uri in ("serial:///dev/ttyUSB0", "dds://eth0"):
            res = handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "connect",
                        "arguments": {"robot": "omnibot", "transport": uri},
                    },
                },
                session=session,
            )
            self.assertFalse(res.get("error"))
            self.assertTrue(res["result"]["isError"])
            self.assertIn("sim://", res["result"]["content"][0]["text"])

    def test_connect_then_telemetry(self):
        session = RobotSession()
        payload = _call(session, "connect", {"robot": "omnibot", "transport": "sim://"})
        self.assertTrue(payload["connected"])
        self.assertEqual(payload["robot"], "omnibot")
        self.assertEqual(payload["backend"], "sim")
        telem = _call(session, "telemetry", {})
        self.assertIn("odom", telem)
        self.assertEqual(telem["backend"], "sim")
        session.disconnect()

    def test_drive_and_telemetry_on_sim(self):
        session = RobotSession()
        session._robot = _detached_sim()
        out = _call(session, "drive", {"vx": 0.2, "seconds": 0.5})
        self.assertTrue(out["drove"])
        self.assertEqual(out["backend"], "sim")
        self.assertGreater(out["telemetry"]["odom"]["x"], 0.05)
        telem = _call(session, "telemetry", {})
        self.assertIn("odom", telem)
        session.disconnect()

    def test_look_demo_world(self):
        session = RobotSession()
        session._robot = _detached_sim()
        with tempfile.TemporaryDirectory() as tmp:
            session.memory_path = f"{tmp}/omnibot.json"
            result = _call(session, "look", {"world": "demo"})
        self.assertGreaterEqual(len(result["detections"]), 1)
        self.assertGreaterEqual(result["remembered"], 1)
        session.disconnect()

    def test_nav_goto_reaches_near_goal(self):
        session = RobotSession()
        session._robot = _detached_sim()
        result = _call(session, "nav_goto", {"x": 0.5, "y": 0.0, "timeout": 20.0})
        self.assertTrue(result["reached"])
        session.disconnect()

    def test_drive_without_session_is_error(self):
        session = RobotSession()
        res = handle_request(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "drive", "arguments": {"vx": 0.1, "seconds": 0.1}},
            },
            session=session,
        )
        self.assertTrue(res["result"]["isError"])

    def test_mcp_help_and_doctor(self):
        with redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit) as ex:
                main(["--help"])
        self.assertEqual(ex.exception.code, 0)
        self.assertIn("mcp", out.getvalue())
        with redirect_stdout(io.StringIO()) as doctor:
            code = main(["doctor"])
        self.assertEqual(code, 0)
        text = doctor.getvalue()
        self.assertIn("nav_goto", text)
        self.assertIn("look", text)
        self.assertIn("emergency_stop", text)


class _RecordingTransport(BaseTransport):
    """In-memory non-sim link. Never opens a device."""

    def __init__(self, protocol: str = "serial") -> None:
        super().__init__()
        self.protocol = protocol
        self.state = ConnectionState.IDLE
        self.calls: list = []
        self.latest = Telemetry()

    def connect(self) -> TransportStatus:
        self.state = ConnectionState.CONNECTED
        return self.status()

    def disconnect(self) -> None:
        self.calls.append(("disconnect",))
        self.state = ConnectionState.DISCONNECTED

    def status(self) -> TransportStatus:
        return TransportStatus(protocol=self.protocol, state=self.state, label="fake")

    def read(self) -> Telemetry:
        return self.latest

    def send_velocity(self, vel: Velocity) -> None:
        self.calls.append(("vel", vel.linear_x, vel.linear_y, vel.angular_z))

    def send_joint_command(self, name: str, position: float) -> None:
        self.calls.append(("joint", name, position))

    def emergency_stop(self) -> None:
        self.calls.append(("estop",))

    def release_stop(self) -> None:
        self.calls.append(("release",))

    def velocities(self) -> list:
        return [call for call in self.calls if call[0] == "vel"]


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
    return json.loads(text) if text[:1] in "{[" else {"text": text}


class TestMcpSafety(unittest.TestCase):
    """OHH-278: sim-only unless --allow-hardware; MCP never arms or releases."""

    def test_listed_names_pass_the_denylist(self) -> None:
        session = RobotSession()
        listed = handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, session=session
        )
        names = [tool["name"] for tool in listed["result"]["tools"]]
        self.assertIn("emergency_stop", names)
        self.assertIn("connect.emergencyStop", names)
        for name in names:
            assert_mcp_tool_name_allowed(name)
        banned = ("arm", "disarm", "release_stop", "release-stop", "profile", "limit")
        for name in names:
            tokens = name.lower().replace(".", " ").replace("_", " ").replace("-", " ")
            for word in banned:
                self.assertNotEqual(name.lower(), word)
                self.assertNotIn(f" {word} ", f" {tokens} ")

    def test_registration_rejects_future_forbidden_names(self) -> None:
        forbidden = (
            "arm",
            "disarm",
            "disarm-override",
            "disarm_override",
            "release_stop",
            "release-stop",
            "releaseStop",
            "set_limit",
            "setLimits",
            "safety_profile",
            "setSafetyProfile",
            "connect.arm",
            "connect.release_stop",
        )
        for name in forbidden:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    assert_mcp_tool_name_allowed(name)
                tool = McpTool(
                    name, "forbidden", {"type": "object"}, lambda p: {}, "connect"
                )
                with self.assertRaises(ValueError):
                    _finalize_tools([tool])
        for name in ("emergency_stop", "connect.emergencyStop", "drive", "look"):
            assert_mcp_tool_name_allowed(name)

    def test_unknown_arm_and_release_tools(self) -> None:
        session = RobotSession()
        for name in (
            "arm",
            "release_stop",
            "release-stop",
            "set_limit",
            "safety_profile",
        ):
            res = handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": {}},
                },
                session=session,
            )
            self.assertEqual(res["error"]["code"], -32601, name)

    def test_hardware_transport_refused_by_default(self) -> None:
        session = RobotSession()
        with patch(
            "ohho.mcp_tools.Robot.connect", side_effect=AssertionError("opened")
        ):
            for uri in (
                "serial:///dev/ttyUSB0",
                "dds://eth0",
                "feetech:///dev/ttyACM0",
            ):
                body = _error_body(
                    session,
                    "connect",
                    {
                        "robot": "omnibot",
                        "transport": uri,
                        "allowHardware": True,
                        "allow_hardware": True,
                    },
                )
                self.assertEqual(body["error"], "hardware_refused")
                self.assertIn("sim://", body["message"])
                self.assertIn("--allow-hardware", body["message"])
        self.assertFalse(session.allow_hardware)
        self.assertIsNone(session.robot)
        catalog = _error_body(
            session,
            "connect.connect",
            {"robotId": "omnibot", "transport": "serial:///dev/ttyUSB0"},
        )
        self.assertEqual(catalog["error"], "hardware_refused")

    def test_env_and_tool_argument_cannot_enable_hardware(self) -> None:
        with patch.dict(
            os.environ, {"OHHO_ALLOW_HARDWARE": "1", "OHHO_ARM_HARDWARE": "1"}
        ):
            session = RobotSession()
            self.assertFalse(session.allow_hardware)
            with patch(
                "ohho.mcp_tools.Robot.connect", side_effect=AssertionError("opened")
            ):
                body = _error_body(
                    session,
                    "connect",
                    {"transport": "dds://eth0", "allowHardware": True},
                )
            self.assertEqual(body["reason"], "sim_only")
            self.assertIsNone(session.robot)

    def test_serve_rejects_hardware_without_the_flag(self) -> None:
        with redirect_stderr(io.StringIO()):
            with patch("ohho.mcp_server.run_stdio", return_value=0) as run:
                code = main(["mcp", "serve", "--transport", "serial:///dev/ttyUSB0"])
        self.assertEqual(code, 2)
        run.assert_not_called()

    def test_serve_env_cannot_replace_the_flag(self) -> None:
        with patch.dict(
            os.environ, {"OHHO_ALLOW_HARDWARE": "1", "OHHO_ARM_HARDWARE": "1"}
        ):
            with redirect_stderr(io.StringIO()):
                with patch("ohho.mcp_server.run_stdio", return_value=0) as run:
                    code = main(["mcp", "serve", "--transport", "dds://eth0"])
        self.assertEqual(code, 2)
        run.assert_not_called()

    def test_allow_hardware_flag_does_not_connect_or_arm(self) -> None:
        with patch("ohho.mcp_server.run_stdio", return_value=0) as run:
            code = main(
                ["mcp", "serve", "--allow-hardware", "--transport", "serial://fake"]
            )
        self.assertEqual(code, 0)
        session = run.call_args.kwargs["session"]
        self.assertTrue(session.allow_hardware)
        self.assertEqual(session.transport, "serial://fake")
        self.assertIsNone(session.robot)

    def test_serve_does_not_accept_arm_flag(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["mcp", "serve", "--arm"])
        self.assertNotEqual(raised.exception.code, 0)

    def test_serve_help_mentions_allow_hardware(self) -> None:
        with redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit) as raised:
                main(["mcp", "serve", "--help"])
        self.assertEqual(raised.exception.code, 0)
        text = out.getvalue()
        self.assertIn("--allow-hardware", text)
        self.assertNotIn("--arm", text)

    def test_client_configs_do_not_pass_the_flag(self) -> None:
        self.assertEqual(SERVER_ARGS, ("mcp", "serve"))
        self.assertNotIn("--allow-hardware", SERVER_ARGS)
        self.assertNotIn("--arm", SERVER_ARGS)

    def test_allow_hardware_uses_safety_gate_and_does_not_arm(self) -> None:
        inner = _RecordingTransport()

        def resolve(uri, spec, runtime, read_only=False):
            self.assertEqual(uri, "serial://fake")
            self.assertFalse(read_only)
            return inner

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {"OHHO_SAFETY_DIR": tmp, "OHHO_ARM_HARDWARE": "1"},
            ):
                with patch("ohho.robot.resolve_transport", resolve):
                    with patch.object(
                        Robot, "arm", side_effect=AssertionError("MCP called arm")
                    ):
                        session = RobotSession(
                            allow_hardware=True,
                            robot_id="omnibot",
                            transport="serial://fake",
                            runtime="native",
                        )
                        bot = session.connect()
                        try:
                            self.assertIsInstance(bot.transport, SafetyGate)
                            self.assertFalse(bot.transport.armed)
                            self.assertEqual(_backend_of(bot), "hardware")
                        finally:
                            session.disconnect()

    def test_two_meters_per_second_is_clamped_on_sim(self) -> None:
        session = RobotSession()
        session._robot = _detached_sim()
        try:
            out = _call(session, "drive", {"vx": 2.0, "seconds": 0.5})
        finally:
            session.disconnect()
        self.assertEqual(out["backend"], "sim")
        self.assertTrue(out["clamped"])
        self.assertEqual(out["requested"]["vx"], 2.0)
        self.assertLessEqual(out["applied"]["vx"], out["cap"]["vx"])
        self.assertLess(out["applied"]["vx"], 2.0)
        self.assertAlmostEqual(out["applied"]["vx"], 0.2)
        moved = out["telemetry"]["odom"]["x"]
        self.assertGreater(moved, 0.05)
        self.assertLess(moved, 0.25)

    def test_disarmed_hardware_refuses_motion_and_clamps_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot, inner = _gated(tmp, "unitree-go2")
            session = RobotSession(allow_hardware=True, robot_id="unitree-go2")
            session._robot = bot
            try:
                with patch.dict(os.environ, {"OHHO_ARM_HARDWARE": "1"}):
                    with patch.object(
                        Robot, "arm", side_effect=AssertionError("MCP called arm")
                    ):
                        body = _error_body(
                            session, "drive", {"vx": 2.0, "seconds": 0.2}
                        )
                self.assertEqual(body["error"], "motion_refused")
                self.assertEqual(body["reason"], "not_armed")
                self.assertEqual(body["backend"], "hardware")
                self.assertFalse(body["sent"])
                self.assertTrue(body["clamped"])
                self.assertLessEqual(body["applied"]["vx"], 0.5)
                self.assertLess(body["applied"]["vx"], 1.5)
                self.assertIn("ask the human", body["message"].lower())
                self.assertFalse(bot.transport.armed)
                for call in inner.velocities():
                    self.assertEqual(call[1:], (0.0, 0.0, 0.0))
            finally:
                session.disconnect()

    def test_emergency_stop_latches_sim_and_blocks_drive(self) -> None:
        session = RobotSession()
        session._robot = _detached_sim()
        try:
            _call(session, "drive", {"vx": 0.2, "seconds": 0.2})
            stopped = _call(session, "emergency_stop", {})
            self.assertTrue(stopped["estop"])
            self.assertTrue(stopped["latched"])
            self.assertEqual(stopped["backend"], "sim")
            x = session.robot.telemetry().odom.x
            body = _error_body(session, "drive", {"vx": 0.2, "seconds": 0.2})
            self.assertEqual(body["reason"], "estop_latched")
            self.assertAlmostEqual(session.robot.telemetry().odom.x, x)
        finally:
            session.disconnect()

    def test_emergency_stop_latches_gated_link(self) -> None:
        prompts: list = []

        def _input(text: str) -> str:
            prompts.append(text)
            raise AssertionError("MCP prompted on stdin")

        with tempfile.TemporaryDirectory() as tmp:
            bot, inner = _gated(tmp, "omnibot", input_fn=_input)
            session = RobotSession(allow_hardware=True)
            session._robot = bot
            try:
                with patch.dict(os.environ, {"OHHO_ARM_HARDWARE": "1"}):
                    # Env is set and stdin is not a TTY: arm() refuses without prompting.
                    self.assertFalse(bot.transport.arm())
                    self.assertEqual(prompts, [])
                    self.assertFalse(bot.transport.armed)
                    stopped = _call(session, "emergency_stop", {})
                    self.assertTrue(stopped["latched"])
                    self.assertEqual(stopped["backend"], "hardware")
                    self.assertTrue(bot.transport.latched)
                    self.assertFalse(bot.transport.armed)
                    self.assertIn(("estop",), inner.calls)
                    self.assertNotIn(("release",), inner.calls)
                    catalog = _call(session, "connect.emergencyStop", {})
                    self.assertTrue(catalog["latched"])
                    body = _error_body(session, "connect.sendVelocity", {"vx": 2.0})
                self.assertEqual(body["reason"], "estop_latched")
                self.assertEqual(body["backend"], "hardware")
                self.assertTrue(body["clamped"])
                self.assertLessEqual(body["applied"]["vx"], 0.2)
            finally:
                session.disconnect()


def _backend_of(bot: Robot) -> str:
    return "sim" if bot.simulated else "hardware"


def _gated(tmp: str, robot_id: str, input_fn=None):
    spec = get_spec(robot_id)
    inner = _RecordingTransport("serial")
    inner.connect()

    def _no_prompt(text: str) -> str:
        raise AssertionError(f"prompted: {text}")

    gate = SafetyGate(
        inner,
        spec,
        audit=AuditLog(tmp),
        auto_tick=False,
        isatty=lambda: False,
        input_fn=input_fn or _no_prompt,
        environ={"OHHO_ARM_HARDWARE": "1"},
    )
    gate.connect()
    runtime = NativeRuntime()
    runtime.start()
    return Robot(spec, gate, runtime), inner


if __name__ == "__main__":
    unittest.main()
