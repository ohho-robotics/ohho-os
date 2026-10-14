import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout

from ohho.adapters.sim import SimTransport
from ohho.cli import main
from ohho.mcp_server import handle_request
from ohho.mcp_split import ON_ROBOT_MCP_PRODUCTS
from ohho.mcp_tools import SIM_TOOL_NAMES, RobotSession, build_tools
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime


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
        telem = _call(session, "telemetry", {})
        self.assertIn("odom", telem)
        session.disconnect()

    def test_drive_and_telemetry_on_sim(self):
        session = RobotSession()
        session._robot = _detached_sim()
        out = _call(session, "drive", {"vx": 0.2, "seconds": 0.5})
        self.assertTrue(out["drove"])
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


if __name__ == "__main__":
    unittest.main()
