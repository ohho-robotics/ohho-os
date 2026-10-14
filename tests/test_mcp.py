import unittest

from ohho.mcp_server import handle_request
from ohho.mcp_split import ON_ROBOT_MCP_PRODUCTS
from ohho.mcp_tools import RobotSession, build_tools


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


if __name__ == "__main__":
    unittest.main()
