"""MCP humanoid tools (OHH-335).

Capability selects the tools. A config cannot add arm, damp, joint,
dance, or trajectory tools. Every tool payload includes ``backend``.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ohho import capabilities as caps
from ohho.adapters.sim import SimTransport
from ohho.mcp_humanoid import (
    HUMANOID_TOOL_NAMES,
    HumanoidMcpSession,
    build_humanoid_tools,
    exposed_tool_names,
    handle_request,
    is_hard_denied_tool,
    payload_of,
)
from ohho.registry import RobotSpec
from ohho.robot import Robot
from ohho.robot_spec import RobotClass
from ohho.runtime.native import NativeRuntime

_DENIED = (
    "arm",
    "Arm",
    "damp",
    "DAMP",
    "move_joints",
    "MoveJoints",
    "humanoid_damp",
    "humanoid_arm",
    "humanoid_move_joints",
    "dance",
    "dances",
    "lion_dance",
    "trajectory",
    "Trajectory",
    "record_trajectory",
    "replay_trajectory",
    "load_custom_trained_traj",
    "handshake",
    "zero_torque_drag",
    "kick",
    "shoot",
)


def _spec(*capabilities: str, **kwargs: object) -> RobotSpec:
    body: dict[str, object] = {
        "id": "field-robot",
        "name": "Field robot",
        "category": "humanoid",
        "capabilities": capabilities,
        "max_lin": 0.5,
        "max_ang": 1.0,
        "robot_class": RobotClass.HUMANOID,
    }
    body.update(kwargs)
    return RobotSpec(**body)  # type: ignore[arg-type]


def _plugin_spec() -> RobotSpec:
    root = Path(__file__).resolve().parents[1] / "plugins" / "ohho-booster"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from ohho_booster.spec import booster_t2_spec

    return booster_t2_spec()


def _config() -> dict[str, object]:
    return {
        "tools": list(_DENIED),
        "allowlist": ["move", "damp", "move_joints", "dance"],
    }


class TestSelection(unittest.TestCase):
    def test_same_id_follows_capabilities_not_the_id(self) -> None:
        locomotion = _spec(caps.HUMANOID_LOCOMOTION, caps.BASE_DRIVE)
        posture = _spec(caps.HUMANOID_POSTURE)
        wheeled = _spec(caps.BASE_DRIVE, id="field-robot", category="mobile_base")
        self.assertEqual(locomotion.id, posture.id)
        self.assertEqual(exposed_tool_names(locomotion, _config()), HUMANOID_TOOL_NAMES)
        self.assertEqual(
            exposed_tool_names(posture, _config()),
            ("emergency_stop", "humanoid_telemetry"),
        )
        self.assertEqual(exposed_tool_names(wheeled, _config()), ())

    def test_config_cannot_add_denied_tools(self) -> None:
        spec = _spec(caps.HUMANOID_LOCOMOTION, caps.HUMANOID_POSTURE)
        names = exposed_tool_names(spec, _config())
        self.assertEqual(names, HUMANOID_TOOL_NAMES)
        for name in _DENIED:
            self.assertNotIn(name, names)
            self.assertTrue(is_hard_denied_tool(name))
        for name in HUMANOID_TOOL_NAMES:
            self.assertFalse(is_hard_denied_tool(name))


class TestBoosterSim(unittest.TestCase):
    def setUp(self) -> None:
        self.home = tempfile.TemporaryDirectory()
        self.spec = _plugin_spec()
        self.transport = SimTransport(self.spec)
        self.transport.connect()
        self.bot = Robot(self.spec, self.transport, NativeRuntime())
        self.session = HumanoidMcpSession(
            transport="sim://",
            config=_config(),
            robot=self.bot,
            home=self.home.name,
            environ={},
        )
        self.forbidden: list[str] = []
        self.bot.arm = self._forbid("arm")  # type: ignore[method-assign]
        self.bot.damp = self._forbid("damp")  # type: ignore[method-assign]
        self.bot.move_joints = self._forbid("move_joints")  # type: ignore[method-assign]
        self.bot.release_stop = self._forbid("release_stop")  # type: ignore[method-assign]

    def tearDown(self) -> None:
        self.bot.disconnect()
        self.home.cleanup()

    def _forbid(self, name: str):
        def _called(*_args: object, **_kwargs: object) -> None:
            self.forbidden.append(name)

        return _called

    def _call(self, name: str, arguments: dict[str, object] | None = None) -> dict:
        response = handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            },
            self.session,
        )
        self.assertIn("result", response)
        body = payload_of(response["result"])
        self.assertIn("backend", body)
        self.assertEqual(body["backend"], "sim")
        return body

    def test_four_tools_exist_and_denied_names_are_absent(self) -> None:
        self.assertEqual(self.spec.id, "booster-t2")
        self.assertIn(caps.HUMANOID_LOCOMOTION, self.spec.capabilities)
        self.assertIn(caps.HUMANOID_POSTURE, self.spec.capabilities)
        listed = handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, self.session
        )
        names = tuple(tool["name"] for tool in listed["result"]["tools"])
        self.assertEqual(names, HUMANOID_TOOL_NAMES)
        for name in _DENIED:
            self.assertNotIn(name, names)
        built = tuple(tool.name for tool in build_humanoid_tools(self.session))
        self.assertEqual(built, HUMANOID_TOOL_NAMES)

    def test_payloads_include_backend_and_walk_is_clamped(self) -> None:
        telemetry = self._call("humanoid_telemetry")
        self.assertIn("humanoid", telemetry)
        walk = self._call("humanoid_walk", {"vx": 2.0, "vy": 0.4, "wz": 2.0})
        self.assertTrue(walk["ok"])
        self.assertTrue(walk["sent"])
        self.assertTrue(walk["clamped"])
        self.assertAlmostEqual(walk["applied"]["vx"], 0.5)
        self.assertAlmostEqual(walk["applied"]["vy"], 0.3)
        self.assertAlmostEqual(walk["applied"]["wz"], 0.6)
        self.assertIn("velocity_cap", walk["reasons"])
        self.assertAlmostEqual(self.transport._cmd.linear_x, 0.5)
        self.assertAlmostEqual(self.transport._cmd.linear_y, 0.3)
        self.assertAlmostEqual(self.transport._cmd.angular_z, 0.6)

        stopped = self._call("humanoid_stop")
        self.assertEqual(stopped["action"], "stop_move")
        self.assertEqual(self.transport._cmd.linear_x, 0.0)
        self.assertEqual(self.transport._cmd.linear_y, 0.0)
        self.assertEqual(self.transport._cmd.angular_z, 0.0)

        estop = self._call("emergency_stop")
        self.assertTrue(estop["latched"])
        refused = self._call("humanoid_walk", {"vx": 0.2})
        self.assertEqual(refused["reason"], "estop_latched")
        self.assertFalse(refused["sent"])
        self.assertEqual(self.forbidden, [])

    def test_bad_arguments_payload_has_backend(self) -> None:
        body = self._call("humanoid_walk", {"vx": "fast"})
        self.assertEqual(body["error"], "bad_arguments")
        self.assertEqual(self.forbidden, [])


class TestSimOnly(unittest.TestCase):
    def test_hardware_uri_is_refused_and_payload_has_backend(self) -> None:
        spec = _spec(caps.HUMANOID_LOCOMOTION, caps.BASE_DRIVE)
        with tempfile.TemporaryDirectory() as home:
            session = HumanoidMcpSession(
                spec.id,
                transport="serial://loopback",
                allow_hardware=False,
                home=home,
                environ={},
            )
            with (
                patch("ohho.mcp_humanoid.get_spec", return_value=spec),
                patch(
                    "ohho.mcp_humanoid.Robot.connect",
                    side_effect=AssertionError("opened"),
                ),
            ):
                response = handle_request(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "humanoid_walk",
                            "arguments": {"vx": 0.1, "allow_hardware": True},
                        },
                    },
                    session,
                )
        body = payload_of(response["result"])
        self.assertEqual(body["backend"], "hardware")
        self.assertEqual(body["error"], "hardware_refused")
        self.assertTrue(response["result"]["isError"])
