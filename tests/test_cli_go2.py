"""CLI tests for unitree-go2 / DDS transport (OHH-120)."""

from __future__ import annotations

import io
import sys
import types
import unittest
from unittest import mock

from ohho import cli
from ohho.adapters.unitree import (
    UnitreeDdsTransport,
    dds_uri_is_hardware,
    dds_uri_is_sim_only,
)
from ohho.registry import get_spec
from ohho.safety import SafetyGate


class TestDdsUriClassification(unittest.TestCase):
    def test_sim_only_lo_domain_1(self):
        self.assertTrue(dds_uri_is_sim_only("dds://lo?domain=1"))
        self.assertTrue(dds_uri_is_sim_only("dds://"))
        self.assertFalse(dds_uri_is_hardware("dds://lo?domain=1"))

    def test_hardware_domain_0(self):
        self.assertTrue(dds_uri_is_hardware("dds://lo"))
        self.assertTrue(dds_uri_is_hardware("dds://eth0"))
        self.assertFalse(dds_uri_is_sim_only("dds://eth0"))

    def test_non_dds_uris(self):
        self.assertFalse(dds_uri_is_hardware("sim://"))
        self.assertFalse(dds_uri_is_hardware(None))


class TestCliGo2Parsing(unittest.TestCase):
    def test_drive_parses_unitree_dds_transport(self):
        args = cli.build_parser().parse_args(
            [
                "drive",
                "unitree-go2",
                "--transport",
                "dds://eth0?domain=0",
                "--vx",
                "0.2",
            ]
        )
        self.assertEqual(args.robot, "unitree-go2")
        self.assertEqual(args.transport, "dds://eth0?domain=0")

    def test_connect_read_only_flag(self):
        args = cli.build_parser().parse_args(
            [
                "connect",
                "unitree-go2",
                "--transport",
                "dds://lo?domain=1",
                "--read-only",
            ]
        )
        self.assertTrue(args.read_only)

    def test_sim_accepts_unitree_go2_robot_flag(self):
        args = cli.build_parser().parse_args(
            ["sim", "--robot", "unitree-go2", "--seconds", "1"]
        )
        self.assertEqual(args.robot, "unitree-go2")


class TestHardwareDdsRefusal(unittest.TestCase):
    def _fake_gated_bot(self, armed: bool) -> mock.Mock:
        gate = mock.Mock(spec=SafetyGate)
        gate.armed = armed
        bot = mock.Mock()
        bot.transport = gate
        bot.gated = True
        bot.disconnect = mock.Mock()
        return bot

    def test_drive_refuses_eth0_without_arm(self):
        args = cli.build_parser().parse_args(
            ["drive", "unitree-go2", "--transport", "dds://eth0"]
        )
        bot = self._fake_gated_bot(armed=False)
        with mock.patch.object(cli, "_connect", return_value=bot):
            with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                code = cli._cmd_drive(args)
        self.assertEqual(code, 2)
        bot.disconnect.assert_called_once()
        self.assertIn("hardware DDS", err.getvalue())

    def test_drive_refuses_lo_domain_0_without_arm(self):
        args = cli.build_parser().parse_args(
            ["drive", "unitree-go2", "--transport", "dds://lo"]
        )
        bot = self._fake_gated_bot(armed=False)
        with mock.patch.object(cli, "_connect", return_value=bot):
            code = cli._cmd_drive(args)
        self.assertEqual(code, 2)

    def test_drive_allows_sim_dds_uri_without_arm(self):
        args = cli.build_parser().parse_args(
            [
                "drive",
                "unitree-go2",
                "--transport",
                "dds://lo?domain=1",
                "--seconds",
                "0.01",
            ]
        )
        bot = self._fake_gated_bot(armed=False)
        bot.spec = get_spec("unitree-go2")
        bot.drive = mock.Mock()
        bot.stop = mock.Mock()
        bot.telemetry.return_value = None
        with mock.patch.object(cli, "_connect", return_value=bot):
            with mock.patch.object(cli.time, "monotonic", side_effect=[0.0, 1.0]):
                code = cli._cmd_drive(args)
        self.assertEqual(code, 0)

    def test_agent_refuses_hardware_dds_without_arm(self):
        args = cli.build_parser().parse_args(
            ["agent", "unitree-go2", "--transport", "dds://eth0", "stop"]
        )
        bot = self._fake_gated_bot(armed=False)
        with mock.patch.object(cli, "_connect", return_value=bot):
            code = cli._cmd_agent(args)
        self.assertEqual(code, 2)


class TestReadOnlyDdsConnect(unittest.TestCase):
    def _fake_unitree_sdk_modules(self, sport_client_cls: mock.Mock) -> dict:
        channel_init = mock.Mock()
        channel_sub = mock.Mock(return_value=mock.Mock())
        channel_mod = types.ModuleType("unitree_sdk2py.core.channel")
        channel_mod.ChannelFactoryInitialize = channel_init
        channel_mod.ChannelSubscriber = channel_sub
        sport_mod = types.ModuleType("unitree_sdk2py.go2.sport.sport_client")
        sport_mod.SportClient = sport_client_cls
        dds_mod = types.ModuleType("unitree_sdk2py.idl.unitree_go.msg.dds_")
        dds_mod.LowState_ = mock.Mock()
        dds_mod.SportModeState_ = mock.Mock()
        return {
            "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
            "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
            "unitree_sdk2py.core.channel": channel_mod,
            "unitree_sdk2py.go2": types.ModuleType("unitree_sdk2py.go2"),
            "unitree_sdk2py.go2.sport": types.ModuleType("unitree_sdk2py.go2.sport"),
            "unitree_sdk2py.go2.sport.sport_client": sport_mod,
            "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
            "unitree_sdk2py.idl.unitree_go": types.ModuleType(
                "unitree_sdk2py.idl.unitree_go"
            ),
            "unitree_sdk2py.idl.unitree_go.msg": types.ModuleType(
                "unitree_sdk2py.idl.unitree_go.msg"
            ),
            "unitree_sdk2py.idl.unitree_go.msg.dds_": dds_mod,
            "_channel_init": channel_init,
        }

    def test_read_only_skips_sport_client(self):
        tp = UnitreeDdsTransport(get_spec("unitree-go2"), "lo?domain=1", read_only=True)
        sport_client_cls = mock.Mock()
        mods = self._fake_unitree_sdk_modules(sport_client_cls)
        channel_init = mods.pop("_channel_init")
        with mock.patch.dict(sys.modules, mods):
            tp._connect_dds()
        sport_client_cls.assert_not_called()
        channel_init.assert_called_once()
        self.assertIsNone(tp._client)

    def test_connect_passes_read_only_to_robot(self):
        args = cli.build_parser().parse_args(
            [
                "connect",
                "unitree-go2",
                "--transport",
                "dds://lo?domain=1",
                "--read-only",
            ]
        )
        fake = mock.Mock()
        fake.gated = True
        fake.arm = mock.Mock()
        with mock.patch.object(cli.Robot, "connect", return_value=fake) as conn:
            cli._connect(args)
        conn.assert_called_once()
        self.assertTrue(conn.call_args.kwargs.get("read_only"))


if __name__ == "__main__":
    unittest.main()
