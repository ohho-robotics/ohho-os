import unittest

from ohho.adapters.sim import SimTransport
from ohho.adapters.unitree import (
    UnitreeDdsTransport,
    sport_mode_to_posture,
    sportstate_to_telemetry,
)
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime


class _FakeSportClient:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))

        return call


def _legged_bot():
    spec = get_spec("unitree-go2")
    tp = SimTransport(spec)
    tp.connect()
    return Robot(spec, tp, NativeRuntime()), tp


class TestLeggedPoseRobot(unittest.TestCase):
    def test_omnibot_pose_methods_are_noop(self):
        spec = get_spec("omnibot")
        tp = SimTransport(spec)
        tp.connect()
        bot = Robot(spec, tp, NativeRuntime())
        bot.stand().lie_down().sit().recover().set_body_pose(0.1, 0.0, 0.0)
        bot.drive(vx=0.2)
        tp.step(1.0)
        self.assertAlmostEqual(tp.read().odom.x, 0.2, places=5)

    def test_sim_drive_refused_while_lying(self):
        bot, tp = _legged_bot()
        self.assertEqual(tp.read().custom.get("posture"), "lying")
        bot.drive(vx=0.5)
        tp.step(1.0)
        self.assertAlmostEqual(tp.read().odom.x, 0.0, places=6)
        bot.stand()
        bot.drive(vx=0.5)
        tp.step(1.0)
        self.assertGreater(tp.read().odom.x, 0.1)

    def test_sim_posture_follows_send_mode(self):
        bot, tp = _legged_bot()
        bot.sit()
        self.assertEqual(tp.read().custom["posture"], "sitting")
        bot.lie_down()
        self.assertEqual(tp.read().custom["posture"], "lying")


class TestUnitreeSendMode(unittest.TestCase):
    def test_send_mode_maps_to_sport_client(self):
        tp = UnitreeDdsTransport(get_spec("unitree-go2"))
        client = _FakeSportClient()
        tp._client = client
        tp.send_mode("stand_up")
        tp.send_mode("stand_down")
        tp.send_mode("sit")
        tp.send_mode("recovery_stand")
        tp.send_mode("balance_stand")
        tp.send_mode("euler", roll=0.1, pitch=0.0, yaw=-0.05)
        self.assertEqual(
            client.calls,
            [
                ("StandUp", ()),
                ("StandDown", ()),
                ("Sit", ()),
                ("RecoveryStand", ()),
                ("BalanceStand", ()),
                ("Euler", (0.1, 0.0, -0.05)),
            ],
        )

    def test_mode_maps_to_posture_in_telemetry(self):
        self.assertEqual(sport_mode_to_posture(5), "lying")
        self.assertEqual(sport_mode_to_posture(10), "sitting")
        self.assertEqual(sport_mode_to_posture(1), "standing")
        t = sportstate_to_telemetry({"mode": 10})
        self.assertEqual(t.custom["posture"], "sitting")
        self.assertEqual(t.custom["mode"], 10)


class TestDriveCliLeggedSim(unittest.TestCase):
    def test_drive_unitree_sim_stands_then_lies_down(self):
        from ohho.cli import _cmd_drive

        class Args:
            robot = "unitree-go2"
            transport = "sim://"
            runtime = "native"
            vx = 0.2
            vy = 0.0
            w = 0.0
            seconds = 0.2
            safety_config = None
            estop_damp = False
            arm = False

        self.assertEqual(_cmd_drive(Args()), 0)


if __name__ == "__main__":
    unittest.main()
