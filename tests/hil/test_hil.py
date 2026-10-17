"""Hardware-in-the-loop tests.

Skipped unless ``OHHO_HIL=1`` and the device variable for that test is set.
An unset variable skips the test. Nothing here fills in ``/dev/ttyUSB0``,
``/dev/ttyACM0``, or ``eth0``.

These tests construct adapters directly (see AGENTS.md). They are not wrapped
in ``SafetyGate``. Keep a physical e-stop in hand. The only motion is the
pulse in ``ohho.hil`` (0.05 m/s on the base, gripper 0.2 rad, Go2 zero).

CI runs ``tests/test_hil_mock.py`` and ``ohho hil --mock``. On the bench, after
``docs/hil-preflight.md``::

    OHHO_HIL=1 OHHO_OMNIBOT_PORT=<base> OHHO_OMNIBOT_ARM=<arm> \\
        OHHO_GO2_IFACE=<iface> ohho hil --hardware --preflight-ok
"""

from __future__ import annotations

import os
import time
import unittest

from ohho import hil


def _require(*names: str) -> dict[str, str]:
    reason = hil.hardware_skip_reason(names)
    if reason:
        raise unittest.SkipTest(reason)
    return {name: os.environ[name].strip() for name in names}


def _skip(*names: str):
    reason = hil.hardware_skip_reason(names)
    if reason:
        return unittest.skip(reason)
    return lambda obj: obj


@_skip(hil.ENV_BASE)
class TestHilOmniBotBase(unittest.TestCase):
    """Drive the real OmniBot base over Yahboom serial."""

    def test_drive_and_read(self):
        from ohho.adapters.yahboom import YahboomTransport
        from ohho.registry import get_spec
        from ohho.schema import Velocity

        ports = _require(hil.ENV_BASE)
        spec = get_spec("omnibot")
        tp = YahboomTransport(spec, ports[hil.ENV_BASE])
        tp.connect()
        try:
            tp.send_velocity(Velocity(hil.HIL_BASE_VX, hil.HIL_BASE_VY, hil.HIL_BASE_W))
            time.sleep(1.0)
            tp.send_velocity(Velocity())
            time.sleep(0.2)
            reading = tp.read()
            self.assertIsNotNone(reading.odom)
        finally:
            tp.disconnect()


@_skip(hil.ENV_ARM)
class TestHilOmniBotArm(unittest.TestCase):
    """Move the real SO-101 arm over Feetech serial."""

    def test_move_joints_and_read(self):
        from ohho.adapters.feetech import FeetechTransport
        from ohho.registry import get_spec

        ports = _require(hil.ENV_ARM)
        spec = get_spec("omnibot")
        tp = FeetechTransport(spec, ports[hil.ENV_ARM])
        tp.connect()
        try:
            tp.send_joint_command("arm_gripper", hil.HIL_GRIPPER_OPEN)
            time.sleep(1.0)
            tp.send_joint_command("arm_gripper", hil.HIL_GRIPPER_SHUT)
            time.sleep(1.0)
            reading = tp.read()
            self.assertEqual(len(reading.joints), 6)
        finally:
            tp.disconnect()


@_skip(hil.ENV_BASE, hil.ENV_ARM)
class TestHilOmniBotComposite(unittest.TestCase):
    """Drive the full OmniBot (base + arm) through the composite transport."""

    def test_composite_drive_and_arm(self):
        from ohho.adapters.composite import CompositeTransport
        from ohho.adapters.feetech import FeetechTransport
        from ohho.adapters.yahboom import YahboomTransport
        from ohho.registry import get_spec
        from ohho.schema import Velocity

        ports = _require(hil.ENV_BASE, hil.ENV_ARM)
        spec = get_spec("omnibot")
        base = YahboomTransport(spec, ports[hil.ENV_BASE])
        arm = FeetechTransport(spec, ports[hil.ENV_ARM])
        composite = CompositeTransport(base, arm)
        composite.connect()
        try:
            composite.send_velocity(
                Velocity(hil.HIL_BASE_VX, hil.HIL_BASE_VY, hil.HIL_BASE_W)
            )
            composite.send_joint_command("arm_gripper", hil.HIL_GRIPPER_OPEN)
            time.sleep(1.0)
            composite.send_velocity(Velocity())
            composite.send_joint_command("arm_gripper", hil.HIL_GRIPPER_SHUT)
            time.sleep(0.5)
            reading = composite.read()
            self.assertIsNotNone(reading.odom)
            self.assertEqual(len(reading.joints), 6)
        finally:
            composite.disconnect()


@_skip(hil.ENV_GO2)
class TestHilUnitreeGo2(unittest.TestCase):
    """Read the real Unitree Go2 over DDS and send zero velocity."""

    def test_drive_and_read(self):
        from ohho.adapters.unitree import UnitreeDdsTransport
        from ohho.registry import get_spec
        from ohho.schema import Velocity

        ports = _require(hil.ENV_GO2)
        spec = get_spec("unitree-go2")
        tp = UnitreeDdsTransport(spec, ports[hil.ENV_GO2])
        tp.connect()
        try:
            time.sleep(1.0)
            reading = tp.read()
            self.assertIsNotNone(reading.odom)
            tp.send_velocity(Velocity(hil.HIL_GO2_VX, hil.HIL_GO2_VY, hil.HIL_GO2_W))
        finally:
            tp.disconnect()


if __name__ == "__main__":
    unittest.main()
