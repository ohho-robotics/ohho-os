"""Bench cases for the built-in mobile manipulator.

``ohho.hil`` loads :func:`bundle` through the plugin. Importing this module
does not open a port or command motion. The fake-transport checks below are
the same pulses the harness already used.
"""

from __future__ import annotations

import struct
from typing import Any

ENV_BASE = "OHHO_OMNIBOT_PORT"
ENV_ARM = "OHHO_OMNIBOT_ARM"


def bundle() -> dict[str, Any]:
    """Cases, device variables, report labels, and fake-transport checks."""
    from ohho.hil import HilCase

    cases = (
        HilCase(
            "omnibot_base",
            "OmniBot base drive and read",
            "yahboom",
            "TestHilOmniBotBase",
            (ENV_BASE,),
        ),
        HilCase(
            "omnibot_arm",
            "OmniBot arm move and read",
            "feetech",
            "TestHilOmniBotArm",
            (ENV_ARM,),
        ),
        HilCase(
            "omnibot_composite",
            "OmniBot composite base and arm",
            "yahboom + feetech",
            "TestHilOmniBotComposite",
            (ENV_BASE, ENV_ARM),
        ),
    )
    mocks = {
        "omnibot_base": mock_omnibot_base,
        "omnibot_arm": mock_omnibot_arm,
        "omnibot_composite": mock_omnibot_composite,
    }
    return {
        "cases": cases,
        "mocks": mocks,
        "exports": {
            "ENV_BASE": ENV_BASE,
            "ENV_ARM": ENV_ARM,
            "mock_omnibot_base": mock_omnibot_base,
            "mock_omnibot_arm": mock_omnibot_arm,
            "mock_omnibot_composite": mock_omnibot_composite,
        },
        "report_fields": (
            ("OmniBot base port", "base_port"),
            ("OmniBot arm port", "arm_port"),
        ),
    }


def mock_omnibot_base() -> str:
    """Fake-serial version of ``TestHilOmniBotBase``."""
    from ohho.hil import (
        HIL_BASE_VX,
        HIL_BASE_VY,
        HIL_BASE_W,
        _FakeSerial,
        _assert_base_pulse,
    )
    from ohho.registry import get_spec
    from ohho.schema import Velocity

    from .proto import packet_motion
    from .yahboom import YahboomTransport

    _assert_base_pulse()
    serial = _FakeSerial()
    transport = YahboomTransport(
        get_spec("omnibot"),
        "mock:omnibot-base",
        serial_factory=lambda: serial,
    )
    transport.connect()
    try:
        serial.written.clear()
        transport.send_velocity(Velocity(HIL_BASE_VX, HIL_BASE_VY, HIL_BASE_W))
        if bytes(serial.written) != packet_motion(HIL_BASE_VX, HIL_BASE_VY, HIL_BASE_W):
            raise AssertionError("base command was not the 0.05 m/s pulse")
        serial.written.clear()
        transport.send_velocity(Velocity())
        if bytes(serial.written) != packet_motion(0.0, 0.0, 0.0):
            raise AssertionError("base stop was not zero velocity")
        transport._ingest(_rx_velocity(HIL_BASE_VX, HIL_BASE_VY, HIL_BASE_W))
        reading = transport.read()
        if reading.odom is None:
            raise AssertionError("odom missing")
        if abs(reading.odom.vx - HIL_BASE_VX) > 1e-6:
            raise AssertionError(f"odom.vx={reading.odom.vx}")
    finally:
        transport.disconnect()
    return f"motion {HIL_BASE_VX:.2f} m/s then zero; odom.vx={HIL_BASE_VX:.3f}"


def mock_omnibot_arm() -> str:
    """Fake-bus version of ``TestHilOmniBotArm``."""
    from ohho.adapters.feetech import HOME_TICKS, FeetechTransport, radians_to_ticks
    from ohho.hil import (
        HIL_BASE_VX,
        HIL_GRIPPER_OPEN,
        HIL_GRIPPER_SHUT,
        _FakeFeetechBus,
        _assert_gripper_pulse,
    )
    from ohho.registry import get_spec
    from ohho.schema import Velocity

    _assert_gripper_pulse()
    spec = get_spec("omnibot")
    bus = _FakeFeetechBus(spec.joint_names)
    transport = FeetechTransport(spec, "mock:omnibot-arm", bus_factory=lambda: bus)
    transport.connect()
    try:
        transport.send_velocity(Velocity(HIL_BASE_VX, 0.0, 0.0))
        if bus.goals:
            raise AssertionError("arm accepted a base velocity")
        transport.send_joint_command("arm_gripper", HIL_GRIPPER_OPEN)
        open_tick = bus.goals.get("arm_gripper")
        transport.send_joint_command("arm_gripper", HIL_GRIPPER_SHUT)
        shut_tick = bus.goals.get("arm_gripper")
        expected_open = radians_to_ticks([HIL_GRIPPER_OPEN])[0]
        if open_tick != expected_open:
            raise AssertionError(f"gripper open tick {open_tick} != {expected_open}")
        if shut_tick != HOME_TICKS:
            raise AssertionError(f"gripper shut tick {shut_tick} != {HOME_TICKS}")
        transport._poll_once()
        reading = transport.read()
        if len(reading.joints) != 6:
            raise AssertionError(f"joints={len(reading.joints)}")
    finally:
        transport.disconnect()
    if set(bus.torque.values()) != {0}:
        raise AssertionError("disconnect left arm torque enabled")
    return f"gripper {HIL_GRIPPER_OPEN:.3f} then {HIL_GRIPPER_SHUT:.3f}; joints=6"


def mock_omnibot_composite() -> str:
    """Fake base+arm version of ``TestHilOmniBotComposite``."""
    from ohho.adapters.composite import CompositeTransport
    from ohho.adapters.feetech import FeetechTransport, radians_to_ticks
    from ohho.hil import (
        HIL_BASE_VX,
        HIL_BASE_VY,
        HIL_BASE_W,
        HIL_GRIPPER_OPEN,
        HIL_GRIPPER_SHUT,
        _FakeFeetechBus,
        _FakeSerial,
        _assert_base_pulse,
        _assert_gripper_pulse,
    )
    from ohho.registry import get_spec
    from ohho.schema import Velocity

    from .proto import packet_motion
    from .yahboom import YahboomTransport

    _assert_base_pulse()
    _assert_gripper_pulse()
    spec = get_spec("omnibot")
    serial = _FakeSerial()
    bus = _FakeFeetechBus(spec.joint_names)
    base = YahboomTransport(spec, "mock:omnibot-base", serial_factory=lambda: serial)
    arm = FeetechTransport(spec, "mock:omnibot-arm", bus_factory=lambda: bus)
    composite = CompositeTransport(base, arm)
    composite.connect()
    try:
        serial.written.clear()
        composite.send_velocity(Velocity(HIL_BASE_VX, HIL_BASE_VY, HIL_BASE_W))
        if bus.goals:
            raise AssertionError("composite velocity reached the arm")
        if bytes(serial.written) != packet_motion(HIL_BASE_VX, HIL_BASE_VY, HIL_BASE_W):
            raise AssertionError("composite base command was not the 0.05 m/s pulse")
        composite.send_joint_command("arm_gripper", HIL_GRIPPER_OPEN)
        expected_open = radians_to_ticks([HIL_GRIPPER_OPEN])[0]
        if bus.goals.get("arm_gripper") != expected_open:
            raise AssertionError("composite gripper command did not reach the arm")
        serial.written.clear()
        composite.send_velocity(Velocity())
        composite.send_joint_command("arm_gripper", HIL_GRIPPER_SHUT)
        if bytes(serial.written) != packet_motion(0.0, 0.0, 0.0):
            raise AssertionError("composite stop was not zero velocity")
        base._ingest(_rx_velocity(HIL_BASE_VX, HIL_BASE_VY, HIL_BASE_W))
        arm._poll_once()
        reading = composite.read()
        if reading.odom is None or abs(reading.odom.vx - HIL_BASE_VX) > 1e-6:
            raise AssertionError("composite odom was not the fed-back pulse")
        if len(reading.joints) != 6:
            raise AssertionError(f"joints={len(reading.joints)}")
    finally:
        composite.disconnect()
    return (
        f"base motion {HIL_BASE_VX:.2f} m/s and gripper {HIL_GRIPPER_OPEN:.3f}; "
        f"odom.vx={HIL_BASE_VX:.3f} joints=6"
    )


def _rx_velocity(vx: float, vy: float, vz: float) -> bytes:
    from . import proto

    payload = struct.pack("<hhh", int(vx * 1000), int(vy * 1000), int(vz * 1000))
    length = 3 + len(payload)
    body = bytes([length, proto.TYPE_VELOCITY]) + payload
    checksum = sum(body) & 0xFF
    return bytes([proto.HEAD_TX, proto.HEAD_RX]) + body + bytes([checksum])
