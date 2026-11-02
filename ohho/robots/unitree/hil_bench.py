"""Bench case for the built-in quadruped.

``ohho.hil`` loads :func:`bundle` through the plugin. Importing this module
does not open DDS or command motion. The fake-transport check below is the
same zero-velocity pulse the harness already used.
"""

from __future__ import annotations

from typing import Any

ENV_GO2 = "OHHO_GO2_IFACE"
HIL_GO2_VX = 0.0
HIL_GO2_VY = 0.0
HIL_GO2_W = 0.0


def bundle() -> dict[str, Any]:
    """Case, device variable, report label, and fake-transport check."""
    from ohho.hil import HilCase

    case = HilCase(
        "unitree_go2",
        "Unitree Go2 drive and read",
        "unitree DDS",
        "TestHilUnitreeGo2",
        (ENV_GO2,),
    )
    return {
        "cases": (case,),
        "mocks": {"unitree_go2": mock_unitree_go2},
        "exports": {
            "ENV_GO2": ENV_GO2,
            "HIL_GO2_VX": HIL_GO2_VX,
            "HIL_GO2_VY": HIL_GO2_VY,
            "HIL_GO2_W": HIL_GO2_W,
            "mock_unitree_go2": mock_unitree_go2,
        },
        "report_fields": (("Go2 interface", "go2_iface"),),
        "device_fields": (("go2_iface", ENV_GO2),),
        "cli_env": f"{ENV_GO2}=<iface>",
        "pulse_note": "and zero velocity on the Go2",
    }


def mock_unitree_go2() -> str:
    """Fake DDS version of ``TestHilUnitreeGo2``. Sends zero velocity only."""
    from ohho.registry import get_spec
    from ohho.schema import Velocity

    from .dds import UnitreeDdsTransport

    _assert_go2_pulse()
    state = {
        "position": [1.25, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "yaw_speed": 0.0,
        "imu_rpy": [0.0, 0.0, 0.3],
        "battery": 0.8,
        "error_code": 0,
    }
    client = _RecordingSport()
    transport = UnitreeDdsTransport(
        get_spec("unitree-go2"),
        "mock0",
        state_factory=lambda: lambda: state,
    )
    if transport._state_factory is None:
        raise AssertionError("Go2 mock must inject state_factory")
    transport.connect()
    try:
        transport._on_state(state)
        reading = transport.read()
        if reading.odom is None or abs(reading.odom.x - 1.25) > 1e-6:
            raise AssertionError("Go2 odom did not come from the fake state")
        transport._client = client
        transport.send_velocity(Velocity(HIL_GO2_VX, HIL_GO2_VY, HIL_GO2_W))
        if client.calls != [("Move", 0.0, 0.0, 0.0)]:
            raise AssertionError(f"Go2 command was not zero velocity: {client.calls}")
    finally:
        transport.disconnect()
    if ("StopMove",) not in client.calls:
        raise AssertionError("Go2 disconnect did not call StopMove")
    return "odom.x=1.250; Move(0, 0, 0); StopMove on disconnect"


def _assert_go2_pulse() -> None:
    if (HIL_GO2_VX, HIL_GO2_VY, HIL_GO2_W) != (0.0, 0.0, 0.0):
        raise AssertionError("Go2 HIL command must stay zero velocity")


class _RecordingSport:
    """Records SportClient calls. Implements only Move and StopMove."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def Move(self, vx: float, vy: float, vyaw: float) -> None:
        self.calls.append(("Move", vx, vy, vyaw))

    def StopMove(self) -> None:
        self.calls.append(("StopMove",))


def look_mock_allowed(uri: str | None) -> bool:
    """True for ``dds://`` on a loopback interface with a non-zero domain."""
    if uri is None:
        return False
    text = uri.strip()
    scheme, sep, rest = text.partition("://")
    if not sep or scheme.lower() != "dds":
        return False
    from ohho.safety import _is_loopback

    from .dds import parse_dds_address

    try:
        iface, domain = parse_dds_address(rest)
    except ValueError:
        return False
    return domain != 0 and _is_loopback(iface)
