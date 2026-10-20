"""``booster+ros2://`` transport that refuses to connect.

The ROS 2 distro on the robot is an open question, so the ``booster_interface``
path (``LowState``, ``LowCmd``, ``MotorCmd``, ``FallDownState``, ``Odometer``,
``RpcService``) is not implemented. This stub is registered so the scheme is
visible and every connect fails closed. It does not import a vendor SDK.
"""

from __future__ import annotations

from typing import Any

from ohho.schema import ConnectionState, Telemetry, TransportStatus, Velocity
from ohho.transport import BaseTransport

_REASON = (
    "booster+ros2:// refuses to connect. The ROS 2 distro on this robot "
    "is an open question, so the booster_interface path is not implemented. "
    "Use booster:// or sim://."
)


class Ros2InterfaceTransport(BaseTransport):
    """Registered scheme that does not open a link."""

    protocol = "booster+ros2"

    def __init__(self, spec: object, address: str = "") -> None:
        super().__init__()
        self.spec = spec
        self.address = address
        self.host = address.partition("?")[0]
        self._state = ConnectionState.IDLE

    def connect(self) -> TransportStatus:
        self._state = ConnectionState.ERROR
        from ohho.adapters.errors import AdapterUnavailable

        raise AdapterUnavailable(_REASON)

    def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    def status(self) -> TransportStatus:
        return TransportStatus(
            protocol=self.protocol,
            state=self._state,
            label=self.address,
            error=_REASON,
        )

    def read(self) -> Telemetry:
        return Telemetry()

    def send_velocity(self, vel: Velocity) -> None:
        del vel
        self._refuse()

    def send_joint_command(self, name: str, position: float) -> None:
        del name, position
        self._refuse()

    def send_humanoid(self, action: str, *args: object, **fields: object) -> None:
        del action, args, fields
        self._refuse()

    def emergency_stop(self) -> None:
        self._refuse()

    def release_stop(self) -> None:
        self._refuse()

    def _refuse(self) -> None:
        from ohho.adapters.errors import AdapterUnavailable

        raise AdapterUnavailable(_REASON)


def ros2_transport(spec: object, address: str, runtime: Any) -> Ros2InterfaceTransport:
    """Factory for ``booster+ros2``. Does not connect and does not import an SDK."""
    del runtime
    return Ros2InterfaceTransport(spec, address)
