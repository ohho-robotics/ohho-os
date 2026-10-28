"""Built-in robot plugin: the generic sim and the aerial specs.

Registered in ``pyproject.toml`` under the ``ohho.robots`` group, so these
specs take the same discovery path as an installed plugin. The mobile
manipulator and the quadruped ship as their own entry points. Spec data stays
in :mod:`ohho.registry`; this module only exposes it.

Importing this module must not import a vendor SDK (``lerobot``, ``rclpy``,
``pymavlink``). Those imports stay inside adapter ``connect()``. Transport
factories below import the adapter module only when called, and they do not
connect.
"""

from __future__ import annotations

from typing import Any

from ..registry import RobotSpec, builtin_specs
from ..transport import Transport


def _sim_transport(spec: RobotSpec, address: str, runtime: Any) -> Transport:
    del address, runtime
    from ..adapters.sim import SimTransport

    return SimTransport(spec)


def _feetech_transport(spec: RobotSpec, address: str, runtime: Any) -> Transport:
    del runtime
    from ..adapters.feetech import FeetechTransport

    return FeetechTransport(spec, address)


def _ros2_transport(spec: RobotSpec, address: str, runtime: Any) -> Transport:
    from ..adapters.ros2 import Ros2Transport

    return Ros2Transport(spec, address, runtime=runtime)


class BuiltinPlugin:
    """Specs and schemes that ship with ohho-os."""

    name = "builtin"

    def specs(self) -> list[RobotSpec]:
        return builtin_specs()

    def transports(self) -> dict[str, Any]:
        return {
            "sim": _sim_transport,
            "simulated": _sim_transport,
            "feetech": _feetech_transport,
            "ros2": _ros2_transport,
        }

    def sim_backends(self) -> dict[str, Any]:
        return {"kinematic": _sim_transport}

    def safety_policy(self, spec: RobotSpec) -> None:
        """``None`` keeps the class default (ground, arm or legged).

        ``SafetyGate`` calls this when the robot connects. Discovery does not.
        """
        del spec
        return None

    def descriptor(self, spec: RobotSpec) -> None:
        """RobotDescriptor v1 arrives with C-04 / OHH-214."""
        del spec
        return None

    def doctor(
        self,
        spec: RobotSpec,
        uri: str | None = None,
        read_only: bool = True,
    ) -> None:
        """Doctor dispatch is R-06. This hook is not called while listing."""
        del spec, uri, read_only
        return None


def plugin() -> BuiltinPlugin:
    """Entry-point target: ``ohho.robots.builtin:plugin``."""
    return BuiltinPlugin()
