"""Built-in robot plugin: sim, OmniBot and Unitree Go2.

Registered in ``pyproject.toml`` under the ``ohho.robots`` group, so the
built-ins take the same discovery path as an installed plugin. Spec data stays
in :mod:`ohho.registry`; this module only exposes it.

Importing this module must not import a vendor SDK (``pyserial``, ``lerobot``,
``cyclonedds``, ``unitree_sdk2py``, ``rclpy``, ``pymavlink``). Those imports
stay inside adapter ``connect()``. Transport factories below import the
adapter module only when called, and they do not connect.
"""

from __future__ import annotations

from typing import Any

from ..registry import RobotSpec, builtin_specs
from ..transport import Transport


def _sim_transport(spec: RobotSpec, address: str, runtime: Any) -> Transport:
    del address, runtime
    from ..adapters.sim import SimTransport

    return SimTransport(spec)


def _serial_transport(spec: RobotSpec | None, address: str, runtime: Any) -> Transport:
    """Yahboom base, or a base+arm composite when the spec can manipulate.

    Same split as the scheme-dict branch in ``resolve_transport``. That branch
    remains the fallback when this plugin did not load.
    """
    del runtime
    from .. import capabilities as caps
    from ..adapters.composite import CompositeTransport
    from ..adapters.feetech import DEFAULT_PORT as default_arm_port
    from ..adapters.feetech import FeetechTransport
    from ..adapters.yahboom import YahboomTransport

    if spec is not None and spec.has(caps.MANIPULATION):
        if "," in address:
            base_port, arm_port = address.split(",", 1)
            base_port, arm_port = base_port.strip(), arm_port.strip()
        else:
            base_port, arm_port = address.strip(), default_arm_port
        return CompositeTransport(
            YahboomTransport(spec, base_port),
            FeetechTransport(spec, arm_port),
        )
    # Callers may pass spec=None; the class stores it the same way the scheme dict does.
    return YahboomTransport(spec, address)  # type: ignore[arg-type]


def _feetech_transport(spec: RobotSpec, address: str, runtime: Any) -> Transport:
    del runtime
    from ..adapters.feetech import FeetechTransport

    return FeetechTransport(spec, address)


def _dds_transport(spec: RobotSpec, address: str, runtime: Any) -> Transport:
    del runtime
    from ..adapters.unitree import UnitreeDdsTransport

    return UnitreeDdsTransport(spec, address)


def _ros2_transport(spec: RobotSpec, address: str, runtime: Any) -> Transport:
    from ..adapters.ros2 import Ros2Transport

    return Ros2Transport(spec, address, runtime=runtime)


def _humanoid_mujoco_transport(
    spec: RobotSpec, address: str, runtime: Any
) -> Transport:
    """Generic humanoid MuJoCo link. MuJoCo itself is imported on connect."""
    del runtime
    from ..adapters.humanoid_mujoco import HumanoidMujocoTransport

    return HumanoidMujocoTransport(spec, address=address)


class BuiltinPlugin:
    """Specs and schemes that ship with ohho-os."""

    name = "builtin"

    def specs(self) -> list[RobotSpec]:
        return builtin_specs()

    def transports(self) -> dict[str, Any]:
        return {
            "sim": _sim_transport,
            "simulated": _sim_transport,
            "serial": _serial_transport,
            "yahboom-serial": _serial_transport,
            "feetech": _feetech_transport,
            "dds": _dds_transport,
            "unitree-dds": _dds_transport,
            "ros2": _ros2_transport,
        }

    def sim_backends(self) -> dict[str, Any]:
        from ..humanoid_sim import HUMANOID_MUJOCO_BACKEND

        return {
            "kinematic": _sim_transport,
            HUMANOID_MUJOCO_BACKEND: _humanoid_mujoco_transport,
        }

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
