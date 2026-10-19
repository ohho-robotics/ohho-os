"""Adapter resolution.

Adapters connect the unified Transport interface to a robot's native protocol.
This build bundles: ``sim`` (always), ``serial`` (Yahboom / OmniBot base, needs
the ``[serial]`` extra at connect time), ``feetech`` (SO-101 arm, needs ``[arm]``),
``composite`` (base+arm merge for mobile-manipulators), and ``dds`` (Unitree Go2,
needs ``[unitree]``). ``sim://`` for robot ids ``px4-sitl`` and ``crazyflie-sim``
selects the kinematic ``drone-sim`` backend. ``crazyflie://`` (cflib) and
``crazyflie+ros2://`` (Crazyswarm2) select the Crazyflie backend.
``px4-x500`` is a registry spec only; nothing in this build connects to it.

Policy: **auto (no explicit transport) runs in simulation** so zero-config use and
tests never touch hardware; pass an explicit transport URI (``serial://…``,
``dds://…``, ``feetech://…``) to engage a real robot. For a mobile-manipulator
that has ``manipulation`` capability, ``serial://<base_port>,<arm_port>``
auto-composes a composite (Yahboom base + Feetech arm).
"""

from __future__ import annotations

import importlib.util
from typing import Optional

from .. import capabilities as caps
from ..registry import RobotSpec
from ..transport import Transport
from .composite import CompositeTransport
from .crazyflie import CrazyflieTransport
from .drone_sim import DroneSimTransport
from .errors import AdapterUnavailable
from .feetech import DEFAULT_PORT as DEFAULT_ARM_PORT
from .feetech import FeetechTransport
from .ros2 import Ros2Transport
from .sim import SimTransport
from .unitree import UnitreeDdsTransport
from .yahboom import YahboomTransport

# sim:// with these robot ids is the kinematic drone stub, not the ground sim.
_DRONE_SIM_IDS = frozenset({"px4-sitl", "crazyflie-sim"})
# Registry data only. resolve_transport refuses every URI, including sim://.
_SPEC_ONLY_IDS = frozenset({"px4-x500"})

# transport scheme -> hardware adapter class (sim/composite handled separately)
_HARDWARE = {
    "serial": YahboomTransport,
    "yahboom-serial": YahboomTransport,
    "feetech": FeetechTransport,
    "dds": UnitreeDdsTransport,
    "unitree-dds": UnitreeDdsTransport,
    "ros2": Ros2Transport,
}


def available_adapters() -> list[str]:
    """Schemes usable in this environment (their runtime deps are importable)."""
    out = ["sim"]
    if importlib.util.find_spec("serial") is not None:
        out.append("serial")
    if importlib.util.find_spec("lerobot") is not None:
        out.append("feetech")
    if importlib.util.find_spec("cyclonedds") is not None:
        out.append("dds")
    if importlib.util.find_spec("rclpy") is not None:
        out.append("ros2")
        out.append("crazyflie+ros2")
    if importlib.util.find_spec("cflib") is not None:
        out.append("crazyflie")
    return out


def _split_ports(address: str) -> tuple[str, str]:
    """Split ``base_port,arm_port`` into parts; arm defaults to the Feetech port."""
    if "," in address:
        base, arm = address.split(",", 1)
        return base.strip(), arm.strip()
    return address.strip(), DEFAULT_ARM_PORT


def _sim_transport(spec: Optional[RobotSpec]) -> Transport:
    """Ground sim, or drone-sim when the robot id is a kinematic aircraft."""
    if spec is not None and spec.id in _DRONE_SIM_IDS:
        return DroneSimTransport(spec)
    return SimTransport(spec)  # type: ignore[arg-type]


def resolve_transport(
    uri: Optional[str] = None,
    spec: Optional[RobotSpec] = None,
    runtime=None,
) -> Transport:
    """Pick a transport for a robot.

    - ``uri=None`` (auto) → the simulator (explicit URIs engage hardware).
      Robot ids ``px4-sitl`` and ``crazyflie-sim`` get ``drone-sim``.
    - ``uri="sim://"`` → the ground simulator, except those two ids, which
      select ``drone-sim``.
    - ``px4-x500`` → ``AdapterUnavailable`` (spec only; nothing connects).
    - ``uri="serial://<port>"`` → Yahboom serial adapter. If the robot has
      ``manipulation`` capability, ``<port>`` may be ``base_port,arm_port`` and
      a composite (Yahboom base + Feetech arm) is returned.
    - ``uri="feetech://<port>"`` → Feetech arm adapter (arm only).
    - ``uri="dds://<iface>"`` → Unitree DDS adapter.
    - ``uri="ros2://<namespace>"`` → ROS 2 transport (bridges to /cmd_vel, /odom,
      /arm/joint_states; requires ``runtime="ros2"``).
    - ``uri="crazyflie://radio://…"`` → Crazyflie radio via cflib.
    - ``uri="crazyflie+ros2://<name>"`` → Crazyswarm2 services for that robot.
    - any other scheme → ``AdapterUnavailable``.
    """
    if spec is not None and spec.id in _SPEC_ONLY_IDS:
        raise AdapterUnavailable(
            f"{spec.id} is a registry spec only; this build does not connect to it. "
            "requires_hardware_arming applies to non-loopback px4+dds:// and "
            "mavlink+serial:// URIs, which are not implemented."
        )
    if uri is None:
        return _sim_transport(spec)
    scheme, _sep, address = uri.partition("://")
    scheme = scheme.lower()
    if scheme in ("sim", "simulated"):
        return _sim_transport(spec)

    if (
        scheme in ("serial", "yahboom-serial")
        and spec is not None
        and spec.has(caps.MANIPULATION)
    ):
        base_port, arm_port = _split_ports(address)
        base = YahboomTransport(spec, base_port)
        arm = FeetechTransport(spec, arm_port)
        return CompositeTransport(base, arm)

    if scheme == "ros2":
        return Ros2Transport(spec, address, runtime=runtime)

    if scheme == "crazyflie":
        if spec is None:
            raise AdapterUnavailable("crazyflie:// needs a robot spec")
        return CrazyflieTransport(spec, address, kind="cflib", runtime=runtime)
    if scheme == "crazyflie+ros2":
        if spec is None:
            raise AdapterUnavailable("crazyflie+ros2:// needs a robot spec")
        return CrazyflieTransport(spec, address, kind="crazyswarm2", runtime=runtime)

    cls = _HARDWARE.get(scheme)
    if cls is None:
        raise AdapterUnavailable(
            f"Unknown transport scheme '{scheme}'. Known: sim, serial, feetech, "
            f"dds, ros2, crazyflie, crazyflie+ros2 — or omit the transport to "
            f"run in simulation."
        )
    return cls(spec, address)


__all__ = [
    "SimTransport",
    "DroneSimTransport",
    "YahboomTransport",
    "FeetechTransport",
    "UnitreeDdsTransport",
    "Ros2Transport",
    "CrazyflieTransport",
    "CompositeTransport",
    "AdapterUnavailable",
    "resolve_transport",
    "available_adapters",
]
