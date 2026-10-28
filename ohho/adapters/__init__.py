"""Adapter resolution.

Adapters connect the unified Transport interface to a robot's native protocol.
This build bundles: ``sim`` (always), ``serial`` (Yahboom / OmniBot base, needs
the ``[serial]`` extra at connect time), ``feetech`` (SO-101 arm, needs ``[arm]``),
``composite`` (base+arm merge for mobile-manipulators), ``dds`` (Unitree Go2,
needs ``[unitree]``), and ``px4+dds`` (PX4 v1.17 uXRCE-DDS, needs a ROS 2
graph and ``px4_msgs`` release/1.17 at connect time). ``sim://`` for robot ids
``px4-sitl`` and ``crazyflie-sim`` selects the kinematic ``drone-sim`` backend.
``px4-x500`` is a registry spec only; nothing in this build connects to it.

Policy: **auto (no explicit transport) runs in simulation** so zero-config use and
tests never touch hardware; pass an explicit transport URI (``serial://…``,
``dds://…``, ``feetech://…``) to engage a real robot. For a mobile-manipulator
that has ``manipulation`` capability, ``serial://<base_port>,<arm_port>``
auto-composes a composite (Yahboom base + Feetech arm).

Plugin registries are asked first. A scheme a plugin registered uses that
factory. ``sim://`` (and auto) uses ``spec.sim_backend`` when it names a plugin
sim backend, and otherwise keeps the simulator return below. Schemes no plugin
registered — including entries added to ``_HARDWARE`` — still resolve here.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

from .. import capabilities as caps
from ..registry import RobotSpec
from ..transport import Transport
from .composite import CompositeTransport
from .drone_sim import DroneSimTransport
from .errors import AdapterUnavailable
from .feetech import DEFAULT_PORT as DEFAULT_ARM_PORT
from .feetech import FeetechTransport
from .px4_dds import Px4DdsTransport
from .ros2 import Ros2Transport
from .sim import SimTransport
from .unitree import UnitreeDdsTransport
from .yahboom import YahboomTransport

# sim:// with these robot ids is the kinematic drone stub, not the ground sim.
_DRONE_SIM_IDS = frozenset({"px4-sitl", "crazyflie-sim"})
# Registry data only. resolve_transport refuses every URI, including sim://.
_SPEC_ONLY_IDS = frozenset({"px4-x500"})

# transport scheme -> hardware adapter class (sim/composite handled separately)
_READ_ONLY_SCHEMES = frozenset({"dds", "unitree-dds"})
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
    if importlib.util.find_spec("unitree_sdk2py") is not None:
        out.append("dds")
    if importlib.util.find_spec("rclpy") is not None:
        out.append("ros2")
    if (
        importlib.util.find_spec("rclpy") is not None
        and importlib.util.find_spec("px4_msgs") is not None
    ):
        out.append("px4+dds")
    return out


def _split_ports(address: str) -> tuple[str, str]:
    """Split ``base_port,arm_port`` into parts; arm defaults to the Feetech port."""
    if "," in address:
        base, arm = address.split(",", 1)
        return base.strip(), arm.strip()
    return address.strip(), DEFAULT_ARM_PORT


def _uri_scheme(uri: Optional[str]) -> Optional[str]:
    if not uri:
        return None
    scheme, sep, _address = uri.partition("://")
    if not sep:
        return None
    return scheme.lower()


def _sim_transport(spec: Optional[RobotSpec]) -> Transport:
    """Ground sim, or drone-sim when the robot id is a kinematic aircraft."""
    if spec is not None and spec.id in _DRONE_SIM_IDS:
        return DroneSimTransport(spec)
    return SimTransport(spec)  # type: ignore[arg-type]


def _plugins_first(
    uri: Optional[str], spec: Optional[RobotSpec], runtime: object
) -> Optional[Transport]:
    """Plugin transport, or the spec's sim backend. ``None`` uses the dict below.

    ``sim://`` and auto stay on the returns in :func:`resolve_transport` unless
    the spec declares ``sim_backend``. That keeps an undeclared ``sim://`` on
    today's simulator, including the id-based drone-sim branch in
    :func:`_sim_transport`.
    """
    from ..plugins import plugin_sim_backend, plugin_transport

    if uri is None:
        return plugin_sim_backend(spec, "", runtime)
    scheme, _sep, address = uri.partition("://")
    scheme = scheme.lower()
    if scheme in ("sim", "simulated"):
        return plugin_sim_backend(spec, address, runtime)
    return plugin_transport(scheme, spec, address, runtime)


def resolve_transport(
    uri: Optional[str] = None,
    spec: Optional[RobotSpec] = None,
    runtime=None,
    *,
    read_only: bool = False,
) -> Transport:
    """Pick a transport for a robot.

    - ``uri=None`` (auto) → the simulator (explicit URIs engage hardware).
      Robot ids ``px4-sitl`` and ``crazyflie-sim`` get ``drone-sim``.
      A declared ``spec.sim_backend`` selects that plugin backend instead.
    - ``uri="sim://"`` → the ground simulator, except those two ids, which
      select ``drone-sim``, or the spec's declared sim backend.
    - ``px4-x500`` → ``AdapterUnavailable`` (spec only; nothing connects).
    - ``uri="serial://<port>"`` → Yahboom serial adapter. If the robot has
      ``manipulation`` capability, ``<port>`` may be ``base_port,arm_port`` and
      a composite (Yahboom base + Feetech arm) is returned.
    - ``uri="feetech://<port>"`` → Feetech arm adapter (arm only).
    - ``uri="dds://<iface>?domain=<n>"`` → Unitree DDS adapter (defaults
      ``dds://lo?domain=1`` for local simulators such as unitree_mujoco).
    - ``uri="ros2://<namespace>"`` → ROS 2 transport (bridges to /cmd_vel, /odom,
      /arm/joint_states; requires ``runtime="ros2"``).
    - ``uri="px4+dds://HOST:PORT?ns=NAME"`` → PX4 uXRCE-DDS transport. This is
      not ``dds://`` (that scheme stays Unitree). ``Robot.connect`` wraps the
      result in ``SafetyGate``. Loopback hosts set ``sim_target``, which is
      how the gate tells a sim agent from a vehicle.
    - a scheme a plugin registered → that plugin's factory.
    - any other scheme → ``AdapterUnavailable``.
    - ``read_only=True`` is only valid for ``dds://`` and ``unitree-dds://``.
      Any other scheme raises :class:`AdapterUnavailable` before a transport
      that could still send motion is built.
    """
    if spec is not None and spec.id in _SPEC_ONLY_IDS:
        raise AdapterUnavailable(
            f"{spec.id} is a registry spec only; this build does not connect to it. "
            "requires_hardware_arming applies to a non-loopback px4+dds:// URI "
            "and to mavlink+serial://, which is not implemented."
        )
    scheme = _uri_scheme(uri)
    if read_only and scheme not in _READ_ONLY_SCHEMES:
        shown = scheme if scheme else "sim"
        raise AdapterUnavailable(
            f"read_only is only supported for dds:// transports, not '{shown}'."
        )
    routed = _plugins_first(uri, spec, runtime)
    if routed is not None:
        if read_only:
            if not isinstance(routed, UnitreeDdsTransport):
                shown = scheme if scheme else "sim"
                raise AdapterUnavailable(
                    f"read_only is only supported for dds:// transports, not '{shown}'."
                )
            routed.read_only = True
        return routed
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

    if scheme == "px4+dds":
        return Px4DdsTransport(spec, uri, runtime=runtime)

    cls = _HARDWARE.get(scheme)
    if cls is None:
        from ..plugins import describe_plugin_schemes

        raise AdapterUnavailable(
            f"Unknown transport scheme '{scheme}'. Known: sim, serial, feetech, "
            f"dds, ros2, px4+dds — or omit the transport to run in simulation. "
            f"{describe_plugin_schemes()}"
        )
    if cls is UnitreeDdsTransport:
        return UnitreeDdsTransport(spec, address, read_only=read_only)
    return cls(spec, address)


__all__ = [
    "SimTransport",
    "DroneSimTransport",
    "YahboomTransport",
    "FeetechTransport",
    "UnitreeDdsTransport",
    "Ros2Transport",
    "Px4DdsTransport",
    "CompositeTransport",
    "AdapterUnavailable",
    "resolve_transport",
    "available_adapters",
]
