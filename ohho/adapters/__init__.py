"""Adapter resolution.

Adapters connect the unified Transport interface to a robot's native protocol.
This build bundles: ``sim`` (always), ``feetech`` (SO-101 arm, needs ``[arm]``),
and ``composite`` (base+arm merge). ``serial://`` is provided by the
mobile-manipulator plugin: a spec with ``manipulation`` composes a base link
and an arm link. ``dds://`` is provided by the quadruped plugin.
``sim://`` for robot ids
``px4-sitl`` and ``crazyflie-sim`` selects the kinematic ``drone-sim`` backend.
``px4-x500`` is a registry spec only; nothing in this build connects to it.

Policy: **auto (no explicit transport) runs in simulation** so zero-config use and
tests never touch hardware; pass an explicit transport URI (``serial://…``,
``dds://…``, ``feetech://…``) to engage a real robot.

Plugin registries are asked first. A scheme a plugin registered uses that
factory. ``sim://`` (and auto) uses ``spec.sim_backend`` when it names a plugin
sim backend, and otherwise keeps the simulator return below. Schemes no plugin
registered — including entries added to ``_HARDWARE`` — still resolve here.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

from ..registry import RobotSpec
from ..transport import Transport
from .drone_sim import DroneSimTransport
from .errors import AdapterUnavailable
from .feetech import FeetechTransport
from .ros2 import Ros2Transport
from .sim import SimTransport

# sim:// with these robot ids is the kinematic drone stub, not the ground sim.
_DRONE_SIM_IDS = frozenset({"px4-sitl", "crazyflie-sim"})
# Registry data only. resolve_transport refuses every URI, including sim://.
_SPEC_ONLY_IDS = frozenset({"px4-x500"})

# transport scheme -> hardware adapter class (sim/composite handled separately).
# ``dds://`` is a plugin scheme. ``read_only`` is valid for ``dds`` and for any
# plugin factory that sets ``supports_read_only``.
_READ_ONLY_SCHEMES = frozenset({"dds"})
_HARDWARE = {
    "feetech": FeetechTransport,
    "ros2": Ros2Transport,
}


def available_adapters() -> list[str]:
    """Schemes usable in this environment (their runtime deps are importable)."""
    out = ["sim"]
    if importlib.util.find_spec("serial") is not None:
        out.append("serial")
    if importlib.util.find_spec("lerobot") is not None:
        out.append("feetech")
    from ..plugins import probe_adapter_schemes

    for scheme in probe_adapter_schemes():
        if scheme not in out:
            out.append(scheme)
    if importlib.util.find_spec("rclpy") is not None:
        out.append("ros2")
    return out


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
    - ``uri="serial://<port>"`` → the plugin that registered ``serial``. A
      mobile manipulator composes ``base_port,arm_port`` there.
    - ``uri="feetech://<port>"`` → Feetech arm adapter (arm only).
    - ``uri="dds://<iface>?domain=<n>"`` → the plugin that registered ``dds``
      (defaults ``dds://lo?domain=1`` for a local simulator).
    - ``uri="ros2://<namespace>"`` → ROS 2 transport (bridges to /cmd_vel, /odom,
      /arm/joint_states; requires ``runtime="ros2"``).
    - a scheme a plugin registered → that plugin's factory.
    - any other scheme → ``AdapterUnavailable``.
    - ``read_only=True`` is only valid for ``dds://`` and a plugin scheme whose
      factory sets ``supports_read_only``. Any other scheme raises
      :class:`AdapterUnavailable` before a transport that could still send
      motion is built.
    """
    if spec is not None and spec.id in _SPEC_ONLY_IDS:
        raise AdapterUnavailable(
            f"{spec.id} is a registry spec only; this build does not connect to it. "
            "requires_hardware_arming applies to non-loopback px4+dds:// and "
            "mavlink+serial:// URIs, which are not implemented."
        )
    scheme = _uri_scheme(uri)
    read_only_ok = scheme in _READ_ONLY_SCHEMES
    if read_only and not read_only_ok:
        from ..plugins import scheme_supports_read_only

        read_only_ok = scheme_supports_read_only(scheme)
    if read_only and not read_only_ok:
        shown = scheme if scheme else "sim"
        raise AdapterUnavailable(
            f"read_only is only supported for dds:// transports, not '{shown}'."
        )
    routed = _plugins_first(uri, spec, runtime)
    if routed is not None:
        if read_only:
            if not hasattr(routed, "read_only"):
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

    if scheme == "ros2":
        return Ros2Transport(spec, address, runtime=runtime)

    cls = _HARDWARE.get(scheme)
    if cls is None:
        from ..plugins import describe_plugin_schemes

        raise AdapterUnavailable(
            f"Unknown transport scheme '{scheme}'. Known: sim, serial, feetech, "
            f"dds, ros2 — or omit the transport to run in simulation. "
            f"{describe_plugin_schemes()}"
        )
    return cls(spec, address)


__all__ = [
    "SimTransport",
    "DroneSimTransport",
    "FeetechTransport",
    "Ros2Transport",
    "AdapterUnavailable",
    "resolve_transport",
    "available_adapters",
]
