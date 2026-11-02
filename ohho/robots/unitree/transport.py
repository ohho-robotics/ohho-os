"""``dds://`` transport. The factory does not connect and does not import the SDK.

The SDK is imported inside :meth:`UnitreeDdsTransport.connect`.
"""

from __future__ import annotations

from typing import Any

from ohho.registry import RobotSpec
from ohho.transport import Transport


def dds_transport(spec: RobotSpec, address: str, runtime: Any) -> Transport:
    """DDS link. ``read_only`` is set by ``resolve_transport`` after construction."""
    del runtime
    from .dds import UnitreeDdsTransport

    return UnitreeDdsTransport(spec, address)


dds_transport.supports_read_only = True  # type: ignore[attr-defined]


def mujoco_sim(spec: RobotSpec, address: str, runtime: Any) -> Transport:
    """Local MuJoCo sport shim: loopback, domain 1. Does not connect.

    An empty address uses that sim endpoint. ``sim://`` does not select this
    backend unless the spec names it.
    """
    del runtime
    from .dds import UnitreeDdsTransport

    target = address.strip() or "lo?domain=1"
    return UnitreeDdsTransport(spec, target)
