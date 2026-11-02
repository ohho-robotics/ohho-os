"""``serial://`` transport: base link, or a base+arm composite.

A spec with ``manipulation`` and an address ``base,arm`` (or a single base
port plus the arm's default port) returns a composite. Any other spec gets
the base link only. The factory does not connect, and it does not import a
vendor SDK until the transport itself connects.
"""

from __future__ import annotations

from typing import Any

from ohho.registry import RobotSpec
from ohho.transport import Transport


def serial_transport(spec: RobotSpec | None, address: str, runtime: Any) -> Transport:
    """Base serial link, or base+arm when the spec can manipulate."""
    del runtime
    from ohho import capabilities as caps
    from ohho.adapters.composite import CompositeTransport
    from ohho.adapters.feetech import DEFAULT_PORT as default_arm_port
    from ohho.adapters.feetech import FeetechTransport

    from .yahboom import YahboomTransport

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
    return YahboomTransport(spec, address)  # type: ignore[arg-type]
