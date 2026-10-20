"""Transport factories.

:func:`booster_transport` is the production caller of
:func:`ohho_booster.sdk.open_session`, which is what imports the vendor SDK.
Importing this module does not.
"""

from __future__ import annotations

from typing import Any

from ohho.registry import RobotSpec

from .ros2_stub import ros2_transport
from .transport import BoosterTransport

__all__ = ["booster_transport", "ros2_transport"]


def booster_transport(
    spec: RobotSpec | None, address: str, runtime: Any
) -> BoosterTransport:
    """Build a ``booster://`` link. Importing this function does not import the SDK."""
    del runtime
    from ohho.adapters.errors import AdapterUnavailable

    from .sdk import SdkUnavailable, open_session

    try:
        session = open_session(str(address or ""))
    except (SdkUnavailable, ValueError) as exc:
        raise AdapterUnavailable(str(exc)) from exc
    return BoosterTransport(spec, str(address or ""), session)
