"""RobotPlugin for the Booster T2 spec.

Importing this module does not import a vendor SDK. ``transports()`` registers
``booster://`` and ``booster+ros2://``. The SDK is imported only when the
``booster://`` factory runs.
"""

from __future__ import annotations

from typing import Any

from ohho.registry import RobotSpec

from .spec import booster_t2_spec


class BoosterPlugin:
    """Specs for this plugin. Safety stays the class default."""

    name = "booster"

    def specs(self) -> list[RobotSpec]:
        return [booster_t2_spec()]

    def transports(self) -> dict[str, Any]:
        from .factory import booster_transport, ros2_transport

        return {"booster": booster_transport, "booster+ros2": ros2_transport}

    def sim_backends(self) -> dict[str, Any]:
        return {}

    def safety_policy(self, spec: RobotSpec) -> None:
        """``None`` keeps the class default.

        ``SafetyGate`` loads this and refuses a policy that loosens class caps.
        """
        del spec
        return None

    def descriptor(self, spec: RobotSpec) -> None:
        """Console descriptor stays with core. This plugin does not emit one."""
        del spec
        return None

    def doctor(
        self,
        spec: RobotSpec,
        uri: str | None = None,
        read_only: bool = True,
    ) -> None:
        """Doctor dispatch is later. Listing plugins does not call this."""
        del spec, uri, read_only
        return None


def plugin() -> BoosterPlugin:
    """Entry-point target: ``ohho_booster:plugin``."""
    return BoosterPlugin()
