"""RobotPlugin for the Booster T2 spec.

Importing this module does not import a vendor SDK. There is no transport
here; the hardware scheme arrives in a later change. ``transports()`` is
empty so discovery cannot collide with another plugin's schemes.
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
        return {}

    def sim_backends(self) -> dict[str, Any]:
        return {}

    def safety_policy(self, spec: RobotSpec) -> None:
        """``None`` keeps the class default. ``SafetyGate`` does not read this."""
        del spec
        return None

    def descriptor(self, spec: RobotSpec) -> None:
        """Console descriptor stays with core. This plugin does not emit one."""
        del spec
        return None

    def prepare_sim(self, spec: RobotSpec) -> None:
        """Bind the walk config for this spec. Fetches pinned files if needed.

        Discovery does not call this. A sim connect does. Other robot ids
        are ignored.
        """
        if spec.id != "booster-t2":
            return
        from .sim import bind_walk_config

        bind_walk_config()

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
