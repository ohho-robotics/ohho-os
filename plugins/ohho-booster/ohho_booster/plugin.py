"""RobotPlugin for the Booster T2 and K1 specs.

Importing this module does not import a vendor SDK. There is no transport
here; the hardware scheme arrives in a later change. ``transports()`` is
empty so discovery cannot collide with another plugin's schemes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ohho.registry import RobotSpec

from .spec import booster_k1_spec, booster_t2_spec

_WALK_IDS = frozenset({"booster-t2", "booster-k1"})


class BoosterPlugin:
    """Specs for this plugin. Safety stays the class default."""

    name = "booster"

    def specs(self) -> list[RobotSpec]:
        return [booster_t2_spec(), booster_k1_spec()]

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

    def asset_source(self) -> Any:
        """Pinned description repository. The checkpoint is fetched beside it."""
        from ohho.plugins import AssetSource

        from .assets import description_repository, load_manifest

        pin = load_manifest()
        return AssetSource(
            repository=description_repository(pin), commit=str(pin["commit"])
        )

    def materialise_assets(self, dest: Path) -> None:
        """Store the pinned checkpoint in the sim-matrix checkout."""
        from .assets import materialise_checkout

        materialise_checkout(Path(dest))

    def prepare_sim(self, spec: RobotSpec) -> None:
        """Bind the walk config for this spec. Fetches pinned files if needed.

        Discovery does not call this. A sim connect does. Other robot ids
        are ignored.
        """
        if spec.id not in _WALK_IDS:
            return
        from .sim import bind_walk_config

        bind_walk_config(spec.id)

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
