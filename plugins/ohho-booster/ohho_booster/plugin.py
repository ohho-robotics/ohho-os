"""RobotPlugin for the Booster T2 spec.

Importing this module does not import a vendor SDK. ``transports()`` registers
``booster://`` and ``booster+ros2://``. The SDK is imported only when the
``booster://`` factory runs.
"""

from __future__ import annotations

from typing import Any

from ohho.config import ConfigError, default_home, load_robot_profile
from ohho.plugins import DoctorReport
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
    ) -> DoctorReport:
        """Read-only checks: SDK, reachability, model, firmware, interface.

        ``read_only=False`` is ignored. This hook never commands the robot.
        The transport URI and network interface come from the arguments and
        from the resolved robot's profile.
        """
        del read_only
        from .doctor import diagnose

        address = uri if isinstance(uri, str) and uri.strip() else None
        interface = None
        try:
            profile = load_robot_profile(spec.id, home=default_home())
        except ConfigError:
            profile = None
        if profile is not None:
            if address is None and profile.transport:
                address = profile.transport
            interface = profile.network_interface
        return diagnose(address, network_interface=interface)


def plugin() -> BoosterPlugin:
    """Entry-point target: ``ohho_booster:plugin``."""
    return BoosterPlugin()
