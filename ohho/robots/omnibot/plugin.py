"""Built-in plugin: spec, ``serial://`` composite, ground+arm policy, descriptor.

Registered as ``ohho.robots.omnibot:plugin``. Importing this package does not
import a vendor SDK and does not connect to a port.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ohho.plugins import DoctorReport
from ohho.registry import RobotSpec

from .policy import policy_for
from .spec import ROBOT_ID, omnibot_spec
from .transport import serial_transport

# Hugging Face checkpoint ids this robot's training defaults already used.
_CHECKPOINTS = {
    "act": "lerobot/act_omnibot",
    "diffusion": "lerobot/diffusion_omnibot",
}


class OmniBotPlugin:
    """Mobile manipulator shipped with ohho-os."""

    name = "omnibot"

    def specs(self) -> list[RobotSpec]:
        return [omnibot_spec()]

    def transports(self) -> dict[str, Any]:
        return {
            "serial": serial_transport,
            "yahboom-serial": serial_transport,
        }

    def sim_backends(self) -> dict[str, Any]:
        return {}

    def safety_policy(self, spec: RobotSpec) -> object | None:
        """Ground rules for the base and arm rules for the joints.

        ``SafetyGate`` calls this when the robot connects. Discovery does not.
        """
        return policy_for(spec)

    def descriptor(self, spec: RobotSpec) -> Mapping[str, Any] | None:
        """RobotDescriptor v1 for this spec, or ``None`` for any other id."""
        if spec.id != ROBOT_ID:
            return None
        from .descriptor_doc import omnibot_document

        return omnibot_document(spec)

    def doctor(
        self,
        spec: RobotSpec,
        uri: str | None = None,
        read_only: bool = True,
    ) -> DoctorReport | None:
        """Read-only check. It does not open a port or command the robot."""
        if spec.id != ROBOT_ID:
            return None
        if not read_only:
            return DoctorReport(
                ok=False,
                summary="doctor is read-only",
                details=("refusing to command the robot",),
            )
        details = [
            "base link: serial",
            "arm link: separate bus, composed when the URI is serial://base,arm",
            f"arm joints: {len(tuple(spec.joint_names))}",
        ]
        if uri:
            details.append(f"uri: {uri}")
        return DoctorReport(
            ok=True,
            summary="mobile manipulator plugin is registered",
            details=tuple(details),
        )

    def preferred_robot_id(self) -> str:
        """CLI default when the caller does not name a robot."""
        return ROBOT_ID

    def legacy_profile_env(self) -> str:
        """Older environment variable that still selects a hardware profile."""
        return "OMNIBOT_HW_PROFILE"

    def default_checkpoint(self, policy: str) -> str | None:
        """Training checkpoint id for ``policy``, or ``None`` when unset."""
        return _CHECKPOINTS.get(policy)

    def hil_bench(self) -> dict[str, Any]:
        """Bench cases and fake-transport checks for this robot.

        ``ohho.hil`` calls this when the harness loads. Discovery does not.
        The hook does not open a port or command motion.
        """
        from .hil_bench import bundle

        return bundle()


def plugin() -> OmniBotPlugin:
    """Entry-point target: ``ohho.robots.omnibot:plugin``."""
    return OmniBotPlugin()
