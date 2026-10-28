"""Built-in plugin: spec, ``dds://``, MuJoCo sim backend, legged policy.

Registered as ``ohho.robots.unitree:plugin``. Importing this package does not
import a vendor SDK and does not connect to a robot.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ohho.plugins import DoctorReport
from ohho.registry import RobotSpec

from .policy import policy_for
from .policy_model import MODEL_NAMES
from .spec import ROBOT_ID, unitree_spec
from .transport import dds_transport, mujoco_sim

_SIM_BACKEND = "go2-mujoco"
_SELFTEST_CHECKS = (
    "dds_link",
    "leg_motors",
    "imu_rpy_rest",
    "bms_soc",
    "error_code",
    "front_camera",
    "stand_lie_cycle",
)


def _cyclonedds_line() -> str:
    if importlib.util.find_spec("cyclonedds") is None:
        return "missing  (pip install 'ohho-os[unitree]')"
    try:
        from importlib.metadata import version

        ver = version("cyclonedds")
    except Exception:
        return "installed (version unknown)"
    warn = ""
    if not ver.startswith("0.10."):
        warn = "  (warn: prefer 0.10.x for unitree_sdk2py)"
    return f"{ver}{warn}"


class UnitreePlugin:
    """Quadruped shipped with ohho-os. ``dds://`` stays this transport."""

    name = "unitree"

    def specs(self) -> list[RobotSpec]:
        return [unitree_spec()]

    def transports(self) -> dict[str, Any]:
        return {
            "dds": dds_transport,
            "unitree-dds": dds_transport,
        }

    def sim_backends(self) -> dict[str, Any]:
        return {_SIM_BACKEND: mujoco_sim}

    def safety_policy(self, spec: RobotSpec) -> object | None:
        """Legged rules for this spec. Discovery does not call this."""
        return policy_for(spec)

    def descriptor(self, spec: RobotSpec) -> Mapping[str, Any] | None:
        """RobotDescriptor v1 for this spec, or ``None`` for any other id."""
        if spec.id != ROBOT_ID:
            return None
        from .descriptor_doc import unitree_document

        return unitree_document(spec)

    def doctor(
        self,
        spec: RobotSpec,
        uri: str | None = None,
        read_only: bool = True,
    ) -> DoctorReport | None:
        """Read-only check. It does not open DDS or command the robot."""
        if spec.id != ROBOT_ID:
            return None
        if not read_only:
            return DoctorReport(
                ok=False,
                summary="doctor is read-only",
                details=("refusing to command the robot",),
            )
        sdk = importlib.util.find_spec("unitree_sdk2py") is not None
        details = [
            "transport: dds",
            f"sim backend: {_SIM_BACKEND} (loopback domain 1)",
            f"sdk: {'importable' if sdk else 'missing'}",
            f"poses: {len(spec.capability_params.get('legged.pose', {}).get('poses', ()))}",
        ]
        if uri:
            details.append(f"uri: {uri}")
        return DoctorReport(
            ok=True,
            summary="quadruped plugin is registered",
            details=tuple(details),
        )

    def probe_schemes(self) -> list[str]:
        """``dds`` when the SDK can be imported. Does not import it."""
        if importlib.util.find_spec("unitree_sdk2py") is not None:
            return ["dds"]
        return []

    def environment_lines(self) -> tuple[str, ...]:
        """Lines ``ohho doctor`` already printed for this adapter."""
        sdk = (
            "importable"
            if importlib.util.find_spec("unitree_sdk2py") is not None
            else "missing  (add [unitree] + GitHub SDK)"
        )
        home = os.environ.get("CYCLONEDDS_HOME", "")
        uri = os.environ.get("CYCLONEDDS_URI", "")
        return (
            f"  unitree_sdk : {sdk}",
            f"  cyclonedds  : {_cyclonedds_line()}",
            f"  CYCLONEDDS  : HOME={home or '(unset)'}  URI={uri or '(unset)'}",
            "  dds         : sim-only (domain 1 / lo) until hardware is armed",
        )

    def uri_is_hardware(self, uri: str | None) -> bool:
        from .dds import dds_uri_is_hardware

        return dds_uri_is_hardware(uri)

    def selftest_profile(self, spec: RobotSpec) -> object | None:
        if spec.id != ROBOT_ID:
            return None
        from ohho.selftest import SelfTestProfile

        return SelfTestProfile("legged_go2", _SELFTEST_CHECKS)

    def hardware_transport_uri(self, spec: RobotSpec) -> str | None:
        """Default DDS URI. ``OHHO_GO2_IFACE`` / ``OHHO_GO2_DOMAIN`` still apply."""
        if spec.id != ROBOT_ID:
            return None
        iface = os.environ.get("OHHO_GO2_IFACE", "eth0")
        domain = os.environ.get("OHHO_GO2_DOMAIN", "").strip()
        if domain:
            return f"dds://{iface}?domain={domain}"
        return f"dds://{iface}"

    def sim_demo_robot_id(self) -> str:
        return ROBOT_ID

    def sim_eval_path(self) -> str:
        return str(Path(__file__).resolve().parents[3] / "evals" / "go2_sim_goals.yaml")

    def teleop_repo_id(self) -> str:
        return "ohho-robotics/go2-sim-teleop-v0"

    def model_class(self, name: str) -> type | None:
        if name not in MODEL_NAMES:
            return None
        from .policy_model import Go2PolicyModel

        return Go2PolicyModel

    def model_uses_base_action(self, name: str) -> bool:
        return name in MODEL_NAMES

    def is_model(self, model: object) -> bool:
        from .policy_model import Go2PolicyModel

        return isinstance(model, Go2PolicyModel)

    def default_base_model_name(self) -> str:
        return "go2"


def plugin() -> UnitreePlugin:
    """Entry-point target: ``ohho.robots.unitree:plugin``."""
    return UnitreePlugin()
