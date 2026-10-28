"""Spec and ``legged.pose`` parameters for the built-in quadruped."""

from __future__ import annotations

from ohho import capabilities as caps
from ohho.registry import RobotSpec
from ohho.robot_spec import RobotClass

ROBOT_ID = "unitree-go2"

# Poses the DDS adapter already forwards. ``euler`` and ``stop_move`` are
# commands, not poses, so they are not listed. This is not a safety allowlist.
POSES = (
    "stand_up",
    "stand_down",
    "balance_stand",
    "recovery_stand",
    "sit",
    "rise_sit",
    "damp",
)


def unitree_spec() -> RobotSpec:
    """The shipped quadruped spec.

    ``source`` is this plugin's name so discovery does not rewrite it.
    ``sim_backend`` stays empty so ``sim://`` and auto keep the in-process
    simulator. The MuJoCo backend is registered separately and used only when
    a spec names it.
    """
    return RobotSpec(
        id=ROBOT_ID,
        name="Unitree Go2",
        category="legged",
        capabilities=(
            caps.BASE_DRIVE,
            caps.BASE_HOLONOMIC,
            caps.PERCEPTION_RGB,
            caps.PERCEPTION_DEPTH,
            caps.PERCEPTION_SCAN,
            caps.LEGGED_POSE,
        ),
        adapter="unitree-dds",
        dof_base=3,
        dof_arm=0,
        max_lin=1.5,
        max_ang=2.0,
        robot_class=RobotClass.QUADRUPED,
        source="unitree",
        capability_params={caps.LEGGED_POSE: {"poses": list(POSES)}},
    )
