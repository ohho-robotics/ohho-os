import unittest

from ohho import capabilities as caps
from ohho.capability_params import (
    CONTROL_MODES,
    PARAMETERIZED,
    POSTURE_MODES,
    AerialParams,
    DriveParams,
    DualArmParams,
    EndEffectorParams,
    HeadGazeParams,
    HumanoidLocomotionParams,
    HumanoidPostureParams,
    LeggedPoseParams,
    ManipulationParams,
    WholeBodyJointParams,
)
from ohho.registry import RobotSpec
from ohho.robot_spec import (
    AerialLimits,
    HumanoidLimits,
    MobileManipulatorLimits,
    GroundLimits,
    thaw_json,
)


class TestCapabilities(unittest.TestCase):
    def test_membership(self):
        self.assertIn(caps.MANIPULATION, caps.ALL)
        self.assertIn(caps.BASE_HOLONOMIC, caps.ALL)
        self.assertIn(caps.AERIAL, caps.ALL)
        self.assertEqual(caps.AERIAL, "aerial")
        self.assertTrue(PARAMETERIZED <= caps.ALL)

    def test_is_known(self):
        self.assertTrue(caps.is_known(caps.BASE_DRIVE))
        self.assertTrue(caps.is_known(caps.AERIAL))
        self.assertTrue(caps.is_known("aerial"))
        self.assertTrue(caps.is_known(caps.HUMANOID_POSTURE))
        self.assertTrue(caps.is_known(caps.HEAD_GAZE))
        self.assertFalse(caps.is_known("not.a.capability"))

    def test_parameter_schemas_round_trip(self):
        cases = {
            caps.BASE_DRIVE: DriveParams(0.2, 0.0, 1.0),
            caps.BASE_HOLONOMIC: DriveParams(0.2, 0.2, 1.0),
            caps.LEGGED_POSE: LeggedPoseParams(("stand_up", "sit")),
            caps.AERIAL: AerialParams(("base_link",), 1.5),
            caps.HUMANOID_LOCOMOTION: HumanoidLocomotionParams(
                ("walk",), speed_min=0.0, speed_max=0.4
            ),
            caps.HUMANOID_POSTURE: HumanoidPostureParams(("prepare", "walk")),
            caps.WHOLE_BODY_JOINT: WholeBodyJointParams(
                {"legs": ("hip", "knee")}, ("position", "torque"), 50.0
            ),
            caps.MANIPULATION: ManipulationParams(("arm",), ("ee",)),
            caps.MANIPULATION_DUAL_ARM: DualArmParams(
                "left_arm", "right_arm", ("left_arm", "right_arm"), ()
            ),
            caps.END_EFFECTOR_GRIPPER: EndEffectorParams(1, 0.0, 20.0),
            caps.END_EFFECTOR_DEX_HAND: EndEffectorParams(6, None, None),
            caps.HEAD_GAZE: HeadGazeParams(-0.5, 0.5, -1.0, 1.0),
        }
        self.assertEqual(set(cases), set(PARAMETERIZED))
        for name, params in cases.items():
            with self.subTest(capability=name):
                parser = {
                    caps.BASE_DRIVE: lambda raw: DriveParams.from_dict(
                        raw, capability=caps.BASE_DRIVE
                    ),
                    caps.BASE_HOLONOMIC: lambda raw: DriveParams.from_dict(
                        raw, capability=caps.BASE_HOLONOMIC
                    ),
                    caps.LEGGED_POSE: LeggedPoseParams.from_dict,
                    caps.AERIAL: AerialParams.from_dict,
                    caps.HUMANOID_LOCOMOTION: HumanoidLocomotionParams.from_dict,
                    caps.HUMANOID_POSTURE: HumanoidPostureParams.from_dict,
                    caps.WHOLE_BODY_JOINT: WholeBodyJointParams.from_dict,
                    caps.MANIPULATION: ManipulationParams.from_dict,
                    caps.MANIPULATION_DUAL_ARM: DualArmParams.from_dict,
                    caps.END_EFFECTOR_GRIPPER: lambda raw: EndEffectorParams.from_dict(
                        raw, capability=caps.END_EFFECTOR_GRIPPER
                    ),
                    caps.END_EFFECTOR_DEX_HAND: lambda raw: EndEffectorParams.from_dict(
                        raw, capability=caps.END_EFFECTOR_DEX_HAND
                    ),
                    caps.HEAD_GAZE: HeadGazeParams.from_dict,
                }[name]
                self.assertEqual(parser(params.to_dict()), params)

    def test_posture_modes_are_the_closed_subset(self):
        self.assertEqual(
            POSTURE_MODES,
            ("damping", "prepare", "walk", "custom", "get_up", "lie_down"),
        )
        self.assertEqual(CONTROL_MODES, ("position", "torque", "hybrid"))
        with self.assertRaises(ValueError):
            HumanoidPostureParams.from_dict({"modes": ["dance"]})
        with self.assertRaises(ValueError):
            WholeBodyJointParams.from_dict(
                {
                    "joint_groups": {},
                    "control_modes": ["velocity"],
                    "rate_hz": None,
                }
            )

    def test_locomotion_speed_max_alone_still_parses(self):
        parsed = HumanoidLocomotionParams.from_dict(
            {"gaits": ["walk", "turn"], "speed_max": 0.4}
        )
        self.assertEqual(parsed.gaits, ("walk", "turn"))
        self.assertIsNone(parsed.speed_min)
        self.assertEqual(parsed.speed_max, 0.4)

    def test_derived_params_follow_the_spec(self):
        spec = RobotSpec(
            id="typed-human",
            name="Typed Human",
            category="humanoid",
            capabilities=(
                caps.HUMANOID_LOCOMOTION,
                caps.HUMANOID_POSTURE,
                caps.WHOLE_BODY_JOINT,
                caps.MANIPULATION,
                caps.MANIPULATION_DUAL_ARM,
                caps.END_EFFECTOR_GRIPPER,
                caps.END_EFFECTOR_DEX_HAND,
                caps.HEAD_GAZE,
            ),
            max_lin=0.4,
            max_ang=0.5,
            joint_groups={
                "left_arm": ("l_shoulder",),
                "right_arm": ("r_shoulder",),
                "gripper": ("jaw",),
                "right_hand": ("finger", "thumb"),
            },
            limits=HumanoidLimits(max_vx=0.4, max_vy=0.2, max_wz=0.5),
        )
        params = thaw_json(spec.capability_params)
        self.assertEqual(params[caps.HUMANOID_LOCOMOTION]["speed_max"], 0.4)
        self.assertEqual(params[caps.HUMANOID_LOCOMOTION]["speed_min"], 0.0)
        self.assertEqual(params[caps.HUMANOID_LOCOMOTION]["gaits"], [])
        self.assertEqual(params[caps.HUMANOID_POSTURE]["modes"], [])
        self.assertEqual(
            params[caps.WHOLE_BODY_JOINT]["joint_groups"]["left_arm"], ["l_shoulder"]
        )
        self.assertEqual(params[caps.WHOLE_BODY_JOINT]["control_modes"], [])
        self.assertIsNone(params[caps.WHOLE_BODY_JOINT]["rate_hz"])
        self.assertEqual(params[caps.MANIPULATION_DUAL_ARM]["left"], "left_arm")
        self.assertEqual(params[caps.MANIPULATION_DUAL_ARM]["right"], "right_arm")
        self.assertEqual(params[caps.END_EFFECTOR_GRIPPER]["dof"], 1)
        self.assertEqual(params[caps.END_EFFECTOR_DEX_HAND]["dof"], 2)
        self.assertIsNone(params[caps.HEAD_GAZE]["pitch_max"])
        self.assertTrue(spec.has(caps.HUMANOID_LOCOMOTION))

        aerial = RobotSpec(
            id="typed-air",
            name="Typed Air",
            category="aerial",
            capabilities=(caps.AERIAL, caps.BASE_DRIVE),
            max_lin=2.0,
            max_ang=0.8,
            limits=AerialLimits(max_climb=1.5),
        )
        air = thaw_json(aerial.capability_params)
        self.assertEqual(air[caps.AERIAL], {"frames": [], "max_climb": 1.5})
        self.assertEqual(air[caps.BASE_DRIVE]["max_vx"], 2.0)
        self.assertEqual(air[caps.BASE_DRIVE]["max_wz"], 0.8)

        uneven = RobotSpec(
            id="typed-mm",
            name="Typed MM",
            category="mobile-manipulator",
            capabilities=(caps.BASE_DRIVE, caps.BASE_HOLONOMIC, caps.MANIPULATION),
            max_lin=0.2,
            max_ang=1.0,
            joint_groups={"arm": ("j1",)},
            limits=MobileManipulatorLimits(
                ground=GroundLimits(max_vx=0.2, max_vy=0.15, max_wz=0.8)
            ),
        )
        mm = thaw_json(uneven.capability_params)
        self.assertEqual(mm[caps.BASE_DRIVE]["max_vy"], 0.15)
        self.assertEqual(mm[caps.BASE_HOLONOMIC]["max_wz"], 0.8)
        self.assertEqual(mm[caps.MANIPULATION]["arm_groups"], ["arm"])


if __name__ == "__main__":
    unittest.main()
