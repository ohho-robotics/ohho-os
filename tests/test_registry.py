import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path

from ohho.registry import (
    RobotSpec,
    UnknownRobot,
    get_spec,
    list_specs,
    load_manifest,
    spec_from_dict,
)
from ohho.robot_spec import (
    CI_SCENARIOS,
    GENERIC_CAPABILITIES,
    AerialLimits,
    ArmLimits,
    CiSim,
    GroundLimits,
    HumanoidLimits,
    LeggedLimits,
    MobileManipulatorLimits,
    RobotAssets,
    RobotClass,
    robot_class_for_category,
)

_EXAMPLE = Path(__file__).resolve().parents[1] / "ohho" / "robots" / "example.json"


class TestRegistry(unittest.TestCase):
    def test_builtins_present(self):
        ids = {s.id for s in list_specs()}
        self.assertTrue({"omnibot", "unitree-go2", "sim"}.issubset(ids))

    def test_get_spec(self):
        self.assertEqual(get_spec("omnibot").dof_arm, 6)
        self.assertEqual(len(get_spec("omnibot").joint_names), 6)

    def test_unknown_raises(self):
        with self.assertRaises(UnknownRobot):
            get_spec("does-not-exist")

    def test_spec_from_dict_nested(self):
        spec = spec_from_dict(
            {
                "id": "x",
                "name": "X",
                "category": "wheeled",
                "capabilities": ["base.drive"],
                "dof": {"base": 3, "arm": 2},
                "limits": {"max_lin": 0.7, "max_ang": 1.3},
            }
        )
        self.assertEqual(spec.dof_arm, 2)
        self.assertAlmostEqual(spec.max_lin, 0.7)
        self.assertAlmostEqual(spec.max_ang, 1.3)

    def test_load_manifest_registers(self):
        data = {
            "id": "tmp-bot",
            "name": "Tmp Bot",
            "category": "wheeled",
            "capabilities": ["base.drive"],
            "adapter": "sim",
            "dof": {"base": 3, "arm": 0},
            "limits": {"max_lin": 0.5, "max_ang": 1.0},
        }
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            spec = load_manifest(path)
            self.assertEqual(spec.id, "tmp-bot")
            self.assertEqual(get_spec("tmp-bot").id, "tmp-bot")
        finally:
            os.unlink(path)


def _json_round_trip(spec: RobotSpec) -> RobotSpec:
    return spec_from_dict(json.loads(json.dumps(spec.to_dict())))


class TestRobotSpecV2(unittest.TestCase):
    def test_omnibot_category_maps_to_mobile_manipulator(self):
        spec = get_spec("omnibot")
        self.assertEqual(spec.category, "mobile-manipulator")
        self.assertEqual(spec.robot_class, RobotClass.MOBILE_MANIPULATOR)
        self.assertEqual(robot_class_for_category(spec.category), spec.robot_class)
        self.assertEqual(spec.source, "builtin")
        self.assertEqual((spec.max_lin, spec.max_ang), (0.2, 1.0))
        self.assertEqual(spec.joint_groups["arm"], spec.joint_names)
        self.assertEqual(
            spec.limits,
            MobileManipulatorLimits(
                ground=GroundLimits(max_vx=0.2, max_vy=0.2, max_wz=1.0)
            ),
        )

    def test_go2_category_maps_to_quadruped(self):
        spec = get_spec("unitree-go2")
        self.assertEqual(spec.category, "legged")
        self.assertEqual(spec.robot_class, RobotClass.QUADRUPED)
        self.assertEqual(robot_class_for_category(spec.category), spec.robot_class)
        self.assertEqual(spec.source, "builtin")
        self.assertEqual((spec.max_lin, spec.max_ang), (1.5, 2.0))
        self.assertEqual(spec.joint_groups, {})
        self.assertEqual(
            spec.limits,
            LeggedLimits(max_vx=spec.max_lin, max_vy=spec.max_lin, max_wz=spec.max_ang),
        )

    def test_sim_category_maps_to_mobile_base(self):
        spec = get_spec("sim")
        self.assertEqual(spec.category, "wheeled")
        self.assertEqual(spec.robot_class, RobotClass.MOBILE_BASE)
        self.assertEqual(robot_class_for_category(spec.category), spec.robot_class)
        self.assertEqual(spec.source, "builtin")
        self.assertEqual((spec.max_lin, spec.max_ang), (1.0, 2.0))
        self.assertEqual(spec.capabilities, GENERIC_CAPABILITIES)
        self.assertEqual(
            spec.limits,
            GroundLimits(max_vx=spec.max_lin, max_vy=spec.max_lin, max_wz=spec.max_ang),
        )

    def test_category_table_covers_every_class(self):
        samples = {
            "wheeled": RobotClass.MOBILE_BASE,
            "mobile-base": RobotClass.MOBILE_BASE,
            "mobile_base": RobotClass.MOBILE_BASE,
            "Wheeled": RobotClass.MOBILE_BASE,
            "mobile-manipulator": RobotClass.MOBILE_MANIPULATOR,
            "mobile_manipulator": RobotClass.MOBILE_MANIPULATOR,
            "arm": RobotClass.ARM,
            "manipulator": RobotClass.ARM,
            "legged": RobotClass.QUADRUPED,
            "  Legged ": RobotClass.QUADRUPED,
            "quadruped": RobotClass.QUADRUPED,
            "aerial": RobotClass.AERIAL,
            "drone": RobotClass.AERIAL,
            "humanoid": RobotClass.HUMANOID,
            "biped": RobotClass.HUMANOID,
            "unknown": RobotClass.MOBILE_BASE,
            "hovercraft": RobotClass.MOBILE_BASE,
        }
        for category, expected in samples.items():
            with self.subTest(category=category):
                self.assertEqual(robot_class_for_category(category), expected)
        self.assertEqual(set(CI_SCENARIOS), set(RobotClass))

    def test_new_fields_default(self):
        spec = RobotSpec(id="d", name="D", category="wheeled")
        self.assertEqual(spec.robot_class, RobotClass.MOBILE_BASE)
        self.assertEqual(spec.capabilities, ())
        self.assertEqual(spec.joint_groups, {})
        self.assertEqual(spec.capability_params, {})
        self.assertEqual(spec.limits, GroundLimits(max_vx=0.5, max_vy=0.5, max_wz=1.0))
        self.assertEqual(spec.assets, RobotAssets())
        self.assertEqual(spec.ci_sim, CiSim(enabled=False, scenario=""))
        self.assertEqual(spec.source, "")
        self.assertEqual(spec.sim_backend, "")
        self.assertEqual(spec.max_lin, 0.5)
        self.assertEqual(spec.max_ang, 1.0)
        self.assertEqual(spec.max_climb, 0.0)
        self.assertEqual(spec.gps_fix, "")
        self.assertFalse(spec.requires_hardware_arming)
        self.assertFalse(spec.experimental)
        self.assertFalse(spec.flight_ready)
        hash(spec)

        names = {item.name for item in dataclasses.fields(RobotSpec)}
        self.assertTrue(
            {
                "id",
                "name",
                "category",
                "capabilities",
                "adapter",
                "dof_base",
                "dof_arm",
                "max_lin",
                "max_ang",
                "joint_names",
                "robot_class",
                "joint_groups",
                "limits",
                "assets",
                "ci_sim",
                "source",
                "capability_params",
                "sim_backend",
            }
            <= names
        )
        for flat in (
            "max_climb",
            "gps_fix",
            "requires_hardware_arming",
            "experimental",
            "flight_ready",
        ):
            self.assertNotIn(flat, names)
            self.assertIsInstance(getattr(RobotSpec, flat), property)

    def test_direct_unknown_category_keeps_caller_capabilities(self):
        spec = RobotSpec(id="x", name="X", category="hovercraft", capabilities=())
        self.assertEqual(spec.robot_class, RobotClass.MOBILE_BASE)
        self.assertEqual(spec.category, "hovercraft")
        self.assertEqual(spec.capabilities, ())

    def test_unknown_category_manifest_uses_generic_caps(self):
        omitted = spec_from_dict({"id": "bare", "category": "hovercraft"})
        self.assertEqual(omitted.category, "hovercraft")
        self.assertEqual(omitted.robot_class, RobotClass.MOBILE_BASE)
        self.assertEqual(omitted.capabilities, GENERIC_CAPABILITIES)
        self.assertIsInstance(omitted.limits, GroundLimits)

        missing = spec_from_dict({"id": "bare-2"})
        self.assertEqual(missing.category, "unknown")
        self.assertEqual(missing.robot_class, RobotClass.MOBILE_BASE)
        self.assertEqual(missing.capabilities, GENERIC_CAPABILITIES)

        explicit = spec_from_dict(
            {
                "id": "kept",
                "category": "hovercraft",
                "capabilities": ["manipulation"],
            }
        )
        self.assertEqual(explicit.robot_class, RobotClass.MOBILE_BASE)
        self.assertEqual(explicit.capabilities, ("manipulation",))

        named = spec_from_dict(
            {"id": "arm-ish", "category": "hovercraft", "robot_class": "arm"}
        )
        self.assertEqual(named.robot_class, RobotClass.ARM)
        self.assertEqual(named.capabilities, ())
        self.assertIsInstance(named.limits, ArmLimits)

    def test_known_category_without_capabilities_stays_empty(self):
        spec = spec_from_dict({"id": "rover", "category": "wheeled"})
        self.assertEqual(spec.robot_class, RobotClass.MOBILE_BASE)
        self.assertEqual(spec.capabilities, ())
        self.assertEqual(spec.limits, GroundLimits(max_vx=0.5, max_vy=0.5, max_wz=1.0))

    def test_old_nested_limits_still_set_flat_clamps(self):
        spec = spec_from_dict(
            {
                "id": "x",
                "name": "X",
                "category": "wheeled",
                "capabilities": ["base.drive"],
                "dof": {"base": 3, "arm": 2},
                "limits": {"max_lin": 0.7, "max_ang": 1.3},
            }
        )
        self.assertEqual(spec.dof_arm, 2)
        self.assertAlmostEqual(spec.max_lin, 0.7)
        self.assertAlmostEqual(spec.max_ang, 1.3)
        self.assertEqual(spec.robot_class, RobotClass.MOBILE_BASE)
        self.assertEqual(spec.limits, GroundLimits(max_vx=0.7, max_vy=0.7, max_wz=1.3))

    def test_example_manifest_still_loads(self):
        spec = load_manifest(_EXAMPLE, register_it=False)
        self.assertEqual(spec.id, "my-rover")
        self.assertEqual(spec.category, "wheeled")
        self.assertEqual(spec.robot_class, RobotClass.MOBILE_BASE)
        self.assertEqual(spec.capabilities, ("base.drive", "perception.rgb"))
        self.assertAlmostEqual(spec.max_lin, 0.8)
        self.assertAlmostEqual(spec.max_ang, 1.5)
        self.assertEqual(spec.joint_names, ())
        self.assertEqual(_json_round_trip(spec), spec)

    def test_serialisation_round_trip(self):
        cases = [
            RobotSpec(
                id="base", name="Base", category="wheeled", max_lin=0.4, max_ang=0.8
            ),
            RobotSpec(
                id="mm",
                name="MM",
                category="mobile-manipulator",
                robot_class=RobotClass.MOBILE_MANIPULATOR,
                joint_groups={"arm": ("shoulder", "elbow")},
                limits=MobileManipulatorLimits(
                    ground=GroundLimits(max_vx=0.2, max_vy=0.2, max_wz=1.0),
                    arm=ArmLimits(max_joint_step=0.25, max_tau=3.0),
                ),
                max_lin=0.2,
                max_ang=1.0,
            ),
            RobotSpec(
                id="arm",
                name="Arm",
                category="arm",
                limits=ArmLimits(max_joint_step=0.3, max_tau=8.0),
            ),
            RobotSpec(
                id="quad",
                name="Quad",
                category="legged",
                max_lin=1.5,
                max_ang=2.0,
            ),
            RobotSpec(
                id="air",
                name="Air",
                category="aerial",
                max_lin=2.0,
                max_ang=0.8,
                limits=AerialLimits(
                    max_climb=1.0,
                    gps_fix="3d",
                    requires_hardware_arming=True,
                    experimental=True,
                    flight_ready=True,
                ),
            ),
            RobotSpec(
                id="human",
                name="Human",
                category="humanoid",
                max_lin=0.4,
                max_ang=0.6,
                joint_names=("l_hip", "l_knee"),
                joint_groups={"left_leg": ("l_hip", "l_knee"), "right_leg": ("r_hip",)},
                limits=HumanoidLimits(
                    max_vx=0.4,
                    max_vy=0.2,
                    max_wz=0.6,
                    max_joint_step=0.1,
                    max_kp=20.0,
                    max_tau=40.0,
                ),
                assets=RobotAssets(
                    urdf="robot.urdf",
                    mjcf="robot.xml",
                    meshes=("link.stl",),
                    licence="Apache-2.0",
                ),
                ci_sim=CiSim(enabled=True, scenario="humanoid_walk"),
                source="example-plugin",
                sim_backend="humanoid-mujoco",
                capability_params={
                    "humanoid.locomotion": {"gaits": ["walk", "turn"], "speed_max": 0.4}
                },
            ),
        ]
        for spec in cases:
            with self.subTest(spec=spec.id):
                again = _json_round_trip(spec)
                self.assertEqual(again, spec)
                self.assertEqual(again.robot_class, spec.robot_class)
                self.assertEqual(type(again.limits), type(spec.limits))

        air = cases[4]
        self.assertEqual(air.max_climb, 1.0)
        self.assertEqual(air.gps_fix, "3d")
        self.assertTrue(air.requires_hardware_arming)
        self.assertTrue(air.experimental)
        self.assertTrue(air.flight_ready)
        self.assertTrue(_json_round_trip(air).flight_ready)

    def test_flat_aerial_manifest_fields_stay_readable(self):
        spec = spec_from_dict(
            {
                "id": "aerial-sim",
                "name": "Aerial sim",
                "category": "aerial",
                "capabilities": ["base.drive"],
                "adapter": "sim",
                "limits": {"max_lin": 2.0, "max_ang": 0.8, "max_climb": 1.0},
                "gps_fix": "none",
                "requires_hardware_arming": True,
                "experimental": True,
                "flight_ready": True,
            }
        )
        self.assertEqual(spec.robot_class, RobotClass.AERIAL)
        self.assertIsInstance(spec.limits, AerialLimits)
        self.assertEqual(spec.max_lin, 2.0)
        self.assertEqual(spec.max_ang, 0.8)
        self.assertEqual(spec.max_climb, 1.0)
        self.assertEqual(spec.gps_fix, "none")
        self.assertTrue(spec.requires_hardware_arming)
        self.assertTrue(spec.experimental)
        self.assertFalse(spec.flight_ready)
        self.assertFalse(spec.limits.flight_ready)

        fuzzy = spec_from_dict(
            {"id": "fuzzy", "category": "aerial", "experimental": "true"}
        )
        self.assertFalse(fuzzy.experimental)
        self.assertEqual(fuzzy.max_climb, 0.0)

    def test_limits_kind_must_match_class(self):
        with self.assertRaises(ValueError):
            spec_from_dict(
                {
                    "id": "x",
                    "category": "wheeled",
                    "limits": {"kind": "aerial", "max_climb": 1.0},
                }
            )
        with self.assertRaises(ValueError):
            spec_from_dict({"id": "x", "robot_class": "spider"})
        with self.assertRaises(TypeError):
            RobotSpec(id="x", name="X", category="wheeled", limits=AerialLimits())

    def test_joint_groups_are_copied(self):
        groups = {"arm": ["shoulder", "elbow"]}
        spec = RobotSpec(
            id="g",
            name="G",
            category="arm",
            joint_groups=groups,
        )
        groups["arm"].append("wrist")
        self.assertEqual(spec.joint_groups["arm"], ("shoulder", "elbow"))
        self.assertIsInstance(spec.limits, ArmLimits)


if __name__ == "__main__":
    unittest.main()
