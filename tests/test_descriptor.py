"""Round-trip the OmniBot RobotDescriptor through the registry and safety profile."""

import json
import os
import tempfile
import unittest

from ohho import capabilities as caps
from ohho.descriptor import (
    MAX_JOINT_STEP_RAD,
    MAX_VX_MPS,
    load_descriptor,
    robot_descriptor,
    spec_from_descriptor,
    validate_descriptor,
)
from ohho.profiles import emit_robot_descriptor, load_robot_descriptor
from ohho.registry import get_spec
from ohho.safety import OMNIBOT_PROFILE


class TestRobotDescriptor(unittest.TestCase):
    def test_ceilings_stay_at_omnibot_values(self):
        self.assertEqual(MAX_VX_MPS, 0.2)
        self.assertEqual(MAX_JOINT_STEP_RAD, 0.15)
        self.assertLessEqual(MAX_VX_MPS, OMNIBOT_PROFILE.max_vx)
        self.assertLessEqual(MAX_JOINT_STEP_RAD, OMNIBOT_PROFILE.max_joint_step)

    def test_omnibot_round_trip(self):
        doc = robot_descriptor("omnibot")
        text = emit_robot_descriptor("omnibot")
        self.assertEqual(json.loads(text), doc)

        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            loaded = load_descriptor(path)
            self.assertEqual(load_robot_descriptor(path), loaded)
        finally:
            os.unlink(path)

        self.assertEqual(loaded, doc)
        self.assertEqual(doc["id"], "omnibot")
        self.assertEqual(doc["model"], "OmniBot")
        self.assertEqual(doc["drive"]["kind"], "mecanum")
        self.assertTrue(doc["drive"]["holonomic"])
        self.assertEqual(doc["apriltags"], [])
        self.assertEqual(doc["topics"]["cmd"], "/cmd_vel")
        self.assertEqual(doc["topics"]["estop"], "/emergency_stop")
        self.assertEqual(doc["telemetry"]["schema_id"], "ohho.telemetry.v1")
        self.assertEqual(doc["telemetry"]["fields"], ["base", "joints", "imu"])

        hard = doc["safety"]["hard"]
        self.assertEqual(hard["max_vx_mps"], 0.2)
        self.assertEqual(hard["max_vy_mps"], 0.2)
        self.assertLessEqual(hard["max_vx_mps"], OMNIBOT_PROFILE.max_vx)
        self.assertLessEqual(hard["max_vy_mps"], OMNIBOT_PROFILE.max_vy)
        self.assertLessEqual(hard["max_wz_rad_s"], OMNIBOT_PROFILE.max_wz)
        self.assertLessEqual(hard["max_accel_x_mps2"], OMNIBOT_PROFILE.max_accel_x)
        self.assertLessEqual(hard["max_accel_y_mps2"], OMNIBOT_PROFILE.max_accel_y)
        self.assertLessEqual(hard["max_accel_wz_rad_s2"], OMNIBOT_PROFILE.max_accel_wz)
        self.assertEqual(hard["max_joint_step_rad"], 0.15)
        self.assertLessEqual(hard["max_joint_step_rad"], OMNIBOT_PROFILE.max_joint_step)
        self.assertEqual(doc["drive"]["caps"]["vx_mps"], hard["max_vx_mps"])

        joints = doc["arms"][0]["joints"]
        self.assertEqual(tuple(joint["name"] for joint in joints), get_spec("omnibot").joint_names)
        for joint in joints:
            lo, hi = OMNIBOT_PROFILE.joint_limits[joint["name"]]
            self.assertEqual(joint["min"], lo)
            self.assertEqual(joint["max"], hi)
            self.assertEqual(joint["home"], 0.0)
        grip = joints[-1]
        self.assertEqual(grip["name"], "arm_gripper")
        self.assertEqual(grip["max"], 0.8)
        self.assertEqual(grip["min"], -0.1)

        spec = spec_from_descriptor(doc)
        self.assertEqual(spec.id, "omnibot")
        self.assertEqual(spec.joint_names, get_spec("omnibot").joint_names)
        self.assertEqual(spec.max_lin, 0.2)
        self.assertLessEqual(spec.max_lin, get_spec("omnibot").max_lin)
        self.assertEqual(spec.max_ang, hard["max_wz_rad_s"])
        self.assertTrue(spec.has(caps.BASE_DRIVE))
        self.assertTrue(spec.has(caps.BASE_HOLONOMIC))
        self.assertTrue(spec.has(caps.MANIPULATION))
        self.assertTrue(spec.has(caps.PERCEPTION_RGB))
        self.assertTrue(spec.has(caps.PERCEPTION_DEPTH))
        self.assertFalse(spec.has(caps.PERCEPTION_SCAN))
        self.assertTrue(get_spec("omnibot").has(caps.PERCEPTION_SCAN))

    def test_sim_kind_does_not_change_caps(self):
        real = robot_descriptor("omnibot")
        sim = robot_descriptor("omnibot", kind="sim")
        self.assertEqual(sim["telemetry"]["kind"], "sim")
        self.assertEqual(sim["safety"], real["safety"])
        self.assertEqual(sim["drive"], real["drive"])

    def test_raised_linear_cap_is_refused(self):
        doc = robot_descriptor("omnibot")
        doc["safety"]["hard"]["max_vx_mps"] = 0.21
        with self.assertRaises(ValueError):
            validate_descriptor(doc)

    def test_unknown_robot_is_refused(self):
        with self.assertRaises(KeyError):
            robot_descriptor("unitree-go2")


if __name__ == "__main__":
    unittest.main()
