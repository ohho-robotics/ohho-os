"""Headless booster-k1 walk.

Layout checks always run. The MuJoCo check needs mujoco, torch and the
pinned description archive plus checkpoint. It skips when those are absent
on a developer machine. In GitHub Actions a missing pin is a failure, because
the Ubuntu 3.12 job fetches it before the suite.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
import urllib.error
from pathlib import Path


def _helpers():
    """Load the booster plugin tests without the ``tests`` package name."""
    path = Path(__file__).with_name("test_ohho_booster.py")
    spec = importlib.util.spec_from_file_location("ohh_k1_booster_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_DOCS = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "ohho-booster"
    / "docs"
    / "k1-walk.md"
)
_README = Path(__file__).resolve().parents[1] / "plugins" / "ohho-booster" / "README.md"


class TestK1WalkLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _helpers()._ensure_plugin_path()
        from ohho_booster.joints import (
            K1_JOINT_GROUPS,
            K1_JOINT_NAMES,
            K1_SIM_JOINT_NAMES,
        )
        from ohho_booster.sim import (
            K1_ACTION_FILTER,
            K1_ACTION_SCALE,
            K1_BASE_POS,
            K1_CLIP_ACTION,
            K1_DEFAULT_Q,
            K1_DQ_SCALE,
            K1_EFFORT_LIMIT,
            K1_HISTORY_LENGTH,
            K1_TRAINING_COMMAND,
            K1_WALK_POLICY_JOINTS,
            k1_action_offset,
            k1_policy_layout,
            k1_sim_config,
            observation_terms,
        )
        from ohho_booster.spec import booster_k1_spec

        cls.joints = K1_JOINT_NAMES
        cls.sim_joints = K1_SIM_JOINT_NAMES
        cls.groups = K1_JOINT_GROUPS
        cls.filter = K1_ACTION_FILTER
        cls.scale = K1_ACTION_SCALE
        cls.base_pos = K1_BASE_POS
        cls.clip_action = K1_CLIP_ACTION
        cls.default_q = K1_DEFAULT_Q
        cls.dq_scale = K1_DQ_SCALE
        cls.effort = K1_EFFORT_LIMIT
        cls.history = K1_HISTORY_LENGTH
        cls.training = K1_TRAINING_COMMAND
        cls.policy_joints = K1_WALK_POLICY_JOINTS
        cls.offset = staticmethod(k1_action_offset)
        cls.layout = staticmethod(k1_policy_layout)
        cls.config = staticmethod(k1_sim_config)
        cls.terms = staticmethod(observation_terms)
        cls.spec = booster_k1_spec()

    def test_frame_is_sixty_nine_values_stacked_ten_times(self):
        from ohho.humanoid_sim import observation_size

        terms = self.terms(self.dq_scale)
        frame = observation_size(len(self.policy_joints), terms)
        self.assertEqual(len(self.joints), 22)
        self.assertEqual(len(self.policy_joints), 20)
        self.assertNotIn("waist", self.groups)
        self.assertEqual(frame, 69)
        self.assertEqual(frame * self.history, 690)
        dq = next(term for term in terms if term.name == "dq")
        self.assertAlmostEqual(dq.scale, 0.1)
        default_dq = next(term for term in self.terms() if term.name == "dq")
        self.assertAlmostEqual(default_dq.scale, 1.0)

    def test_policy_order_is_the_deployment_order(self):
        self.assertEqual(
            self.policy_joints[6:10],
            (
                "left_hip_roll_joint",
                "right_hip_roll_joint",
                "left_elbow_pitch_joint",
                "right_elbow_pitch_joint",
            ),
        )
        self.assertEqual(self.policy_joints.index("right_elbow_pitch_joint"), 9)
        self.assertEqual(sorted(self.sim_joints), sorted(self.joints))
        self.assertNotEqual(self.sim_joints, self.joints)
        self.assertNotIn("left_shoulder_pitch_joint", self.joints)

    def test_layout_matches_the_deployment_controller(self):
        layout = self.layout("unused.pt")
        self.assertTrue(layout.q_relative)
        self.assertTrue(layout.actor_module)
        self.assertFalse(layout.seed_last_action)
        self.assertIsNone(layout.extra_factory)
        self.assertEqual(layout.history_length, 10)
        self.assertEqual(layout.history_init, "repeat")
        self.assertEqual(layout.clip_observation, 100.0)
        self.assertEqual(layout.clip_action, self.clip_action)
        self.assertEqual(layout.clip_action, 100.0)
        self.assertAlmostEqual(layout.action_scale, self.scale)
        self.assertAlmostEqual(layout.action_scale, 0.25)
        self.assertAlmostEqual(layout.action_filter, self.filter)
        self.assertAlmostEqual(layout.action_filter, 0.8)
        offset = self.offset()
        self.assertEqual(len(offset), 20)
        self.assertAlmostEqual(offset[9], 0.2)
        self.assertEqual(sum(1 for value in offset if value), 1)
        config = self.config(mjcf="model.xml", checkpoint="unused.pt")
        self.assertEqual(config.base_pos, self.base_pos)
        self.assertEqual(config.base_pos[2], 0.6)
        self.assertEqual(config.base_body, "trunk")
        self.assertEqual(config.effort_limit, self.effort)
        self.assertEqual(len(self.default_q), 22)

    def test_training_range_is_not_written_into_the_spec(self):
        from ohho import capabilities as caps

        self.assertEqual(self.spec.id, "booster-k1")
        self.assertEqual(self.spec.sim_backend, "humanoid-mujoco")
        self.assertEqual(self.spec.dof_arm, 8)
        self.assertEqual(self.spec.dof_base, 3)
        self.assertNotIn(caps.MANIPULATION, self.spec.capabilities)
        self.assertNotIn(caps.MANIPULATION_DUAL_ARM, self.spec.capabilities)
        modes = self.spec.capability_params[caps.HUMANOID_POSTURE]["modes"]
        self.assertNotIn("soccer", modes)
        self.assertEqual(self.spec.ci_sim.scenario, "humanoid_walk")
        self.assertTrue(self.spec.ci_sim.enabled)
        self.assertEqual(self.training["vx_forward_max"], 1.6)
        self.assertEqual(self.training["vx_backward_max"], 0.3)
        self.assertEqual(self.training["vy_max"], 0.3)
        self.assertEqual(self.training["vyaw_max"], 1.8)
        self.assertGreater(self.training["vx_forward_max"], self.spec.max_lin)
        self.assertLessEqual(self.spec.max_lin, 0.5)
        self.assertLessEqual(self.spec.max_ang, 1.0)

    def test_docs_mark_sim_only_and_name_the_serial_model(self):
        text = _DOCS.read_text(encoding="utf-8")
        readme = _README.read_text(encoding="utf-8")
        for needle in (
            "K1_22dof.xml",
            "K1_22dof_parallel.xml",
            "serial",
            "1.6",
            "0.3",
            "1.8",
            "690",
            "0.1",
            "0.8",
            "0.2",
            "Isaac Lab",
            "sim only, not hardware-tested",
        ):
            self.assertIn(needle, text)
        self.assertIn("booster-k1", readme)
        self.assertIn("sim only, not hardware-tested", readme)


@unittest.skipUnless(_HAS_MUJOCO, "mujoco is not installed")
@unittest.skipUnless(_HAS_TORCH, "torch is not installed")
class TestBoosterK1Walk(unittest.TestCase):
    def test_stands_walks_turns_and_stops(self):
        helpers = _helpers()
        helpers._ensure_plugin_path()
        from ohho_booster.assets import ensure_assets

        try:
            ensure_assets(robot_id="booster-k1")
        except (OSError, urllib.error.URLError, ValueError) as exc:
            if os.environ.get("GITHUB_ACTIONS") == "true":
                raise
            self.skipTest(f"pinned assets absent: {exc}")

        from ohho.adapters import resolve_transport
        from ohho.registry import get_spec
        from ohho_booster import plugin as plugin_factory
        from ohho_booster.walk import run_walk_check, walk_limits, within_limits

        booster = helpers._Entry("booster", "ohho_booster:plugin", plugin_factory)

        def reader():
            return [helpers._builtin_entry(), booster]

        with helpers._discovered(reader):
            spec = get_spec("booster-k1")
            self.assertIsNotNone(spec)
            assert spec is not None
            transport = resolve_transport("sim://", spec)
            transport.connect()
            try:
                result = run_walk_check(transport)
                steps = int(getattr(transport, "pd_steps"))
            finally:
                transport.disconnect()

        # 30 s at the 0.002 s physics step, before the walk, turn and stop.
        self.assertGreaterEqual(steps, 15000)
        self.assertTrue(within_limits(result, walk_limits("booster-k1")))
        self.assertGreater(result.min_height_m, 0.45)
        self.assertGreater(result.stand_height_m, 0.45)
        self.assertGreater(result.walked_m, 0.9)
        self.assertLess(result.walked_m, 1.2)
        self.assertGreater(abs(result.yaw_change_rad), 0.45)
        self.assertLess(result.stop_speed_m_s, 0.05)
