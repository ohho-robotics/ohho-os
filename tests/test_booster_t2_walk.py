"""Headless booster-t2 walk (OHH-331).

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
    """Load the booster plugin tests without the ``tests`` package name.

    A local ``tests`` distribution can shadow this checkout. Loading the file
    keeps the shared discovery helpers available in that case.
    """
    path = Path(__file__).with_name("test_ohho_booster.py")
    spec = importlib.util.spec_from_file_location("ohh331_booster_helpers", path)
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
    / "t2-walk.md"
)


class TestWalkLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _helpers()._ensure_plugin_path()
        from ohho_booster.sim import (
            ACTION_SCALE,
            BASE_BODY,
            BASE_POS,
            DAMPING,
            DEFAULT_Q,
            HISTORY_LENGTH,
            PHYSICS_DT_S,
            POLICY_RATE_HZ,
            STIFFNESS,
            TRAINING_COMMAND,
            WALK_POLICY_JOINTS,
            HeadingObservation,
            observation_terms,
            walk_policy_layout,
            walk_sim_config,
        )
        from ohho_booster.spec import booster_t2_spec

        cls.action_scale = ACTION_SCALE
        cls.base_body = BASE_BODY
        cls.base_pos = BASE_POS
        cls.damping = DAMPING
        cls.default_q = DEFAULT_Q
        cls.history_length = HISTORY_LENGTH
        cls.physics_dt = PHYSICS_DT_S
        cls.rate_hz = POLICY_RATE_HZ
        cls.stiffness = STIFFNESS
        cls.training = TRAINING_COMMAND
        cls.policy_joints = WALK_POLICY_JOINTS
        cls.heading = HeadingObservation
        cls.terms = staticmethod(observation_terms)
        cls.layout = staticmethod(walk_policy_layout)
        cls.config = staticmethod(walk_sim_config)
        cls.spec = booster_t2_spec()

    def test_frame_is_eighty_values_stacked_ten_times(self):
        from ohho.humanoid_sim import observation_size

        frame = observation_size(len(self.policy_joints), self.terms()) + 2
        self.assertEqual(frame, 80)
        self.assertEqual(frame * self.history_length, 800)
        self.assertEqual(len(self.policy_joints), 23)
        self.assertEqual(len(self.action_scale), 23)
        self.assertEqual(len(self.stiffness), 31)
        self.assertEqual(len(self.damping), 31)
        self.assertEqual(len(self.default_q), 31)
        self.assertAlmostEqual(self.action_scale[0], 0.2608)
        self.assertAlmostEqual(self.action_scale[-1], 0.0950)

    def test_layout_matches_the_deployment_controller(self):
        layout = self.layout("unused.pt")
        self.assertTrue(layout.q_relative)
        self.assertEqual(layout.history_length, 10)
        self.assertEqual(layout.history_init, "repeat")
        self.assertEqual(layout.clip_observation, 100.0)
        self.assertEqual(layout.clip_action, 10.0)
        self.assertEqual(layout.rate_hz, self.rate_hz)
        self.assertEqual(layout.rate_hz, 50.0)
        config = self.config(mjcf="model.xml", checkpoint="unused.pt")
        self.assertAlmostEqual(config.timestep, self.physics_dt)
        self.assertAlmostEqual(config.timestep, 0.002)
        self.assertEqual(config.base_pos, self.base_pos)
        self.assertEqual(config.base_body, self.base_body)
        self.assertEqual(config.base_body, "trunk")
        self.assertTrue(config.mjcf.endswith("model.xml"))

    def test_training_range_is_not_written_into_the_spec(self):
        self.assertEqual(self.spec.sim_backend, "humanoid-mujoco")
        self.assertGreater(self.training["vx_forward_max"], self.spec.max_lin)
        self.assertEqual(self.training["vx_forward_max"], 3.0)
        self.assertEqual(self.training["vx_backward_max"], 0.7)
        self.assertLessEqual(self.spec.max_lin, 0.5)
        self.assertGreaterEqual(self.spec.max_lin, 0.3)
        self.assertLessEqual(self.spec.max_ang, 1.0)

    def test_heading_resets_integrates_and_clamps(self):
        extra = self.heading()
        error, flag = extra.extend(yaw=0.2, command=(0.0, 0.0, 0.0), dt=0.02)
        self.assertAlmostEqual(error, 0.0)
        self.assertEqual(flag, 1.0)
        error, flag = extra.extend(yaw=0.0, command=(0.3, 0.0, 0.5), dt=0.02)
        self.assertAlmostEqual(error, 0.0)
        self.assertEqual(flag, 0.0)
        error, flag = extra.extend(yaw=0.0, command=(0.3, 0.0, 0.5), dt=0.02)
        self.assertAlmostEqual(error, 0.01, places=6)
        self.assertEqual(flag, 0.0)
        error, flag = extra.extend(yaw=2.0, command=(0.3, 0.0, 0.5), dt=0.02)
        self.assertAlmostEqual(error, -0.5)
        self.assertEqual(flag, 0.0)
        extra.reset()
        error, flag = extra.extend(yaw=1.0, command=(0.0, 0.0, 0.0), dt=0.02)
        self.assertAlmostEqual(error, 0.0)
        self.assertEqual(flag, 1.0)

    def test_prepare_sim_ignores_other_robots(self):
        from unittest.mock import patch

        from ohho.registry import RobotSpec
        from ohho_booster.plugin import BoosterPlugin

        other = RobotSpec(id="sim", name="Sim", category="mobile")
        with patch("ohho_booster.sim.bind_walk_config") as bound:
            BoosterPlugin().prepare_sim(other)
            bound.assert_not_called()

    def test_docs_record_the_model_choice_and_the_training_range(self):
        text = _DOCS.read_text(encoding="utf-8")
        for needle in (
            "T2_31dof.xml",
            "T2_31dof_parallel.xml",
            "serial",
            "3.0",
            "0.7",
            "0.8",
            "2.0",
            "SafetyGate",
        ):
            self.assertIn(needle, text)


@unittest.skipUnless(_HAS_MUJOCO, "mujoco is not installed")
@unittest.skipUnless(_HAS_TORCH, "torch is not installed")
class TestBoosterT2Walk(unittest.TestCase):
    def test_stands_walks_turns_and_stops(self):
        helpers = _helpers()
        helpers._ensure_plugin_path()
        from ohho_booster.assets import ensure_assets

        try:
            ensure_assets()
        except (OSError, urllib.error.URLError) as exc:
            if os.environ.get("GITHUB_ACTIONS") == "true":
                raise
            self.skipTest(f"pinned assets absent: {exc}")

        from ohho.adapters import resolve_transport
        from ohho.registry import get_spec
        from ohho_booster import plugin as plugin_factory
        from ohho_booster.walk import run_walk_check

        booster = helpers._Entry("booster", "ohho_booster:plugin", plugin_factory)

        def reader():
            return [helpers._builtin_entry(), booster]

        with helpers._discovered(reader):
            spec = get_spec("booster-t2")
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
        self.assertTrue(result.stayed_up)
        self.assertGreater(result.min_height_m, 0.9)
        self.assertGreater(result.stand_height_m, 0.9)
        self.assertGreater(result.walked_m, 0.9)
        self.assertLess(result.walked_m, 1.2)
        self.assertGreater(abs(result.yaw_change_rad), 0.45)
        self.assertLess(result.stop_speed_m_s, 0.05)
