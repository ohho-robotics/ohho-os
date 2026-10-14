"""Unit tests for Go2 episode logging: LeRobot v2.0 schema, SafetyGate clamping,

video encoding, and 10-episode sim teleop via sim-serve (OHH-129).
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from ohho.adapters.sim import SimTransport
from ohho.data import DatasetReader, Recorder, record_go2_teleop_episodes
from ohho.data.recorder import (
    GO2_ACTION_NAMES,
    GO2_STATE_NAMES,
)
from ohho.data.teleop import make_sim_camera_frame
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime
from ohho.safety import SafetyGate


def _has_numpy() -> bool:
    return importlib.util.find_spec("numpy") is not None


def _has_websockets() -> bool:
    return importlib.util.find_spec("websockets") is not None


def _has_pyarrow() -> bool:
    return importlib.util.find_spec("pyarrow") is not None


def _has_lerobot() -> bool:
    return importlib.util.find_spec("lerobot") is not None


def _has_video_encode() -> bool:
    return (
        importlib.util.find_spec("av") is not None
        or importlib.util.find_spec("imageio") is not None
    )


_GO2_VIDEO_SKIP = "numpy + av or imageio required — install [data] and `pip install av`"
_GO2_TELEOP_SKIP = (
    "websockets, numpy, pyarrow, av|imageio, lerobot required — "
    "[sim-serve], [data], `pip install av`, [arm] extras"
)


def _go2_bot():
    spec = get_spec("unitree-go2")
    tp = SimTransport(spec)
    tp.connect()
    return Robot(spec, tp, NativeRuntime())


def _go2_gated_bot():
    spec = get_spec("unitree-go2")
    inner = SimTransport(spec)
    inner.connect()
    gate = SafetyGate(inner, spec, auto_tick=False)
    gate.connect()
    bot = Robot(spec, gate, NativeRuntime())
    bot.arm()
    return bot


class TestGo2Schema(unittest.TestCase):
    def test_go2_state_and_action_dimensions(self):
        bot = _go2_bot()
        rec = Recorder(bot, repo_id="ohho-robotics/go2-test", fps=10.0)
        rec.start_episode(task="walk to the red box")

        # Set walk mode
        rec.set_mode("walk")
        bot.drive(vx=0.4, vy=0.0, w=0.1)
        bot.transport.step(0.1)  # type: ignore[attr-defined]

        frame = rec.capture_frame()
        # 31-D observation.state
        self.assertEqual(len(frame.observation_state), 31)
        # 4-D action
        self.assertEqual(len(frame.action), 4)
        # 4-D action_raw
        self.assertIsNotNone(frame.action_raw)
        self.assertEqual(len(frame.action_raw), 4)

        # Base velocity in state
        self.assertAlmostEqual(frame.observation_state[0], 0.4, places=2)
        # Mode ID in action
        self.assertAlmostEqual(frame.action[3], 2.0)  # walk mode = 2.0
        rec.stop_episode()

    def test_go2_named_features_in_info_json(self):
        bot = _go2_bot()
        rec = Recorder(bot, repo_id="ohho-robotics/go2-features-test", fps=10.0)
        rec.start_episode(task="test schema")
        bot.drive(vx=0.2, w=0.1)
        rec.capture_frame()
        rec.stop_episode()

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = rec.save(tmp)
            info_file = Path(dataset_path) / "meta" / "info.json"
            self.assertTrue(info_file.exists())
            with open(info_file, encoding="utf-8") as f:
                info = json.load(f)

            self.assertEqual(info["codebase_version"], "2.0")
            self.assertEqual(info["robot_type"], "unitree-go2")
            self.assertIn("observation.state", info["features"])
            self.assertIn("action", info["features"])
            self.assertIn("action_raw", info["features"])

            state_feat = info["features"]["observation.state"]
            self.assertEqual(state_feat["shape"], [31])
            self.assertEqual(state_feat["names"], GO2_STATE_NAMES)

            action_feat = info["features"]["action"]
            self.assertEqual(action_feat["shape"], [4])
            self.assertEqual(action_feat["names"], GO2_ACTION_NAMES)

            raw_feat = info["features"]["action_raw"]
            self.assertEqual(raw_feat["shape"], [4])
            self.assertEqual(raw_feat["names"], GO2_ACTION_NAMES)

    def test_safety_gate_clamped_command_vs_raw_request(self):
        bot = _go2_gated_bot()
        rec = Recorder(bot, repo_id="ohho-robotics/go2-clamp-test", fps=10.0)
        rec.start_episode(task="clamp check")

        # Request velocity exceeding Go2 limits (max_lin = 1.5, max_ang = 2.0)
        # Request vx = 10.0 (way over 1.5)
        bot.drive(vx=10.0, vy=5.0, w=4.0)
        frame = rec.capture_frame()

        # Raw requested action preserves 10.0
        self.assertAlmostEqual(frame.action_raw[0], 10.0)
        self.assertAlmostEqual(frame.action_raw[1], 5.0)
        self.assertAlmostEqual(frame.action_raw[2], 4.0)

        # Clamped action reflects what actually reached the motors (capped by SafetyGate)
        self.assertLessEqual(frame.action[0], 1.5)
        self.assertLessEqual(frame.action[1], 1.5)
        self.assertLessEqual(frame.action[2], 2.0)
        rec.stop_episode()

        # Verify saved dataset contains both
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = rec.save(tmp)
            reader = DatasetReader(dataset_path)
            loaded_frames = reader.load_episode(0)
            self.assertEqual(len(loaded_frames), 1)
            self.assertAlmostEqual(loaded_frames[0].action_raw[0], 10.0)
            self.assertLessEqual(loaded_frames[0].action[0], 1.5)

    @unittest.skipUnless(
        _has_numpy() and _has_video_encode(),
        _GO2_VIDEO_SKIP,
    )
    def test_front_camera_video_mp4_encoding(self):
        bot = _go2_bot()
        step_count = 0

        def fake_cam():
            nonlocal step_count
            step_count += 1
            return make_sim_camera_frame(0.1 * step_count, 0.0, width=64, height=48)

        rec = Recorder(
            bot,
            repo_id="ohho-robotics/go2-video-test",
            fps=10.0,
            camera_fn=fake_cam,
        )
        rec.start_episode(task="camera test")
        for _ in range(5):
            bot.drive(vx=0.2)
            bot.transport.step(0.1)  # type: ignore[attr-defined]
            rec.capture_frame()
        rec.stop_episode()

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = rec.save(tmp)
            p = Path(dataset_path)

            # Check video directory and MP4 file
            vid_dir = p / "videos" / "chunk-000" / "observation.images.front"
            self.assertTrue(vid_dir.exists())
            mp4_file = vid_dir / "episode_000000.mp4"
            self.assertTrue(mp4_file.exists())
            self.assertGreater(mp4_file.stat().st_size, 0)

            # Check info.json
            with open(p / "meta" / "info.json", encoding="utf-8") as f:
                info = json.load(f)
            self.assertEqual(info["total_videos"], 1)
            self.assertIn("observation.images.front", info["features"])
            vid_feat = info["features"]["observation.images.front"]
            self.assertEqual(vid_feat["dtype"], "video")
            self.assertEqual(vid_feat["shape"], [3, 48, 64])

            # Check reader
            reader = DatasetReader(dataset_path)
            self.assertTrue(reader.has_videos)
            self.assertIn("observation.images.front", reader.video_keys)
            self.assertEqual(reader.get_video_path(0), mp4_file)

    def test_dataset_reader_stats_named_features(self):
        bot = _go2_bot()
        rec = Recorder(bot, repo_id="ohho-robotics/go2-stats-test", fps=10.0)
        rec.start_episode(task="stats test")
        rec.set_mode("walk")
        bot.drive(vx=0.3, vy=0.05, w=0.1)
        rec.capture_frame()
        bot.drive(vx=0.5, vy=0.10, w=0.2)
        rec.capture_frame()
        rec.stop_episode()

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = rec.save(tmp)
            reader = DatasetReader(dataset_path)
            stats = reader.stats()

            self.assertIn("observation.state", stats)
            self.assertIn("action", stats)
            self.assertIn("action_raw", stats)

            # Check positional keys
            self.assertIn("dim0_min", stats["action"])
            self.assertIn("dim0_max", stats["action"])

            # Check named keys for Go2
            self.assertIn("base_vx_min", stats["action"])
            self.assertIn("base_vx_max", stats["action"])
            self.assertIn("mode_id_mean", stats["action"])
            self.assertAlmostEqual(stats["action"]["mode_id_mean"], 2.0)

            self.assertIn("roll_min", stats["observation.state"])
            self.assertIn("pitch_max", stats["observation.state"])
            self.assertIn("posture_id_min", stats["observation.state"])


class TestGo2SimTeleop10Episodes(unittest.TestCase):
    """OHH-129 Acceptance test: 10 teleop episodes in sim via sim-serve."""

    @unittest.skipUnless(
        _has_websockets()
        and _has_numpy()
        and _has_pyarrow()
        and _has_video_encode()
        and _has_lerobot(),
        _GO2_TELEOP_SKIP,
    )
    def test_record_10_teleop_episodes_and_lerobot_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, stats = record_go2_teleop_episodes(
                output_dir=os.path.join(tmp, "go2_teleop_dataset"),
                num_episodes=10,
                steps_per_episode=15,
                fps=10.0,
                repo_id="ohho-robotics/go2-sim-teleop-v0",
                task="walk to the red box",
                with_video=True,
            )

            p = Path(dataset_path)
            # 1. Verify files exist
            self.assertTrue((p / "README.md").exists(), "dataset card missing")
            self.assertTrue((p / "meta" / "info.json").exists())
            self.assertTrue((p / "meta" / "tasks.jsonl").exists())
            self.assertTrue((p / "meta" / "episodes.jsonl").exists())
            self.assertTrue((p / "meta" / "stats.json").exists())

            # 2. Verify all 10 episodes and MP4 videos exist
            data_dir = p / "data" / "chunk-000"
            vid_dir = p / "videos" / "chunk-000" / "observation.images.front"
            for ep in range(10):
                self.assertTrue((data_dir / f"episode_{ep:06d}.parquet").exists())
                self.assertTrue((vid_dir / f"episode_{ep:06d}.mp4").exists())

            # 3. Verify DatasetReader stats
            reader = DatasetReader(dataset_path)
            self.assertEqual(reader.episode_count, 10)
            self.assertEqual(reader.frame_count, 150)
            self.assertEqual(reader.state_dim, 31)
            self.assertEqual(reader.action_dim, 4)
            self.assertEqual(reader.task_text(0), "walk to the red box")

            self.assertIn("base_vx_mean", stats["action"])
            self.assertIn("mode_id_mean", stats["action"])
            self.assertIn("observation.state", stats)
            self.assertIn("action_raw", stats)

            # 4. Verify LeRobotDataset loads dataset cleanly without error
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(repo_id="local", root=dataset_path)
            self.assertEqual(len(ds), 150)
            self.assertIn("action", ds.meta.features)
            self.assertIn("observation.state", ds.meta.features)
            self.assertIn("action_raw", ds.meta.features)
            self.assertIn("observation.images.front", ds.meta.features)


if __name__ == "__main__":
    unittest.main()
