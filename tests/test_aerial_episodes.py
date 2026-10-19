"""Aerial LeRobot v2.1 episodes from the kinematic drone-sim (OHH-292).

No hardware, no Docker, no PX4, and no ``OHHO_ARM_HARDWARE``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from ohho.adapters.drone_sim import DroneSimTransport
from ohho.adapters.sim import SimTransport
from ohho.aerial import Frame
from ohho.data import DatasetReader, Recorder
from ohho.data.recorder import (
    AERIAL_ACTION_FRAME,
    AERIAL_ACTION_NAMES,
    AERIAL_STATE_NAMES,
    Episode,
    Frame as DataFrame,
)
from ohho.data.writer import write_dataset
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime


def _has_websockets() -> bool:
    return importlib.util.find_spec("websockets") is not None


def _has_lerobot() -> bool:
    return importlib.util.find_spec("lerobot") is not None


def _has_pyarrow() -> bool:
    return importlib.util.find_spec("pyarrow") is not None


def _aircraft(robot_id: str = "px4-sitl"):
    spec = get_spec(robot_id)
    transport = DroneSimTransport(spec)
    transport.connect()
    bot = Robot(spec, transport, NativeRuntime())
    return bot, transport


def _hover(bot: Robot, transport: DroneSimTransport, alt_m: float = 0.4) -> None:
    bot.arm_flight()
    bot.takeoff(alt_m)
    for _ in range(80):
        snap = transport.step(0.1)
        drone = snap.custom["drone"]
        if drone["flight_mode"] == "hold" and drone["landed_state"] == "in_air":
            return
    raise AssertionError("drone-sim did not reach hover")


class TestAerialEpisodeSchema(unittest.TestCase):
    def test_ignored_setpoint_on_the_ground_is_zeros(self):
        bot, _transport = _aircraft()
        rec = Recorder(bot, repo_id="local/px4-sitl", fps=10.0)
        rec.start_episode(task="sim takeoff hover land")
        bot.offboard_velocity(0.4, 0.1, -0.2, 0.05, Frame.BODY_FRD)
        frame = rec.capture_frame()
        self.assertEqual(frame.action, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(frame.action_raw, [0.4, 0.1, -0.2, 0.05])
        self.assertEqual(len(frame.observation_state), 11)
        rec.stop_episode()

    def test_flight_action_is_the_setpoint_not_the_lagged_state(self):
        bot, transport = _aircraft()
        rec = Recorder(bot, repo_id="local/px4-sitl", fps=10.0)
        rec.start_episode(task="sim square")
        _hover(bot, transport)
        bot.offboard_velocity(0.5, -0.1, 0.2, 0.05, Frame.BODY_FRD)
        frame = rec.capture_frame()

        self.assertEqual(frame.action, [0.5, -0.1, 0.2, 0.05])
        self.assertEqual(frame.action_raw, [0.5, -0.1, 0.2, 0.05])
        self.assertEqual(len(frame.observation_state), 11)
        # Hover altitude is up. drone-sim pose z is NED, so z is negative.
        self.assertLess(frame.observation_state[2], -0.3)
        self.assertEqual(frame.observation_state[3], 0.0)  # roll
        self.assertGreater(frame.observation_state[10], 0.5)  # battery
        self.assertLessEqual(frame.observation_state[10], 1.0)
        # The setpoint has not been integrated yet, so body vx is not state vx.
        self.assertLess(abs(frame.observation_state[6]), 0.2)
        self.assertIsNone(frame.image)
        rec.stop_episode()

    def test_info_json_names_and_reader(self):
        bot, transport = _aircraft()
        rec = Recorder(bot, repo_id="local/px4-sitl", fps=10.0)
        rec.start_episode(task="sim takeoff hover land")
        _hover(bot, transport)
        bot.offboard_velocity(0.3, 0.0, 0.0, 0.0, Frame.BODY_FRD)
        rec.capture_frame()
        transport.step(0.1)
        rec.capture_frame()
        rec.stop_episode()

        with tempfile.TemporaryDirectory() as tmp:
            path = rec.save(tmp)
            info_path = Path(path) / "meta" / "info.json"
            with open(info_path, encoding="utf-8") as handle:
                info = json.load(handle)

            self.assertEqual(info["codebase_version"], "v2.1")
            self.assertEqual(info["robot_type"], "px4-sitl")
            self.assertEqual(info["action_frame"], AERIAL_ACTION_FRAME)
            state = info["features"]["observation.state"]
            action = info["features"]["action"]
            self.assertEqual(state["shape"], [11])
            self.assertEqual(state["names"], AERIAL_STATE_NAMES)
            self.assertEqual(action["shape"], [4])
            self.assertEqual(action["names"], AERIAL_ACTION_NAMES)
            self.assertNotIn("observation.images.front", info["features"])
            self.assertNotIn("observation.lidar", info["features"])
            self.assertEqual(info["total_videos"], 0)
            self.assertFalse((Path(path) / "videos").exists())

            reader = DatasetReader(path)
            self.assertEqual(reader.robot_type, "px4-sitl")
            self.assertEqual(reader.state_dim, 11)
            self.assertEqual(reader.action_dim, 4)
            self.assertEqual(reader.task_text(0), "sim takeoff hover land")
            frames = reader.load_episode(0)
            self.assertEqual(len(frames), 2)
            self.assertTrue(any(abs(value) > 1e-6 for value in frames[0].action))
            self.assertEqual(frames[0].action, [0.3, 0.0, 0.0, 0.0])
            self.assertLess(frames[0].observation_state[2], -0.3)
            for index, frame in enumerate(frames):
                self.assertAlmostEqual(frame.timestamp, index / 10.0)

    def test_crazyflie_sim_reuses_the_aerial_names(self):
        episode = Episode(task="sim square", episode_index=0, task_index=0, fps=10.0)
        episode.frames.append(
            DataFrame(
                observation_state=[0.0] * 11,
                action=[0.2, 0.0, 0.0, 0.0],
                timestamp=0.0,
                frame_index=0,
                episode_index=0,
                task_index=0,
                next_done=True,
                action_raw=[0.2, 0.0, 0.0, 0.0],
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_dataset(
                [episode],
                tmp,
                "local/crazyflie-sim",
                10.0,
                ["sim square"],
                robot_type="crazyflie-sim",
            )
            with open(Path(tmp) / "meta" / "info.json", encoding="utf-8") as handle:
                info = json.load(handle)
            self.assertEqual(info["robot_type"], "crazyflie-sim")
            self.assertEqual(info["action_frame"], "BODY_FRD")
            self.assertEqual(
                info["features"]["observation.state"]["names"], AERIAL_STATE_NAMES
            )
            self.assertNotIn("observation.lidar", info["features"])

    def test_ground_dataset_is_not_labelled_aerial(self):
        spec = get_spec("omnibot")
        transport = SimTransport(spec)
        transport.connect()
        bot = Robot(spec, transport, NativeRuntime())
        rec = Recorder(bot, repo_id="local/omnibot", fps=10.0)
        rec.start_episode(task="pick up the cup")
        bot.drive(vx=0.1)
        frame = rec.capture_frame()
        self.assertEqual(len(frame.observation_state), 9)
        self.assertEqual(len(frame.action), 9)
        rec.stop_episode()
        with tempfile.TemporaryDirectory() as tmp:
            path = rec.save(tmp)
            with open(
                os.path.join(path, "meta", "info.json"), encoding="utf-8"
            ) as handle:
                info = json.load(handle)
            self.assertNotIn("action_frame", info)
            self.assertEqual(info["features"]["observation.state"]["shape"], [9])
            self.assertNotEqual(info["robot_type"], "px4-sitl")


@unittest.skipUnless(_has_websockets(), "websockets not installed — [sim-serve] extra")
class TestAerialSimServeEpisodes(unittest.TestCase):
    def test_record_a_few_sim_teleop_episodes(self):
        from ohho.data import record_aerial_teleop_episodes

        with tempfile.TemporaryDirectory() as tmp:
            path, stats = record_aerial_teleop_episodes(
                output_dir=os.path.join(tmp, "px4-sitl"),
                num_episodes=3,
                fps=10.0,
                alt_m=0.4,
            )
            with open(Path(path) / "meta" / "info.json", encoding="utf-8") as handle:
                info = json.load(handle)
            self.assertEqual(info["codebase_version"], "v2.1")
            self.assertEqual(info["robot_type"], "px4-sitl")
            self.assertEqual(info["action_frame"], "BODY_FRD")
            self.assertEqual(
                info["features"]["observation.state"]["names"], AERIAL_STATE_NAMES
            )
            self.assertEqual(info["features"]["action"]["names"], AERIAL_ACTION_NAMES)
            self.assertNotIn("observation.images.front", info["features"])
            self.assertNotIn("observation.lidar", info["features"])
            self.assertEqual(info["tasks"], ["sim takeoff hover land", "sim square"])

            reader = DatasetReader(path)
            self.assertEqual(reader.episode_count, 3)
            self.assertGreater(reader.frame_count, 0)
            airborne = [
                frame
                for frame in reader.all_frames()
                if frame.observation_state[2] < -0.15
            ]
            self.assertTrue(airborne)
            self.assertTrue(
                any(abs(value) > 1e-3 for frame in airborne for value in frame.action)
            )
            square = reader.load_episode(1)
            commanded = {round(frame.action[0], 5) for frame in square}
            self.assertIn(0.4, commanded)
            self.assertIn(-0.4, commanded)
            self.assertIn("vx_mean", stats["action"])

    @unittest.skipUnless(
        _has_lerobot() and _has_pyarrow(),
        "lerobot and pyarrow required to open the dataset with LeRobotDataset",
    )
    def test_lerobot_dataset_opens_the_sim_episode(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from ohho.data import record_aerial_teleop_episodes

        with tempfile.TemporaryDirectory() as tmp:
            path, _stats = record_aerial_teleop_episodes(
                output_dir=os.path.join(tmp, "px4-sitl"),
                num_episodes=2,
                fps=10.0,
                alt_m=0.4,
                repo_id="local/px4-sitl-sim",
            )
            dataset = LeRobotDataset(
                repo_id="local/px4-sitl-sim",
                root=path,
                revision="v2.1",
                download_videos=False,
            )
            self.assertGreater(len(dataset), 0)
            self.assertEqual(
                dataset.meta.features["observation.state"]["names"],
                AERIAL_STATE_NAMES,
            )
            self.assertEqual(
                dataset.meta.features["action"]["names"], AERIAL_ACTION_NAMES
            )


if __name__ == "__main__":
    unittest.main()
