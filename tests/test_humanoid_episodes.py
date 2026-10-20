"""Humanoid LeRobot v2.1 episodes (OHH-340).

Feature names come from the spec's joint groups. One episode is recorded
from sim-serve on the headless humanoid simulator and reloaded. A plugin
robot is used when that plugin and its pinned sim assets are installed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from ohho import capabilities as caps
from ohho.adapters.sim import SimTransport
from ohho.data import DatasetReader, Recorder, record_humanoid_teleop_episode
from ohho.data.humanoid_features import (
    BASE_STATE_NAMES,
    EE_POSE_AXES,
    GRIPPER_NAMES,
    VELOCITY_NAMES,
)
from ohho.humanoid_telemetry import apply_humanoid_telemetry, build_humanoid_telemetry
from ohho.registry import _REGISTRY, register
from ohho.robot import Robot
from ohho.robot_spec import RobotAssets, RobotClass
from ohho.runtime.native import NativeRuntime
from ohho.schema import JointReading, Odometry, Telemetry
from ohho.sim_loop import validate_lerobot_dataset, validate_schema

_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "humanoid" / "dummy_humanoid.xml"
)
_ROOT = Path(__file__).resolve().parents[1]
_VENDOR = re.compile(r"booster|\bt2\b", re.IGNORECASE)
_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_WS = importlib.util.find_spec("websockets") is not None
_HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None
_HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None
_HAS_NUMPY = importlib.util.find_spec("numpy") is not None

_GROUPS = {
    "left_arm": ("l_shoulder", "l_elbow"),
    "right_arm": ("r_shoulder", "r_elbow"),
    "left_leg": ("l_hip",),
    "right_leg": ("r_hip",),
}


def _expected_state_names(groups: dict[str, tuple[str, ...]]) -> list[str]:
    names = [f"{group}.{joint}" for group, joints in groups.items() for joint in joints]
    return names + list(BASE_STATE_NAMES)


def _expected_ee_action_names() -> list[str]:
    names = list(VELOCITY_NAMES)
    for side in ("left", "right"):
        names.extend(f"{side}_ee_{axis}" for axis in EE_POSE_AXES)
    names.extend(GRIPPER_NAMES)
    return names


def _expected_arm_action_names(groups: dict[str, tuple[str, ...]]) -> list[str]:
    names = list(VELOCITY_NAMES)
    for group in ("left_arm", "right_arm"):
        names.extend(f"{group}.{joint}" for joint in groups[group])
    names.extend(GRIPPER_NAMES)
    return names


def _schema_spec():
    from ohho.registry import RobotSpec

    names = tuple(joint for joints in _GROUPS.values() for joint in joints)
    return RobotSpec(
        id="schema-humanoid",
        name="Schema humanoid",
        category="humanoid",
        robot_class=RobotClass.HUMANOID,
        capabilities=(
            caps.BASE_DRIVE,
            caps.BASE_HOLONOMIC,
            caps.HUMANOID_POSTURE,
            caps.HUMANOID_LOCOMOTION,
            caps.WHOLE_BODY_JOINT,
            caps.MANIPULATION_DUAL_ARM,
        ),
        joint_names=names,
        joint_groups=_GROUPS,
        max_lin=0.5,
        max_ang=1.0,
    )


def _schema_bot() -> Robot:
    spec = _schema_spec()
    transport = SimTransport(spec)
    transport.connect()
    transport.send_humanoid = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return Robot(spec, transport, NativeRuntime())


def _install_block(bot: Robot) -> None:
    groups = _GROUPS
    joints = []
    reported: dict[str, dict[str, list[float]]] = {}
    value = 0.1
    for group, names in groups.items():
        q = []
        for name in names:
            joints.append(JointReading(name, value))
            q.append(value)
            value += 0.1
        reported[group] = {"q": q}
    block = build_humanoid_telemetry(
        mode="WALK",
        vendor_mode="",
        fall_state="upright",
        base_rpy=(0.1, -0.2, 0.3),
        base_ang_vel=(0.01, 0.02, 0.03),
        joints=reported,
        odom={"x": 1.5, "y": -0.4, "yaw": 0.3},
        battery={"fraction": 1.0, "voltage": 48.0},
        spec=bot.spec,
    )
    bot._latest = apply_humanoid_telemetry(
        Telemetry(joints=joints, odom=Odometry(x=1.5, y=-0.4, theta=0.3)),
        block,
    )


def _reload(test: unittest.TestCase, path: str, frames: int) -> None:
    test.assertTrue(validate_schema(path))
    reader = DatasetReader(path)
    test.assertEqual(reader.episode_count, 1)
    test.assertEqual(reader.frame_count, frames)
    if _HAS_LEROBOT and _HAS_PYARROW:
        test.assertTrue(validate_lerobot_dataset(path, mock=False))


class TestHumanoidFeatureNames(unittest.TestCase):
    def test_info_uses_joint_groups_and_records_the_arm_form(self):
        bot = _schema_bot()
        rec = Recorder(bot, repo_id="local/schema-humanoid", fps=10.0)
        rec.start_episode(task="walk")
        bot.drive(vx=2.0, vy=0.1, w=-0.2)
        bot.move_ee("left", (0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0), 0.0)
        bot.gripper("right", 0.4)
        _install_block(bot)
        frame = rec.capture_frame()
        rec.stop_episode()

        state_names = _expected_state_names(_GROUPS)
        action_names = _expected_ee_action_names()
        self.assertEqual(len(frame.observation_state), len(state_names))
        expected_q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for got, want in zip(frame.observation_state[:6], expected_q):
            self.assertAlmostEqual(got, want)
        tail = [0.1, -0.2, 0.3, 0.01, 0.02, 0.03, 1.5, -0.4, 0.3]
        for got, want in zip(frame.observation_state[-9:], tail):
            self.assertAlmostEqual(got, want)
        self.assertEqual(frame.action[0], 0.5)
        self.assertEqual(frame.action_raw[0], 2.0)
        self.assertEqual(frame.action[1], 0.1)
        self.assertEqual(frame.action[action_names.index("left_ee_x")], 0.1)
        self.assertEqual(frame.action[action_names.index("right_gripper")], 0.4)

        with tempfile.TemporaryDirectory() as tmp:
            path = rec.save(tmp)
            with open(
                os.path.join(path, "meta", "info.json"), encoding="utf-8"
            ) as handle:
                info = json.load(handle)
            self.assertEqual(info["codebase_version"], "v2.1")
            self.assertEqual(info["robot_type"], "schema-humanoid")
            self.assertEqual(info["arm_action"], "ee_pose")
            self.assertEqual(
                info["features"]["observation.state"]["names"], state_names
            )
            self.assertEqual(info["features"]["action"]["names"], action_names)
            self.assertNotIn("observation.images.head", info["features"])
            _reload(self, path, 1)

    def test_custom_mode_records_arm_joints_from_groups(self):
        bot = _schema_bot()
        rec = Recorder(
            bot, repo_id="local/schema-humanoid", fps=10.0, arm_action="arm_joints"
        )
        rec.start_episode(task="reach")
        bot.move_joints("left_arm", [0.4, -0.5])
        bot.drive(vx=0.2)
        rec.capture_frame()
        rec.stop_episode()
        action_names = _expected_arm_action_names(_GROUPS)
        with tempfile.TemporaryDirectory() as tmp:
            path = rec.save(tmp)
            with open(
                os.path.join(path, "meta", "info.json"), encoding="utf-8"
            ) as handle:
                info = json.load(handle)
            self.assertEqual(info["arm_action"], "arm_joints")
            self.assertEqual(info["features"]["action"]["names"], action_names)
            reader = DatasetReader(path)
            frame = reader.load_episode(0)[0]
            self.assertEqual(
                frame.action[action_names.index("left_arm.l_shoulder")], 0.4
            )
            self.assertEqual(frame.action[action_names.index("left_arm.l_elbow")], -0.5)
            self.assertEqual(frame.action[action_names.index("vx")], 0.2)

    @unittest.skipUnless(_HAS_NUMPY, "numpy is not installed")
    def test_head_image_is_optional(self):
        import numpy as np

        bot = _schema_bot()
        rec = Recorder(
            bot,
            repo_id="local/schema-humanoid",
            fps=10.0,
            camera_fn=lambda: np.zeros((8, 6, 3), dtype=np.uint8),
        )
        rec.start_episode(task="look")
        rec.capture_frame()
        rec.stop_episode()
        self.assertIn("observation.images.head", rec.episodes[0].images)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("ohho.data.writer._encode_mp4_video"):
                path = rec.save(tmp)
            with open(
                os.path.join(path, "meta", "info.json"), encoding="utf-8"
            ) as handle:
                info = json.load(handle)
            self.assertIn("observation.images.head", info["features"])
            self.assertEqual(
                info["features"]["observation.images.head"]["dtype"], "video"
            )
            self.assertEqual(
                info["features"]["observation.images.head"]["shape"],
                [3, 8, 6],
            )

    def test_new_episode_modules_do_not_name_a_vendor(self):
        paths = (
            "ohho/data/humanoid_features.py",
            "ohho/data/humanoid_teleop.py",
            "ohho/data/writer.py",
            "ohho/data/recorder.py",
            "ohho/adapters/humanoid_mujoco.py",
        )
        for rel in paths:
            text = (_ROOT / rel).read_text(encoding="utf-8")
            match = _VENDOR.search(text)
            self.assertIsNone(match, f"{rel}: {match.group(0) if match else ''}")


def _episode_spec():
    from ohho.humanoid_sim import HUMANOID_MUJOCO_BACKEND
    from ohho.registry import RobotSpec

    groups = {
        "left_leg": ("left_hip", "left_knee"),
        "right_leg": ("right_hip", "right_knee"),
    }
    return RobotSpec(
        id="episode-humanoid",
        name="Episode humanoid",
        category="humanoid",
        robot_class=RobotClass.HUMANOID,
        capabilities=(
            caps.BASE_DRIVE,
            caps.BASE_HOLONOMIC,
            caps.HUMANOID_POSTURE,
            caps.HUMANOID_LOCOMOTION,
            caps.WHOLE_BODY_JOINT,
        ),
        capability_params={
            caps.HUMANOID_POSTURE: {
                "modes": ["damping", "prepare", "walk", "custom", "get_up", "lie_down"],
            },
        },
        sim_backend=HUMANOID_MUJOCO_BACKEND,
        joint_names=("left_hip", "left_knee", "right_hip", "right_knee"),
        joint_groups=groups,
        assets=RobotAssets(mjcf=str(_FIXTURE), licence="Apache-2.0"),
        max_lin=0.5,
        max_ang=1.0,
    )


@unittest.skipUnless(_HAS_MUJOCO, "mujoco is not installed")
@unittest.skipUnless(_HAS_WS, "websockets is not installed")
class TestHumanoidSimTeleop(unittest.TestCase):
    def test_one_episode_reloads(self):
        spec = _episode_spec()
        register(spec)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = record_humanoid_teleop_episode(
                    spec.id,
                    tmp,
                    steps=4,
                    fps=10.0,
                    repo_id="local/episode-humanoid",
                    task="walk",
                    vx=0.2,
                )
                with open(
                    Path(path) / "meta" / "info.json", encoding="utf-8"
                ) as handle:
                    info = json.load(handle)
                self.assertEqual(info["codebase_version"], "v2.1")
                self.assertEqual(info["robot_type"], spec.id)
                self.assertEqual(info["arm_action"], "ee_pose")
                self.assertEqual(
                    info["features"]["observation.state"]["names"],
                    _expected_state_names(dict(spec.joint_groups)),
                )
                self.assertNotIn("observation.images.head", info["features"])
                reader = DatasetReader(path)
                self.assertAlmostEqual(reader.load_episode(0)[-1].action[0], 0.2)
                _reload(self, path, 4)
        finally:
            _REGISTRY.pop(spec.id, None)

    def test_camera_is_absent_when_the_model_has_none(self):
        from ohho.adapters.humanoid_mujoco import HumanoidMujocoTransport

        transport = HumanoidMujocoTransport(_episode_spec())
        transport.connect()
        try:
            self.assertIsNone(transport.get_camera_frame())
        finally:
            transport.disconnect()


def _booster_helpers():
    path = Path(__file__).with_name("test_ohho_booster.py")
    spec = importlib.util.spec_from_file_location("ohh340_booster_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(_HAS_MUJOCO, "mujoco is not installed")
@unittest.skipUnless(_HAS_TORCH, "torch is not installed")
@unittest.skipUnless(_HAS_WS, "websockets is not installed")
class TestPluginHumanoidEpisode(unittest.TestCase):
    def test_plugin_robot_type_and_joint_group_names(self):
        helpers = _booster_helpers()
        helpers._ensure_plugin_path()
        from ohho_booster.assets import ensure_assets
        from ohho_booster.joints import JOINT_GROUPS

        try:
            ensure_assets()
        except (OSError, urllib.error.URLError) as exc:
            if os.environ.get("GITHUB_ACTIONS") == "true":
                raise
            self.skipTest(f"pinned assets absent: {exc}")

        from ohho_booster import plugin as plugin_factory

        groups = {name: tuple(joints) for name, joints in JOINT_GROUPS.items()}
        booster = helpers._Entry("booster", "ohho_booster:plugin", plugin_factory)

        def reader():
            return [helpers._builtin_entry(), booster]

        with helpers._discovered(reader):
            with tempfile.TemporaryDirectory() as tmp:
                path = record_humanoid_teleop_episode(
                    "booster-t2",
                    tmp,
                    steps=4,
                    fps=10.0,
                    repo_id="local/humanoid-sim-teleop",
                    task="walk",
                    vx=0.2,
                    ee_pose=(0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0),
                )
                with open(
                    Path(path) / "meta" / "info.json", encoding="utf-8"
                ) as handle:
                    info = json.load(handle)
                self.assertEqual(info["codebase_version"], "v2.1")
                self.assertEqual(info["robot_type"], "booster-t2")
                self.assertEqual(info["arm_action"], "ee_pose")
                self.assertEqual(
                    info["features"]["observation.state"]["names"],
                    _expected_state_names(groups),
                )
                self.assertEqual(
                    len(info["features"]["observation.state"]["names"]), 40
                )
                action_names = info["features"]["action"]["names"]
                self.assertEqual(action_names[:3], ["vx", "vy", "wz"])
                self.assertIn("left_gripper", action_names)
                self.assertIn("right_gripper", action_names)
                frame = DatasetReader(path).load_episode(0)[-1]
                self.assertAlmostEqual(
                    frame.action[action_names.index("left_ee_x")], 0.1
                )
                self.assertAlmostEqual(frame.action[0], 0.2)
                _reload(self, path, 4)


if __name__ == "__main__":
    unittest.main()
