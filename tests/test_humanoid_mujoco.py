"""Generic humanoid-mujoco backend (OHH-329).

MuJoCo, ONNX Runtime and torch are optional. A test skips only when the
package it needs is not installed. The models under ``tests/fixtures/humanoid``
are small stand-ins with no meshes.
"""

from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path

from ohho import capabilities as caps
from ohho.adapters import resolve_transport
from ohho.adapters.errors import AdapterUnavailable
from ohho.adapters.humanoid_mujoco import HumanoidMujocoTransport
from ohho.humanoid import FallState, HumanoidMode, is_upright
from ohho.humanoid_sim import (
    HUMANOID_MUJOCO_BACKEND,
    HumanoidSimConfig,
    ObservationTerm,
    PolicyLayout,
    append_observation_frame,
    bind_sim_config,
    build_observation,
    clear_sim_config,
    clip_vector,
    default_observation_terms,
    fall_state_from_tilt,
    flatten_history,
    observation_size,
    policy_stride,
    projected_gravity,
    quat_wxyz_to_rpy,
    resolve_sim_config,
    rpy_to_quat_wxyz,
)
from ohho.policy_runner import LocomotionPolicy
from ohho.registry import RobotSpec
from ohho.robot import Robot
from ohho.robot_spec import RobotAssets, RobotClass
from ohho.runtime.native import NativeRuntime
from ohho.schema import Velocity

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "humanoid"
HINGE = _FIXTURES / "hinge.xml"
DUMMY = _FIXTURES / "dummy_humanoid.xml"
DUMMY_JOINTS = ("left_hip", "left_knee", "right_hip", "right_knee")
DUMMY_GROUPS = {
    "left_leg": ("left_hip", "left_knee"),
    "right_leg": ("right_hip", "right_knee"),
}
_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None
_NEW_CORE = (
    Path("ohho/humanoid_sim.py"),
    Path("ohho/policy_runner.py"),
    Path("ohho/adapters/humanoid_mujoco.py"),
    Path("ohho/mujoco_viewer.py"),
)
_VENDOR = re.compile(
    r"booster|unitree|yahboom|omnibot|crazyflie|\bt2\b|\bgo2\b|\bpx4\b",
    re.IGNORECASE,
)


def _spec(
    mjcf: Path,
    joints: tuple[str, ...],
    groups: dict[str, tuple[str, ...]],
    *,
    robot_id: str = "dummy-humanoid",
) -> RobotSpec:
    return RobotSpec(
        id=robot_id,
        name="Dummy humanoid",
        category="humanoid",
        robot_class=RobotClass.HUMANOID,
        capabilities=(
            caps.BASE_DRIVE,
            caps.BASE_HOLONOMIC,
            caps.HUMANOID_POSTURE,
            caps.WHOLE_BODY_JOINT,
        ),
        sim_backend=HUMANOID_MUJOCO_BACKEND,
        joint_names=joints,
        joint_groups=groups,
        assets=RobotAssets(mjcf=str(mjcf), licence="Apache-2.0"),
        max_lin=1.0,
        max_ang=1.0,
    )


def _transport(
    mjcf: Path,
    joints: tuple[str, ...],
    groups: dict[str, tuple[str, ...]],
    *,
    kp: float = 20.0,
    kd: float = 1.0,
    default_q: float | tuple[float, ...] = 0.0,
    policy: PolicyLayout | None = None,
    robot_id: str = "dummy-humanoid",
) -> HumanoidMujocoTransport:
    spec = _spec(mjcf, joints, groups, robot_id=robot_id)
    config = HumanoidSimConfig(
        mjcf=str(mjcf),
        joint_names=joints,
        kp=kp,
        kd=kd,
        default_q=default_q,
        policy=policy,
    )
    return HumanoidMujocoTransport(spec, config=config)


def _write_torch(path: Path, n_obs: int, n_act: int, gain: float) -> None:
    import torch

    class _Copy(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            flat = obs.reshape(obs.shape[0], -1)
            out = flat.new_zeros(flat.shape[0], n_act)
            out[:, 0] = flat[:, 6] * gain
            return out

    module = _Copy().eval()
    traced = torch.jit.trace(module, torch.zeros(1, n_obs))
    traced.save(str(path))


def _write_onnx(path: Path, n_obs: int, n_act: int, gain: float) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = np.zeros((n_obs, n_act), dtype=np.float32)
    weight[6, 0] = np.float32(gain)
    tensor = numpy_helper.from_array(weight, name="W")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "W"], ["action"])],
        "policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, n_obs])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, n_act])],
        [tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    # Recent onnx emits IR 14. onnxruntime 1.31 accepts IR <= 13.
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


class TestNoVendorNames(unittest.TestCase):
    def test_new_modules_do_not_name_a_vendor(self):
        root = Path(__file__).resolve().parents[1]
        for rel in _NEW_CORE:
            text = (root / rel).read_text(encoding="utf-8")
            self.assertIsNone(
                _VENDOR.search(text),
                f"{rel} names a vendor: {_VENDOR.search(text)}",
            )


class TestObservationAndTilt(unittest.TestCase):
    def test_default_layout_order_and_scale(self):
        terms = (
            ObservationTerm("command", scale=2.0),
            ObservationTerm("q"),
        )
        obs = build_observation(
            base_ang_vel=(0, 0, 0),
            projected_gravity=(0, 0, -1),
            command=(0.2, -0.1, 0.3),
            q=(1.0, 2.0),
            dq=(0, 0),
            last_action=(0, 0),
            terms=terms,
        )
        self.assertEqual(obs, (0.4, -0.2, 0.6, 1.0, 2.0))
        self.assertEqual(
            [term.name for term in default_observation_terms()],
            [
                "base_ang_vel",
                "projected_gravity",
                "command",
                "q",
                "dq",
                "last_action",
            ],
        )
        self.assertEqual(observation_size(4, default_observation_terms()), 21)

    def test_unknown_term_is_rejected(self):
        with self.assertRaises(ValueError):
            ObservationTerm("vendor_joint")

    def test_policy_stride_follows_the_rate(self):
        self.assertEqual(policy_stride(0.001, 50), 20)
        self.assertEqual(policy_stride(0.002, 50), 10)
        self.assertEqual(policy_stride(0.001, 1000), 1)

    def test_tilt_thresholds(self):
        self.assertIs(fall_state_from_tilt(0.5, 0.0), FallState.UPRIGHT)
        self.assertIs(fall_state_from_tilt(0.0, -0.5), FallState.UPRIGHT)
        self.assertIs(fall_state_from_tilt(0.51, 0.0), FallState.FALLING)
        self.assertIs(fall_state_from_tilt(0.0, 1.0), FallState.FALLEN)
        self.assertIs(fall_state_from_tilt(-1.2, 0.2), FallState.FALLEN)

    def test_history_stack_and_clip(self):
        repeated = append_observation_frame(None, (1.0, -2.0), length=3, init="repeat")
        self.assertEqual(repeated, ((1.0, -2.0), (1.0, -2.0), (1.0, -2.0)))
        zeros = append_observation_frame(None, (1.0, -2.0), length=3, init="zeros")
        self.assertEqual(zeros, ((0.0, 0.0), (0.0, 0.0), (1.0, -2.0)))
        pushed = append_observation_frame(zeros, (3.0, 4.0), length=3, init="zeros")
        self.assertEqual(pushed, ((0.0, 0.0), (1.0, -2.0), (3.0, 4.0)))
        self.assertEqual(flatten_history(pushed), (0.0, 0.0, 1.0, -2.0, 3.0, 4.0))
        self.assertEqual(clip_vector((12.0, -0.2, 4.0), 5.0), (5.0, -0.2, 4.0))
        with self.assertRaises(ValueError):
            PolicyLayout(joint_names=("a",), history_length=0)
        with self.assertRaises(ValueError):
            PolicyLayout(joint_names=(), action_scale=(1.0,))

    def test_gravity_and_quaternion_round_trip(self):
        self.assertEqual(
            projected_gravity([1, 0, 0, 0, 1, 0, 0, 0, 1]), (0.0, 0.0, -1.0)
        )
        quat = rpy_to_quat_wxyz(0.4, -0.2, 0.3)
        roll, pitch, yaw = quat_wxyz_to_rpy(*quat)
        self.assertAlmostEqual(roll, 0.4, places=6)
        self.assertAlmostEqual(pitch, -0.2, places=6)
        self.assertAlmostEqual(yaw, 0.3, places=6)


class TestConfigResolution(unittest.TestCase):
    def tearDown(self) -> None:
        clear_sim_config()

    def test_assets_and_groups_supply_the_model(self):
        spec = _spec(HINGE, ("hinge",), {"link": ("hinge",)})
        config = resolve_sim_config(spec, "")
        self.assertEqual(config.mjcf, str(HINGE))
        self.assertEqual(config.joint_names, ("hinge",))
        self.assertIsNone(config.policy)

    def test_xml_address_overrides_the_path(self):
        spec = _spec(HINGE, ("hinge",), {"link": ("hinge",)})
        config = resolve_sim_config(spec, str(DUMMY))
        self.assertEqual(config.mjcf, str(DUMMY))

    def test_bind_wins_over_assets(self):
        spec = _spec(HINGE, ("hinge",), {"link": ("hinge",)}, robot_id="bound-bot")
        bound = HumanoidSimConfig(
            mjcf=str(DUMMY),
            joint_names=DUMMY_JOINTS,
            kp=3.0,
            kd=0.5,
            default_q=0.1,
        )
        bind_sim_config(spec.id, bound)
        config = resolve_sim_config(spec, "arena")
        self.assertEqual(config.mjcf, str(DUMMY))
        self.assertEqual(config.kp, (3.0, 3.0, 3.0, 3.0))
        self.assertEqual(resolve_sim_config(spec, str(HINGE)).mjcf, str(HINGE))

    def test_missing_mjcf_is_an_error(self):
        spec = RobotSpec(id="empty", name="Empty", category="humanoid")
        with self.assertRaises(ValueError):
            resolve_sim_config(spec, "")


class TestBackendRegistration(unittest.TestCase):
    def test_sim_uri_selects_the_registered_backend(self):
        spec = _spec(HINGE, ("hinge",), {"link": ("hinge",)})
        transport = resolve_transport("sim://", spec)
        self.assertIsInstance(transport, HumanoidMujocoTransport)
        self.assertEqual(transport.protocol, "simulated")
        folded = RobotSpec(
            id="folded",
            name="Folded",
            category="humanoid",
            sim_backend="Humanoid-MuJoCo",
            assets=RobotAssets(mjcf=str(HINGE)),
            joint_names=("hinge",),
            joint_groups={"link": ("hinge",)},
        )
        again = resolve_transport("sim://", folded)
        self.assertIsInstance(again, HumanoidMujocoTransport)


@unittest.skipUnless(_HAS_MUJOCO, "mujoco is not installed")
class TestPdModesAndTelemetry(unittest.TestCase):
    def test_two_mjcf_files_load(self):
        hinge = _transport(HINGE, ("hinge",), {"link": ("hinge",)})
        dummy = _transport(DUMMY, DUMMY_JOINTS, DUMMY_GROUPS, robot_id="walker")
        hinge.connect()
        dummy.connect()
        self.assertEqual([joint.name for joint in hinge.read().joints], ["hinge"])
        self.assertEqual(
            [joint.name for joint in dummy.read().joints], list(DUMMY_JOINTS)
        )
        self.assertAlmostEqual(hinge.timestep, 0.001)
        self.assertAlmostEqual(dummy.timestep, 0.002)
        self.assertNotEqual(hinge.timestep, dummy.timestep)

    def test_timestep_and_base_pose_override(self):
        spec = _spec(DUMMY, DUMMY_JOINTS, DUMMY_GROUPS, robot_id="spawn")
        config = HumanoidSimConfig(
            mjcf=str(DUMMY),
            joint_names=DUMMY_JOINTS,
            kp=10.0,
            kd=0.0,
            default_q=0.2,
            timestep=0.004,
            base_pos=(0.1, -0.2, 1.4),
        )
        transport = HumanoidMujocoTransport(spec, config=config)
        transport.connect()
        self.assertAlmostEqual(transport.timestep, 0.004)
        x, y, z = transport.base_position
        self.assertAlmostEqual(x, 0.1, places=5)
        self.assertAlmostEqual(y, -0.2, places=5)
        self.assertAlmostEqual(z, 1.4, places=5)
        positions = {joint.name: joint.position for joint in transport.read().joints}
        for name in DUMMY_JOINTS:
            self.assertAlmostEqual(positions[name], 0.2, places=5)

    def test_pd_matches_the_formula_and_runs_every_model_step(self):
        transport = _transport(
            HINGE, ("hinge",), {"link": ("hinge",)}, kp=10.0, kd=2.0, default_q=0.5
        )
        transport.connect()
        self.assertEqual(transport.mode, HumanoidMode.OFF)
        transport.send_humanoid("prepare")
        transport.step()
        self.assertEqual(transport.mode, HumanoidMode.PREPARE)
        self.assertEqual(transport.effective_kp(), (10.0,))
        self.assertAlmostEqual(transport.last_tau[0], 5.0)
        resting = _transport(
            HINGE, ("hinge",), {"link": ("hinge",)}, kp=10.0, kd=2.0, default_q=0.5
        )
        resting.connect()
        resting.step()
        self.assertEqual(resting.last_tau, (0.0,))
        before = transport.pd_steps
        transport.step(0.02)
        self.assertEqual(transport.pd_steps - before, 20)

    def test_prepare_holds_damp_releases_and_custom_tracks(self):
        transport = _transport(
            HINGE, ("hinge",), {"link": ("hinge",)}, kp=80.0, kd=2.0, default_q=0.4
        )
        transport.connect()
        self.assertIsNone(transport._policy)
        transport.send_humanoid("prepare")
        transport.step(0.8)
        held = transport.read().joints[0].position
        self.assertAlmostEqual(held, 0.4, delta=0.05)
        self.assertEqual(transport.policy_calls, 0)

        transport.send_humanoid("custom")
        transport.send_joint_command("hinge", -0.25)
        self.assertAlmostEqual(transport.q_target[0], -0.25)
        transport.send_humanoid("move_joints", group="link", q=(-0.3,))
        self.assertAlmostEqual(transport.q_target[0], -0.3)
        transport.step(0.8)
        self.assertAlmostEqual(transport.read().joints[0].position, -0.3, delta=0.05)
        self.assertEqual(transport.policy_calls, 0)

        transport.send_humanoid("damp")
        self.assertEqual(transport.mode, HumanoidMode.DAMPING)
        self.assertEqual(transport.effective_kp(), (0.0,))
        released = transport.read().joints[0].position
        transport.step(0.4)
        self.assertGreater(abs(transport.read().joints[0].position - released), 0.1)

    def test_damp_uses_zero_stiffness(self):
        transport = _transport(
            HINGE, ("hinge",), {"link": ("hinge",)}, kp=10.0, kd=4.0, default_q=0.5
        )
        transport.connect()
        transport.send_humanoid("damp")
        transport.step()
        self.assertAlmostEqual(transport.last_tau[0], 0.0)
        transport.set_joint_velocity("hinge", 0.5)
        transport.step()
        self.assertAlmostEqual(transport.last_tau[0], -2.0, places=5)

    def test_plugin_joint_order_selects_the_actuator(self):
        transport = _transport(
            DUMMY,
            ("left_knee", "left_hip", "right_knee", "right_hip"),
            {
                "left_leg": ("left_knee", "left_hip"),
                "right_leg": ("right_knee", "right_hip"),
            },
            kp=10.0,
            kd=0.0,
            default_q=(0.2, 0.0, 0.0, 0.0),
            robot_id="reordered",
        )
        transport.connect()
        transport.send_humanoid("prepare")
        transport.step()
        self.assertAlmostEqual(transport.last_tau[0], 2.0)
        self.assertAlmostEqual(transport.last_tau[1], 0.0)
        knee = transport.actuator_index("left_knee")
        hip = transport.actuator_index("left_hip")
        self.assertIsNotNone(knee)
        self.assertIsNotNone(hip)
        self.assertNotEqual(knee, hip)
        self.assertAlmostEqual(float(transport._data.ctrl[knee]), 2.0)
        self.assertAlmostEqual(float(transport._data.ctrl[hip]), 0.0)

    def test_telemetry_block_odom_shadow_and_push(self):
        transport = _transport(DUMMY, DUMMY_JOINTS, DUMMY_GROUPS, robot_id="walker")
        transport.connect()
        transport.send_humanoid("prepare")
        transport.set_base_pose(x=1.25, y=-0.4, z=1.0, yaw=0.35)
        snap = transport.read()
        block = snap.custom["humanoid"]
        self.assertEqual(block["mode"], "PREPARE")
        self.assertEqual(block["vendor_mode"], "")
        self.assertEqual(block["fall_state"], "upright")
        self.assertTrue(block["experimental"])
        self.assertTrue(is_upright(snap))
        self.assertAlmostEqual(snap.odom.x, 1.25, places=4)
        self.assertAlmostEqual(snap.odom.y, -0.4, places=4)
        self.assertAlmostEqual(snap.odom.theta, 0.35, places=4)
        self.assertEqual(set(block["joints"]), {"left_leg", "right_leg"})
        self.assertEqual(len(block["joints"]["left_leg"]["q"]), 2)
        self.assertEqual(len(block["joints"]["left_leg"]["dq"]), 2)
        self.assertEqual(len(block["joints"]["left_leg"]["tau"]), 2)
        self.assertNotIn("temp", block["joints"]["left_leg"])

        transport.set_base_pose(roll=0.7)
        self.assertEqual(transport.read().custom["humanoid"]["fall_state"], "falling")
        self.assertFalse(is_upright(transport.read()))
        transport.set_base_pose(roll=1.2)
        self.assertEqual(transport.read().custom["humanoid"]["fall_state"], "fallen")

        calm = _transport(DUMMY, DUMMY_JOINTS, DUMMY_GROUPS, robot_id="calm")
        pushed = _transport(DUMMY, DUMMY_JOINTS, DUMMY_GROUPS, robot_id="pushed")
        calm.connect()
        pushed.connect()
        pushed.inject_push((40.0, 0.0, 0.0), duration_s=0.1)
        calm.step(0.2)
        pushed.step(0.2)
        self.assertGreater(pushed.read().odom.x, calm.read().odom.x + 0.05)

    def test_robot_drive_reaches_the_transport_without_a_gate(self):
        transport = _transport(HINGE, ("hinge",), {"link": ("hinge",)})
        transport.connect()
        bot = Robot(transport.spec, transport, NativeRuntime())
        self.assertFalse(bot.gated)
        bot.prepare()
        self.assertEqual(transport.mode, HumanoidMode.PREPARE)
        bot.damp()
        self.assertEqual(transport.mode, HumanoidMode.DAMPING)
        bot.walk_mode()
        bot.drive(vx=0.2, vy=-0.1, w=0.3)
        self.assertAlmostEqual(transport.command.linear_x, 0.2)
        self.assertAlmostEqual(transport.command.linear_y, -0.1)
        self.assertAlmostEqual(transport.command.angular_z, 0.3)
        bot.emergency_stop()
        self.assertEqual(transport.command.linear_x, 0.0)
        narrow = RobotSpec(
            id="narrow",
            name="Narrow",
            category="humanoid",
            capabilities=(caps.BASE_DRIVE, caps.BASE_HOLONOMIC, caps.HUMANOID_POSTURE),
            sim_backend=HUMANOID_MUJOCO_BACKEND,
            joint_names=("hinge",),
            joint_groups={"link": ("hinge",)},
            assets=RobotAssets(mjcf=str(HINGE)),
            max_lin=0.4,
            max_ang=0.5,
        )
        clamped = HumanoidMujocoTransport(
            narrow,
            config=HumanoidSimConfig(mjcf=str(HINGE), joint_names=("hinge",)),
        )
        clamped.connect()
        Robot(narrow, clamped, NativeRuntime()).drive(vx=2.0, vy=2.0, w=2.0)
        self.assertAlmostEqual(clamped.command.linear_x, 0.4)
        self.assertAlmostEqual(clamped.command.linear_y, 0.4)
        self.assertAlmostEqual(clamped.command.angular_z, 0.5)

    def test_unloadable_mjcf_raises(self):
        transport = _transport(HINGE, ("hinge",), {"link": ("hinge",)})
        transport._config = HumanoidSimConfig(  # type: ignore[method-assign]
            mjcf=str(HINGE.parent / "missing.xml"),
            joint_names=("hinge",),
        )
        with self.assertRaises(AdapterUnavailable):
            transport.connect()


class _PolicyCases:
    write_policy = staticmethod(_write_torch)
    kind = "torchscript"

    def test_runner_copies_the_command(self):
        path = Path(self.id().replace(".", "_") + ".pt")
        if self.kind == "onnx":
            path = path.with_suffix(".onnx")
        self.addCleanup(path.unlink, missing_ok=True)
        self.write_policy(path, 12, 1, 1.0)
        policy = LocomotionPolicy(str(path), rate_hz=50)
        obs = [0.0] * 12
        obs[6] = 0.25
        action = policy.act(obs)
        self.assertEqual(policy.kind, self.kind)
        self.assertEqual(policy.calls, 1)
        self.assertAlmostEqual(action[0], 0.25)
        self.assertEqual(policy.rate_hz, 50)

    def test_policy_rate_scale_order_and_idle_modes(self):
        hinge_path = Path(self.id().replace(".", "_") + "_hinge.pt")
        dummy_path = Path(self.id().replace(".", "_") + "_dummy.pt")
        if self.kind == "onnx":
            hinge_path = hinge_path.with_suffix(".onnx")
            dummy_path = dummy_path.with_suffix(".onnx")
        self.addCleanup(hinge_path.unlink, missing_ok=True)
        self.addCleanup(dummy_path.unlink, missing_ok=True)
        self.write_policy(hinge_path, observation_size(1, _scaled_terms()), 1, 1.0)
        self.write_policy(
            dummy_path, observation_size(4, default_observation_terms()), 4, 1.0
        )

        hinge = _transport(
            HINGE,
            ("hinge",),
            {"link": ("hinge",)},
            kp=10.0,
            kd=0.0,
            default_q=0.0,
            policy=PolicyLayout(
                joint_names=("hinge",),
                terms=_scaled_terms(),
                checkpoint=str(hinge_path),
                rate_hz=50,
                action_scale=1.0,
            ),
        )
        hinge.connect()
        hinge.send_humanoid("prepare")
        hinge.step(0.05)
        self.assertEqual(hinge.policy_calls, 0)
        hinge.send_humanoid("walk_mode")
        hinge.send_velocity(Velocity(0.2, -0.1, 0.05))
        self.assertEqual(hinge.policy_stride_steps(), 20)
        for _ in range(19):
            hinge.step()
        self.assertEqual(hinge.policy_calls, 0)
        hinge.step()
        self.assertEqual(hinge.policy_calls, 1)
        self.assertAlmostEqual(hinge.last_action[0], 0.4, places=4)
        self.assertAlmostEqual(hinge.q_target[0], 0.4, places=4)
        hinge.emergency_stop()
        hinge.step(0.04)
        self.assertEqual(hinge.policy_calls, 1)
        self.assertEqual(hinge.command.linear_x, 0.0)

        dummy = _transport(
            DUMMY,
            DUMMY_JOINTS,
            DUMMY_GROUPS,
            kp=10.0,
            kd=0.0,
            policy=PolicyLayout(
                joint_names=("left_knee", "left_hip", "right_knee", "right_hip"),
                checkpoint=str(dummy_path),
                rate_hz=500,
                action_scale=1.0,
            ),
            robot_id="ordered",
        )
        dummy.connect()
        dummy.send_humanoid("walk_mode")
        dummy.send_velocity(Velocity(0.2, 0.0, 0.0))
        self.assertEqual(dummy.policy_stride_steps(), 1)
        dummy.step()
        self.assertEqual(dummy.policy_calls, 1)
        self.assertAlmostEqual(
            dummy.q_target[DUMMY_JOINTS.index("left_knee")], 0.2, places=4
        )
        self.assertAlmostEqual(
            dummy.q_target[DUMMY_JOINTS.index("left_hip")], 0.0, places=4
        )
        dummy.send_humanoid("custom")
        calls = dummy.policy_calls
        dummy.send_humanoid("move_joints", "left_leg", (0.1, -0.2))
        dummy.step(0.02)
        self.assertEqual(dummy.policy_calls, calls)
        self.assertAlmostEqual(dummy.q_target[0], 0.1, places=5)
        self.assertAlmostEqual(dummy.q_target[1], -0.2, places=5)

    def test_history_scale_and_clip(self):
        path = Path(self.id().replace(".", "_") + "_hist.pt")
        if self.kind == "onnx":
            path = path.with_suffix(".onnx")
        self.addCleanup(path.unlink, missing_ok=True)
        frame = observation_size(1, _scaled_terms())
        self.write_policy(path, frame * 2, 1, 1.0)
        transport = _transport(
            HINGE,
            ("hinge",),
            {"link": ("hinge",)},
            kp=10.0,
            kd=0.0,
            default_q=0.0,
            policy=PolicyLayout(
                joint_names=("hinge",),
                terms=_scaled_terms(),
                checkpoint=str(path),
                rate_hz=50,
                action_scale=(2.0,),
                history_length=2,
                history_init="repeat",
                clip_action=0.1,
            ),
            robot_id="scaled",
        )
        transport.connect()
        transport.send_humanoid("walk_mode")
        transport.send_velocity(Velocity(0.2, 0.0, 0.0))
        for _ in range(20):
            transport.step()
        self.assertEqual(transport.policy_calls, 1)
        self.assertAlmostEqual(transport.last_action[0], 0.1, places=4)
        self.assertAlmostEqual(transport.q_target[0], 0.2, places=4)

    def test_extra_values_are_part_of_the_frame(self):
        path = Path(self.id().replace(".", "_") + "_extra.pt")
        if self.kind == "onnx":
            path = path.with_suffix(".onnx")
        self.addCleanup(path.unlink, missing_ok=True)
        frame = observation_size(1, _scaled_terms()) + 1
        self.write_policy(path, frame, 1, 1.0)
        seen: list[object] = []

        class _Flag:
            def reset(self) -> None:
                return None

            def extend(
                self, *, yaw: float, command: tuple[float, float, float], dt: float
            ):
                seen.append((yaw, command, dt))
                return (1.0,)

        transport = _transport(
            HINGE,
            ("hinge",),
            {"link": ("hinge",)},
            kp=10.0,
            kd=0.0,
            policy=PolicyLayout(
                joint_names=("hinge",),
                terms=_scaled_terms(),
                checkpoint=str(path),
                rate_hz=50,
                extra_factory=_Flag,
            ),
            robot_id="extra",
        )
        transport.connect()
        transport.send_humanoid("walk_mode")
        transport.send_velocity(Velocity(0.2, 0.0, 0.0))
        for _ in range(20):
            transport.step()
        self.assertEqual(transport.policy_calls, 1)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1], (0.2, 0.0, 0.0))
        self.assertAlmostEqual(seen[0][2], 0.02)


def _scaled_terms() -> tuple[ObservationTerm, ...]:
    terms = []
    for name in (
        "base_ang_vel",
        "projected_gravity",
        "command",
        "q",
        "dq",
        "last_action",
    ):
        scale = 2.0 if name == "command" else 1.0
        terms.append(ObservationTerm(name, scale=scale))
    return tuple(terms)


@unittest.skipUnless(_HAS_MUJOCO, "mujoco is not installed")
@unittest.skipUnless(_HAS_TORCH, "torch is not installed")
class TestTorchScriptPolicy(_PolicyCases, unittest.TestCase):
    write_policy = staticmethod(_write_torch)
    kind = "torchscript"


@unittest.skipUnless(_HAS_MUJOCO, "mujoco is not installed")
@unittest.skipUnless(_HAS_ONNX, "onnx is not installed")
@unittest.skipUnless(_HAS_ORT, "onnxruntime is not installed")
class TestOnnxPolicy(_PolicyCases, unittest.TestCase):
    write_policy = staticmethod(_write_onnx)
    kind = "onnx"


class _FakeViewer:
    def __init__(self) -> None:
        self.syncs = 0
        self.closes = 0
        self.running = True

    def is_running(self) -> bool:
        return self.running

    def sync(self) -> None:
        self.syncs += 1

    def close(self) -> None:
        self.closes += 1
        self.running = False


@unittest.skipUnless(_HAS_MUJOCO, "mujoco is not installed")
class TestPassiveViewerSync(unittest.TestCase):
    def test_step_syncs_the_handle_and_disconnect_closes_it(self) -> None:
        transport = _transport(DUMMY, DUMMY_JOINTS, DUMMY_GROUPS)
        transport.connect()
        viewer = _FakeViewer()
        transport.attach_viewer(viewer)
        transport.step()
        transport.step()
        self.assertEqual(viewer.syncs, 2)
        self.assertTrue(transport.viewer_is_running())
        viewer.running = False
        transport.step()
        self.assertEqual(viewer.syncs, 2)
        self.assertFalse(transport.viewer_is_running())
        transport.disconnect()
        self.assertEqual(viewer.closes, 1)
        self.assertFalse(transport.viewer_is_running())

    def test_a_viewer_error_does_not_fail_the_step(self) -> None:
        transport = _transport(HINGE, ("hinge",), {"link": ("hinge",)})
        transport.connect()

        class _Boom:
            def is_running(self) -> bool:
                return True

            def sync(self) -> None:
                raise RuntimeError("gl")

            def close(self) -> None:
                raise RuntimeError("close")

        transport.attach_viewer(_Boom())
        snap = transport.step()
        self.assertIsNotNone(snap.timestamp)
        transport.disconnect()

    def test_step_holds_the_viewer_lock_around_physics(self) -> None:
        transport = _transport(HINGE, ("hinge",), {"link": ("hinge",)})
        transport.connect()
        held: list[bool] = []

        class _Locked:
            def __init__(self) -> None:
                self.syncs = 0
                self.inside = False

            def is_running(self) -> bool:
                return True

            def lock(self) -> object:
                owner = self

                class _Guard:
                    def __enter__(self) -> None:
                        owner.inside = True

                    def __exit__(self, *exc: object) -> bool:
                        held.append(owner.inside)
                        owner.inside = False
                        return False

                return _Guard()

            def sync(self) -> None:
                self.syncs += 1
                held.append(self.inside)

            def close(self) -> None:
                return None

        viewer = _Locked()
        transport.attach_viewer(viewer)
        transport.step()
        self.assertEqual(held, [True, False])
        self.assertEqual(viewer.syncs, 1)
        transport.disconnect()


if __name__ == "__main__":
    unittest.main()
