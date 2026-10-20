"""Upper-body VLA contract, smoke train, and headless eval (OHH-341)."""

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

from ohho import capabilities as caps
from ohho.adapters.sim import SimTransport
from ohho.cli import main
from ohho.data import DatasetReader, Recorder
from ohho.data.humanoid_features import BASE_STATE_NAMES
from ohho.humanoid import apply_posture
from ohho.humanoid_policy import caps_from_spec, class_profile, humanoid_mcp_allowed
from ohho.humanoid_vla import evaluate_upper_body, write_scripted_checkpoint
from ohho.policy_contract import (
    apply_policy_action,
    apply_prediction,
    contract_for_spec,
)
from ohho.registry import _REGISTRY, get_spec, register
from ohho.robot import Robot
from ohho.robot_spec import RobotAssets, RobotClass
from ohho.runtime.native import NativeRuntime
from ohho.safety import AuditLog, SafetyGate
from ohho.schema import JointReading, Telemetry
from ohho.serve import apply_prediction as serve_apply
from ohho.train.recipes import smoke_finetune, training_recipe
from ohho.transport import BaseTransport

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "tests" / "fixtures" / "humanoid" / "upper_body.xml"
_VENDOR = re.compile(r"booster|\bt2\b", re.IGNORECASE)
_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None

_GROUPS = {
    "left_arm": ("left_shoulder", "left_elbow"),
    "right_arm": ("right_shoulder", "right_elbow"),
    "left_leg": ("left_hip",),
    "right_leg": ("right_hip",),
}


class _Spy:
    def __init__(self, spec) -> None:
        self.spec = spec
        self.calls: list[tuple] = []

    def drive(self, vx: float = 0.0, vy: float = 0.0, w: float = 0.0):
        self.calls.append(("drive", vx, vy, w))

    def move_joints(self, *args, **kwargs):
        self.calls.append(("joints", args, kwargs))

    def move_ee(self, *args, **kwargs):
        self.calls.append(("ee", args))

    def gripper(self, *args, **kwargs):
        self.calls.append(("gripper", args))


class _Clock:
    def __init__(self, t: float = 10.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _Link(BaseTransport):
    protocol = "simulated"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple] = []

    def connect(self):
        from ohho.schema import ConnectionState, TransportStatus

        self._state = ConnectionState.CONNECTED
        return TransportStatus(protocol=self.protocol, state=self._state, label="link")

    def disconnect(self) -> None:
        from ohho.schema import ConnectionState

        self._state = ConnectionState.DISCONNECTED

    def status(self):
        from ohho.schema import ConnectionState, TransportStatus

        state = getattr(self, "_state", ConnectionState.IDLE)
        return TransportStatus(protocol=self.protocol, state=state, label="link")

    def read(self) -> Telemetry:
        return Telemetry()

    def send_velocity(self, vel) -> None:
        self.calls.append(("vel", (vel.linear_x, vel.linear_y, vel.angular_z)))

    def send_joint_command(self, name: str, position: float) -> None:
        self.calls.append(("joint", name, position))

    def emergency_stop(self) -> None:
        self.calls.append(("estop",))

    def release_stop(self) -> None:
        self.calls.append(("release",))

    def send_humanoid(self, action: str, *args, **fields) -> None:
        self.calls.append(("humanoid", action, args, fields))


def _upper_spec():
    from ohho.humanoid_sim import HUMANOID_MUJOCO_BACKEND
    from ohho.registry import RobotSpec

    return RobotSpec(
        id="upper-body-humanoid",
        name="Upper body humanoid",
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
        joint_names=tuple(joint for joints in _GROUPS.values() for joint in joints),
        joint_groups=_GROUPS,
        assets=RobotAssets(mjcf=str(_FIXTURE), licence="Apache-2.0"),
        max_lin=0.5,
        max_ang=1.0,
    )


def _schema_spec():
    from ohho.registry import RobotSpec

    groups = {
        "left_arm": ("l_shoulder", "l_elbow"),
        "right_arm": ("r_shoulder", "r_elbow"),
        "left_leg": ("l_hip",),
        "right_leg": ("r_hip",),
    }
    return RobotSpec(
        id="vla-schema-humanoid",
        name="Schema humanoid",
        category="humanoid",
        robot_class=RobotClass.HUMANOID,
        capabilities=(
            caps.BASE_DRIVE,
            caps.BASE_HOLONOMIC,
            caps.HUMANOID_POSTURE,
            caps.HUMANOID_LOCOMOTION,
            caps.WHOLE_BODY_JOINT,
        ),
        joint_groups=groups,
        max_lin=0.5,
        max_ang=1.0,
    ), groups


def _record(bot: Robot, path: str, *, arm_action: str | None = None) -> str:
    recorder = Recorder(
        bot,
        repo_id="local/vla-smoke",
        fps=10.0,
        camera_fn=lambda: None,
        arm_action=arm_action,
    )
    recorder.start_episode("reach")
    for _ in range(4):
        if arm_action == "arm_joints":
            bot.move_joints("left_arm", [0.1, 0.0])
        else:
            bot.move_joints([0.1, 0.0, 0.0, 0.0, 0.0, 0.2])
            bot.drive(vx=0.05)
            bot.transport.step(0.1)  # type: ignore[attr-defined]
        recorder.capture_frame()
    recorder.stop_episode()
    return recorder.save(path)


def _pose(mode: str) -> Telemetry:
    return Telemetry(
        joints=[
            JointReading("left_shoulder", 0.0),
            JointReading("left_elbow", 0.0),
            JointReading("right_shoulder", 0.0),
            JointReading("right_elbow", 0.0),
            JointReading("left_hip", 0.0),
        ],
        custom={
            "error_code": 0,
            "roll": 0.0,
            "pitch": 0.0,
            "humanoid": {
                "mode": mode,
                "fall_state": "upright",
                "vendor_mode": "",
                "base_rpy": [0.0, 0.0, 0.0],
            },
        },
    )


class TestContract(unittest.TestCase):
    def test_mobile_arm_and_humanoid_share_apply(self) -> None:
        omnibot = get_spec("omnibot")
        mobile = contract_for_spec(omnibot)
        self.assertEqual(mobile.kind, "mobile_arm")
        self.assertEqual(mobile.action_dim, 9)
        self.assertEqual(mobile.action_names[-3:], ("base_vx", "base_vy", "base_omega"))
        self.assertEqual(mobile.spans_of("arm_sequence")[0].kind, "arm_sequence")

        transport = SimTransport(omnibot)
        bot = Robot(omnibot, transport, NativeRuntime())
        action = [0.0] * 9
        action[0] = 0.2
        action[6] = 0.1
        apply_policy_action(bot, action, mobile)
        self.assertAlmostEqual(transport._joint_targets["arm_shoulder_pan"], 0.2)
        self.assertAlmostEqual(transport._cmd.linear_x, 0.1)

        go2 = contract_for_spec(get_spec("unitree-go2"))
        self.assertEqual(go2.kind, "legged")
        spy = _Spy(get_spec("unitree-go2"))
        apply_policy_action(spy, [0.3, 0.0, 0.1, 4.0], go2)
        self.assertEqual(spy.calls, [("drive", 0.3, 0.0, 0.1)])

        base = contract_for_spec(get_spec("sim"))
        self.assertEqual(base.kind, "base")
        self.assertEqual(base.action_names, ("base_vx", "base_vy", "base_omega"))

        spec = _upper_spec()
        human = contract_for_spec(spec)
        self.assertEqual(human.kind, "humanoid")
        self.assertIn("base_roll", human.state_names)
        self.assertIn("left_leg.left_hip", human.state_names)
        self.assertEqual(human.action_names[:3], ("vx", "vy", "wz"))
        self.assertIn("left_arm.left_shoulder", human.action_names)
        self.assertNotIn("left_leg.left_hip", human.action_names)
        self.assertTrue(human.state_names[-len(BASE_STATE_NAMES) :] == BASE_STATE_NAMES)

        body = _Spy(spec)
        vector = [0.0] * human.action_dim
        vector[0] = 0.05
        vector[human.action_names.index("left_arm.left_shoulder")] = 0.1
        apply_policy_action(body, vector, human)
        drives = [call for call in body.calls if call[0] == "drive"]
        self.assertEqual(drives, [("drive", 0.05, 0.0, 0.0)])
        groups = [call[1][0] for call in body.calls if call[0] == "joints"]
        self.assertIn("left_arm", groups)
        self.assertNotIn("left_leg", groups)

    def test_short_action_is_rejected(self) -> None:
        contract = contract_for_spec(get_spec("sim"))
        with self.assertRaises(ValueError):
            apply_policy_action(_Spy(get_spec("sim")), [0.0], contract)


class TestServeApply(unittest.TestCase):
    def test_prediction_uses_the_same_mapper(self) -> None:
        omnibot = get_spec("omnibot")
        bot = Robot(omnibot, SimTransport(omnibot), NativeRuntime())
        vector = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        apply_prediction(bot, {"vector": vector})
        self.assertAlmostEqual(bot.transport._joint_targets["arm_shoulder_pan"], 0.1)

        spec = _upper_spec()
        spy = _Spy(spec)
        contract = contract_for_spec(spec)
        named = [0.0] * contract.action_dim
        shoulder = contract.action_names.index("left_arm.left_shoulder")
        named[shoulder] = 0.2
        prediction = {
            "vector": named,
            "names": list(contract.action_names),
            "state_names": list(contract.state_names),
            "arm_action": "arm_joints",
        }
        serve_apply(spy, prediction)
        joints = [call for call in spy.calls if call[0] == "joints"]
        self.assertEqual(joints[0][1][0], "left_arm")
        self.assertAlmostEqual(joints[0][1][1][0], 0.2)
        self.assertFalse(any(call[1][0] == "left_leg" for call in joints))


class TestSmokeTrain(unittest.TestCase):
    def test_same_entry_point_for_mobile_and_humanoid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            omnibot = get_spec("omnibot")
            mobile_bot = Robot(omnibot, SimTransport(omnibot), NativeRuntime())
            mobile_bot.transport.connect()
            mobile_ds = _record(mobile_bot, str(root / "mobile"))
            spec, _groups = _schema_spec()
            human_bot = Robot(spec, SimTransport(spec), NativeRuntime())
            human_bot.transport.connect()
            human_ds = _record(human_bot, str(root / "human"), arm_action="arm_joints")

            mobile = smoke_finetune(mobile_ds, str(root / "mobile-out"), steps=2)
            human = smoke_finetune(human_ds, str(root / "human-out"), steps=2)
            self.assertIn(mobile["status"], ("trained", "mock"))
            self.assertIn(human["status"], ("trained", "mock"))
            self.assertEqual(mobile["ran_steps"], 2)
            self.assertNotEqual(mobile["recipe"]["kind"], human["recipe"]["kind"])
            self.assertEqual(human["recipe"]["kind"], "humanoid")
            self.assertEqual(human["recipe"]["arm_action"], "arm_joints")
            self.assertEqual(mobile["recipe"]["kind"], "mobile_arm")
            for result, dataset in ((mobile, mobile_ds), (human, human_ds)):
                config_path = Path(result["checkpoint"]) / "config.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                info = DatasetReader(dataset).info
                self.assertEqual(
                    config["state_names"],
                    info.features["observation.state"]["names"],
                )
                self.assertEqual(
                    config["action_names"], info.features["action"]["names"]
                )
                self.assertEqual(config["state_dim"], len(config["state_names"]))

            dry = smoke_finetune(human_ds, str(root / "smol"), "smolvla")
            self.assertEqual(dry["status"], "dry_run")
            self.assertEqual(dry["ran_steps"], 0)
            self.assertEqual(dry["recipe"]["gpu_finetune"], "not_run")
            self.assertTrue((root / "smol" / "recipe.json").is_file())
            self.assertFalse((root / "smol" / "smolvla").exists())

    def test_recipe_rejects_a_bad_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            omnibot = get_spec("omnibot")
            bot = Robot(omnibot, SimTransport(omnibot), NativeRuntime())
            bot.transport.connect()
            dataset = _record(bot, tmp)
            with self.assertRaises(ValueError):
                training_recipe(dataset, "missing")


class TestGateRoutesArm(unittest.TestCase):
    def test_arm_action_goes_through_the_humanoid_policy(self) -> None:
        spec = _upper_spec()
        limits = {
            "left_shoulder": (-1.2, 1.2),
            "left_elbow": (-1.2, 1.2),
            "right_shoulder": (-1.2, 1.2),
            "right_elbow": (-1.2, 1.2),
        }
        profile = class_profile(caps_from_spec(spec))
        from dataclasses import replace

        profile = replace(profile, joint_limits=limits)
        clock = _Clock()
        link = _Link()
        with tempfile.TemporaryDirectory() as tmp:
            gate = SafetyGate(
                link,
                spec,
                profile=profile,
                audit=AuditLog(tmp),
                clock=clock,
                environ={},
                isatty=lambda: False,
                input_fn=lambda _prompt: "",
                auto_tick=False,
            )
            gate.connect()
            link._emit_telemetry(_pose("PREPARE"))
            self.assertTrue(gate.arm())
            bot = Robot(spec, gate, NativeRuntime())
            apply_posture(bot, "custom")
            self.assertIn(("humanoid", "custom", (), {}), link.calls)
            link._emit_telemetry(_pose("CUSTOM"))
            clock.advance(0.2)
            contract = contract_for_spec(spec)
            vector = [0.0] * contract.action_dim
            vector[0] = 0.05
            vector[contract.action_names.index("left_arm.left_shoulder")] = 0.1
            apply_policy_action(bot, vector, contract)
            velocities = [call[1] for call in link.calls if call[0] == "vel"]
            self.assertTrue(velocities)
            self.assertGreater(velocities[-1][0], 0.0)
            self.assertLessEqual(velocities[-1][0], 0.05)
            joint_calls = [
                call
                for call in link.calls
                if call[0] == "humanoid" and call[1] == "move_joints"
            ]
            self.assertTrue(joint_calls)
            forwarded = joint_calls[0]
            if forwarded[3]:
                self.assertEqual(forwarded[3]["group"], "left_arm")
                self.assertAlmostEqual(forwarded[3]["q"][0], 0.1)
            else:
                self.assertEqual(forwarded[2][0], "left_arm")
                self.assertAlmostEqual(forwarded[2][1][0], 0.1)
            self.assertFalse(
                any(
                    call[1] == "move_joints"
                    and "left_leg" in (call[2], call[3].values())
                    for call in link.calls
                )
            )
            bot.move_joints("left_leg", [0.1])
            self.assertFalse(
                any(
                    call[0] == "humanoid"
                    and call[1] == "move_joints"
                    and (
                        call[3].get("group") == "left_leg"
                        or (call[2] and call[2][0] == "left_leg")
                    )
                    for call in link.calls
                )
            )
        self.assertFalse(humanoid_mcp_allowed("custom"))


@unittest.skipUnless(_HAS_MUJOCO, "mujoco is not installed")
class TestCollectAndEval(unittest.TestCase):
    def test_scripted_reach_records_and_succeeds(self) -> None:
        from ohho.data.upper_body import record_upper_body_episodes

        spec = _upper_spec()
        register(spec)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                dataset = record_upper_body_episodes(
                    spec,
                    str(root / "data"),
                    episodes=2,
                    steps=8,
                )
                info = json.loads(
                    (Path(dataset) / "meta" / "info.json").read_text(encoding="utf-8")
                )
                self.assertEqual(info["codebase_version"], "v2.1")
                self.assertEqual(info["arm_action"], "arm_joints")
                names = info["features"]["action"]["names"]
                state_names = info["features"]["observation.state"]["names"]
                self.assertEqual(names[:3], ["vx", "vy", "wz"])
                self.assertIn("left_arm.left_shoulder", names)
                self.assertNotIn("left_leg.left_hip", names)
                self.assertIn("left_leg.left_hip", state_names)
                reader = DatasetReader(dataset)
                shoulder = names.index("left_arm.left_shoulder")
                state_shoulder = state_names.index("left_arm.left_shoulder")
                hip = state_names.index("left_leg.left_hip")
                for index, goal in ((0, 0.25), (1, 0.35)):
                    frames = reader.load_episode(index)
                    self.assertAlmostEqual(frames[-1].action[shoulder], goal, places=3)
                    self.assertAlmostEqual(
                        frames[-1].observation_state[state_shoulder], goal, delta=0.08
                    )
                    self.assertLess(abs(frames[-1].observation_state[hip]), 0.05)

                checkpoint = write_scripted_checkpoint(
                    str(root / "scripted"), target=0.3
                )
                report = evaluate_upper_body(
                    spec,
                    checkpoint,
                    str(root / "eval.json"),
                    episodes=2,
                    steps=16,
                )
                self.assertEqual(report["successes"], 2)
                self.assertEqual(report["success_rate"], 1.0)
                self.assertEqual(report["gpu_finetune"], "not_run")
                self.assertIn("not been run", report["note"])
                written = json.loads((root / "eval.json").read_text(encoding="utf-8"))
                self.assertEqual(written["gpu_finetune"], "not_run")
                self.assertEqual(written["policy"], "scripted")
        finally:
            _REGISTRY.pop(spec.id, None)

    @unittest.skipUnless(_HAS_TORCH, "torch is not installed")
    def test_act_checkpoint_evals_without_a_crash(self) -> None:
        from ohho.data.upper_body import record_upper_body_episodes

        spec = _upper_spec()
        register(spec)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                dataset = record_upper_body_episodes(
                    spec, str(root / "data"), episodes=1, steps=4
                )
                trained = smoke_finetune(dataset, str(root / "train"), steps=2)
                report = evaluate_upper_body(
                    spec,
                    trained["checkpoint"],
                    str(root / "act-eval.json"),
                    episodes=1,
                    steps=4,
                )
                self.assertGreaterEqual(report["success_rate"], 0.0)
                self.assertLessEqual(report["success_rate"], 1.0)
                self.assertEqual(report["gpu_finetune"], "not_run")
        finally:
            _REGISTRY.pop(spec.id, None)


class TestNoVendorNames(unittest.TestCase):
    def test_new_modules_do_not_name_a_vendor(self) -> None:
        paths = (
            "ohho/policy_contract.py",
            "ohho/data/upper_body.py",
            "ohho/humanoid_vla.py",
            "ohho/train/recipes.py",
            "docs/humanoid-vla.md",
            "scripts/upper_body_vla.py",
        )
        for rel in paths:
            text = (_ROOT / rel).read_text(encoding="utf-8")
            match = _VENDOR.search(text)
            self.assertIsNone(match, rel)


class TestCli(unittest.TestCase):
    def test_train_smoke_and_vla_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            omnibot = get_spec("omnibot")
            bot = Robot(omnibot, SimTransport(omnibot), NativeRuntime())
            bot.transport.connect()
            dataset = _record(bot, str(root / "data"))
            code = main(
                [
                    "train",
                    "--dataset",
                    dataset,
                    "--policy",
                    "act",
                    "--smoke",
                    "--steps",
                    "2",
                    "--output",
                    str(root / "out"),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((root / "out" / "smoke.json").is_file())
            self.assertEqual(main(["vla"]), 2)

    @unittest.skipUnless(_HAS_MUJOCO, "mujoco is not installed")
    def test_vla_collect_and_eval(self) -> None:
        spec = _upper_spec()
        register(spec)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                code = main(
                    [
                        "vla",
                        "collect",
                        spec.id,
                        "--episodes",
                        "1",
                        "--steps",
                        "4",
                        "--output",
                        str(root / "data"),
                    ]
                )
                self.assertEqual(code, 0)
                checkpoint = write_scripted_checkpoint(str(root / "ckpt"), target=0.3)
                code = main(
                    [
                        "vla",
                        "eval",
                        spec.id,
                        "--checkpoint",
                        checkpoint,
                        "--episodes",
                        "1",
                        "--steps",
                        "8",
                        "--output",
                        str(root / "report.json"),
                    ]
                )
                self.assertEqual(code, 0)
                report = json.loads((root / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["success_rate"], 1.0)
                self.assertEqual(report["gpu_finetune"], "not_run")
        finally:
            _REGISTRY.pop(spec.id, None)


if __name__ == "__main__":
    unittest.main()
