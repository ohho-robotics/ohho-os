"""PPO recipe, task registry, and ONNX actor export."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ohho.train.ppo import (
    TORCH_AVAILABLE,
    EpisodeTracker,
    mini_batch_count,
    resolve_ppo_params,
    rsl_rl_train_config,
    skrl_algorithm_config,
)
from ohho.train.recipe_schema import example_path, load_recipe_file, validate_recipe
from ohho.train.rl_tasks import (
    CARTPOLE_REWARD_THRESHOLD,
    MUJOCO_INVERTED_PENDULUM_REWARD_THRESHOLD,
    RlTaskError,
    TaskRegistryError,
    VecEnv,
    VecStep,
    get_task,
    register,
    shaped_reward,
    task_ids,
    unregister,
)
from ohho.train.run_cli import (
    EXIT_DATASET,
    EXIT_INVALID,
    EXIT_OK,
    EXIT_TRAIN_FAILED,
    STREAM,
    execute,
)
from ohho.train.run_manifest import SCHEMA_ID as RUN_SCHEMA_ID
from ohho.train.run_manifest import validate_run_manifest

_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None
_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
_CAN_TRAIN = TORCH_AVAILABLE and _HAS_ONNX


def _run(path: str | Path, overrides: list[str] | None = None):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = execute(str(path), overrides)
    events = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    return code, events, stderr.getvalue()


def _events_of(events: list[dict], name: str) -> list[dict]:
    return [event for event in events if event.get("event") == name]


class _ProbeEnv(VecEnv):
    """Constant per-step reward. Episodes end on the horizon and succeed."""

    def __init__(self, weight: float, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.weight = weight
        self._steps = [0] * self.num_envs

    def reset(self) -> list[list[float]]:
        self._steps = [0] * self.num_envs
        return [[0.0, 0.0] for _ in range(self.num_envs)]

    def step(self, actions: list[object]) -> VecStep:
        if len(actions) != self.num_envs:
            raise RlTaskError(
                f"probe expected {self.num_envs} actions, got {len(actions)}"
            )
        observations: list[list[float]] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        successes: list[bool] = []
        for index in range(self.num_envs):
            self._steps[index] += 1
            done = self._steps[index] >= self.episode_steps
            rewards.append(self.weight)
            if done:
                self._steps[index] = 0
                terminated.append(False)
                truncated.append(True)
                successes.append(True)
            else:
                terminated.append(False)
                truncated.append(False)
                successes.append(False)
            observations.append([0.0, 0.0])
        return VecStep(observations, rewards, terminated, truncated, successes)


class _ProbeTask:
    id = "probe_goal"

    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def requires_gpu(self) -> bool:
        return False

    def observation_dim(self) -> int:
        return 2

    def action_dim(self) -> int:
        return 1

    def discrete(self) -> bool:
        return False

    def reward_threshold(self) -> float:
        return 1.0

    def default_episode_steps(self) -> int:
        return 2

    def make(self, *, num_envs, reward_weights, seed, episode_steps=None):
        del seed
        weight = float(reward_weights.get("goal", 1.0))
        horizon = 2 if episode_steps is None else int(episode_steps)
        return _ProbeEnv(
            weight,
            num_envs=num_envs,
            observation_dim=2,
            action_dim=1,
            discrete=False,
            episode_steps=horizon,
        )


PROBE = _ProbeTask()


class TestPpoSchema(unittest.TestCase):
    def test_example_names_the_cpu_task(self) -> None:
        recipe = load_recipe_file(example_path("ppo"))
        self.assertEqual(validate_recipe(recipe), [])
        params = recipe["hyperparameters"]
        self.assertEqual(params["task"], "cartpole")
        self.assertEqual(params["backend"], "builtin")
        self.assertEqual(params["num_envs"], 4)
        self.assertEqual(params["clip_coef"], 0.2)
        self.assertEqual(params["entropy_coef"], 0.01)
        self.assertEqual(params["gae_lambda"], 0.95)
        self.assertEqual(params["rollout_steps"], 32)
        self.assertEqual(params["reward_weights"]["alive"], 1.0)

    def test_schema_rejects_bad_ppo_numbers(self) -> None:
        recipe = load_recipe_file(example_path("ppo"))
        clip = json.loads(json.dumps(recipe))
        clip["hyperparameters"]["clip_coef"] = 0
        self.assertTrue(any("clip_coef" in error for error in validate_recipe(clip)))
        weights = json.loads(json.dumps(recipe))
        weights["hyperparameters"]["reward_weights"] = {"alive": "high"}
        self.assertTrue(any("alive" in error for error in validate_recipe(weights)))
        envs = json.loads(json.dumps(recipe))
        envs["hyperparameters"]["num_envs"] = 0
        self.assertTrue(any("num_envs" in error for error in validate_recipe(envs)))

    def test_reference_task_is_documented(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = (root / "docs" / "train" / "ppo.md").read_text(encoding="utf-8")
        self.assertIn("mujoco-inverted-pendulum", text)
        self.assertIn(str(int(MUJOCO_INVERTED_PENDULUM_REWARD_THRESHOLD)), text)
        self.assertIn("follow-up", text)
        self.assertIn("Isaac Lab", text)
        project = (root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('[project.entry-points."ohho.rl_tasks"]', project)
        self.assertIn("ohho.train.rl_tasks:MUJOCO_INVERTED_PENDULUM", project)
        self.assertIn("ohho.train.rl_tasks:ISAAC_VELOCITY", project)

    def test_train_steps_alias_and_unknown_key(self) -> None:
        params = resolve_ppo_params(
            {"hyperparameters": {"train_steps": 3, "num_envs": 2, "rollout_steps": 4}}
        )
        self.assertEqual(params.updates, 3)
        self.assertEqual(params.task, "cartpole")
        self.assertEqual(mini_batch_count(params), 1)
        both = resolve_ppo_params(
            {"hyperparameters": {"updates": 2, "train_steps": 9, "batch_size": 2}}
        )
        self.assertEqual(both.updates, 2)
        self.assertEqual(
            mini_batch_count(
                resolve_ppo_params(
                    {
                        "hyperparameters": {
                            "num_envs": 4,
                            "rollout_steps": 8,
                            "batch_size": 8,
                        }
                    }
                )
            ),
            4,
        )
        with self.assertRaises(Exception) as caught:
            resolve_ppo_params({"hyperparameters": {"mock": True}})
        self.assertIn("unknown hyperparameters", str(caught.exception))


class TestTaskRegistry(unittest.TestCase):
    def test_builtins_and_duplicate_id(self) -> None:
        self.assertTrue(
            {"cartpole", "mujoco-inverted-pendulum", "isaac-velocity"}
            <= set(task_ids())
        )
        cartpole = get_task("cartpole")
        self.assertIsNotNone(cartpole)
        assert cartpole is not None
        self.assertTrue(cartpole.available())
        self.assertFalse(cartpole.requires_gpu())
        self.assertEqual(cartpole.reward_threshold(), CARTPOLE_REWARD_THRESHOLD)
        isaac = get_task("isaac-velocity")
        assert isaac is not None
        self.assertFalse(isaac.available())
        self.assertTrue(isaac.requires_gpu())
        self.assertIn("GPU", isaac.unavailable_reason())
        pendulum = get_task("mujoco-inverted-pendulum")
        assert pendulum is not None
        self.assertEqual(
            pendulum.reward_threshold(), MUJOCO_INVERTED_PENDULUM_REWARD_THRESHOLD
        )
        self.assertEqual(pendulum.available(), _HAS_MUJOCO)

        class Other:
            id = "cartpole"

            def available(self) -> bool:
                return True

            def unavailable_reason(self) -> str:
                return ""

            def observation_dim(self) -> int:
                return 1

            def action_dim(self) -> int:
                return 1

            def discrete(self) -> bool:
                return True

            def make(self, **kwargs: object) -> None:
                del kwargs
                raise AssertionError("replacement task must not be built")

        with self.assertRaises(TaskRegistryError):
            register(Other(), source="test")
        self.assertIs(get_task("cartpole"), cartpole)

    def test_cartpole_is_deterministic_and_reward_weights_apply(self) -> None:
        self.assertEqual(shaped_reward(x=0.0, theta=0.0, reward_weights={}), 1.0)
        self.assertEqual(
            shaped_reward(x=2.0, theta=0.0, reward_weights={"x_penalty": 0.5}),
            1.0 - 2.0,
        )
        self.assertLess(
            shaped_reward(x=0.2, theta=0.1, reward_weights={"theta_penalty": 3.0}),
            shaped_reward(x=0.2, theta=0.1, reward_weights={}),
        )
        task = get_task("cartpole")
        assert task is not None
        first = task.make(num_envs=2, reward_weights={}, seed=7, episode_steps=30)
        second = task.make(num_envs=2, reward_weights={}, seed=7, episode_steps=30)
        self.assertEqual(first._states, second._states)
        plain = task.make(num_envs=1, reward_weights={}, seed=3, episode_steps=30)
        penalised = task.make(
            num_envs=1,
            reward_weights={"x_penalty": 10.0},
            seed=3,
            episode_steps=30,
        )
        plain._states[0] = [0.5, 0.0, 0.1, 0.0]
        penalised._states[0] = [0.5, 0.0, 0.1, 0.0]
        plain._steps[0] = 0
        penalised._steps[0] = 0
        left = plain.step([0])
        right = penalised.step([0])
        self.assertLess(right.rewards[0], left.rewards[0])

    @unittest.skipUnless(_HAS_MUJOCO, "mujoco is not installed")
    def test_mujoco_reference_steps(self) -> None:
        task = get_task("mujoco-inverted-pendulum")
        assert task is not None
        env = task.make(
            num_envs=2, reward_weights={"alive": 1.0}, seed=1, episode_steps=20
        )
        self.assertEqual(env.observation_dim, 4)
        self.assertEqual(env.action_dim, 1)
        self.assertFalse(env.discrete)
        self.assertEqual(env.episode_steps, 20)
        first = env.reset()
        self.assertEqual(len(first), 2)
        self.assertEqual(len(first[0]), 4)
        step = env.step([[0.0], [0.2]])
        self.assertEqual(len(step.rewards), 2)
        self.assertTrue(all(math.isfinite(reward) for reward in step.rewards))
        self.assertTrue(all(len(obs) == 4 for obs in step.observations))


class TestPpoPreflight(unittest.TestCase):
    def test_isaac_and_bad_config_do_not_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            cases = [
                (["hyperparameters.task=isaac-velocity"], "GPU"),
                (["hyperparameters.backend=isaac"], "follow-up"),
                (["hyperparameters.task=not_a_task"], "unknown PPO task"),
                (["hyperparameters.backend=nope"], "unknown PPO backend"),
                (["hyperparameters.extra=1"], "unknown hyperparameters"),
                (["precision=bf16"], "precision"),
                (
                    ["dataset.path=null", "dataset.version_id=dv_ppo"],
                    "not resolved",
                ),
            ]
            for overrides, needle in cases:
                with self.subTest(needle=needle):
                    code, events, err = _run(
                        example_path("ppo"), [*overrides, f"output.dir={out}"]
                    )
                    if "not resolved" in needle:
                        self.assertEqual(code, EXIT_DATASET, err)
                    else:
                        self.assertEqual(code, EXIT_INVALID, err)
                    self.assertIn(needle, err)
                    self.assertFalse(_events_of(events, "train_start"))
                    self.assertFalse((out / "run_manifest.json").exists())

    def test_external_backends_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code, events, err = _run(
                example_path("ppo"),
                ["hyperparameters.backend=rsl_rl", f"output.dir={out}"],
            )
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("discrete", err)
            self.assertFalse(_events_of(events, "train_start"))
            register(PROBE, source="test")
            try:
                recipe = _probe_recipe(Path(tmp), backend="rsl_rl")
                code, events, err = _run(recipe, [f"output.dir={out}"])
                self.assertEqual(code, EXIT_INVALID, err)
                self.assertIn("rsl_rl", err)
                self.assertIn("not installed", err)
                self.assertFalse(_events_of(events, "train_start"))
                recipe = _probe_recipe(Path(tmp), backend="skrl")
                code, _events, err = _run(recipe, [f"output.dir={out}"])
                self.assertEqual(code, EXIT_INVALID, err)
                self.assertIn("skrl", err)
                self.assertIn("not installed", err)
            finally:
                unregister("probe_goal")

    def test_missing_torch_and_onnx_do_not_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with patch("ohho.train.ppo.TORCH_AVAILABLE", False):
                code, events, err = _run(example_path("ppo"), [f"output.dir={out}"])
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("needs torch", err)
            self.assertFalse(_events_of(events, "train_start"))
            real = importlib.util.find_spec

            def hide_onnx(name: str):
                if name == "onnx":
                    return None
                return real(name)

            with (
                patch("ohho.train.ppo.TORCH_AVAILABLE", True),
                patch("ohho.train.ppo.importlib.util.find_spec", hide_onnx),
            ):
                code, events, err = _run(example_path("ppo"), [f"output.dir={out}"])
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("onnx", err)
            self.assertFalse(_events_of(events, "train_start"))
            self.assertFalse(out.exists())


class TestBuiltinPpo(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_gae_matches_a_known_rollout(self) -> None:
        import torch

        from ohho.train.ppo import compute_gae

        rewards = torch.tensor([[1.0], [1.0]])
        values = torch.zeros(2, 1)
        open_episode = torch.zeros(2, 1)
        advantages = compute_gae(
            rewards, values, open_episode, torch.zeros(1), 1.0, 1.0
        )
        self.assertTrue(torch.allclose(advantages[:, 0], torch.tensor([2.0, 1.0])))
        advantages = compute_gae(
            rewards, values, open_episode, torch.tensor([10.0]), 1.0, 1.0
        )
        self.assertTrue(torch.allclose(advantages[:, 0], torch.tensor([12.0, 11.0])))
        finished = torch.tensor([[0.0], [1.0]])
        advantages = compute_gae(
            rewards, values, finished, torch.tensor([10.0]), 1.0, 1.0
        )
        self.assertTrue(torch.allclose(advantages[:, 0], torch.tensor([2.0, 1.0])))

    @unittest.skipUnless(_CAN_TRAIN, "torch and onnx are not installed")
    def test_probe_task_reports_reward_and_success(self) -> None:
        register(PROBE, source="test")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out = root / "out"
                recipe = _probe_recipe(root, backend="builtin")
                code, events, err = _run(recipe, [f"output.dir={out}"])
                self.assertEqual(code, EXIT_OK, err)
                live = [
                    event
                    for event in _events_of(events, "metrics")
                    if "episode_reward" in event
                ]
                self.assertEqual(len(live), 1)
                self.assertAlmostEqual(live[0]["episode_reward"], 2.5)
                self.assertEqual(live[0]["success"], 1.0)
                self.assertEqual(live[0]["stream"], STREAM)
                self.assertLess(
                    events.index(live[0]),
                    events.index(_events_of(events, "done")[0]),
                )
                metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
                self.assertAlmostEqual(metrics["episode_reward"], 2.5)
                self.assertEqual(metrics["success"], 1.0)
                self.assertEqual(metrics["intermediate"][0]["episode_reward"], 2.5)
                self.assertIsInstance(metrics["final_loss"], float)
                weights = _events_of(events, "base_weights")
                self.assertEqual(weights[0]["status"], "not_loaded")
                manifest = json.loads(
                    (out / "run_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["status"], "succeeded")
                self.assertEqual(manifest["family"], "ppo")
                self.assertFalse(manifest["base_weights"]["loaded"])
                self.assertIsNone(manifest["base_weights"]["hash"])
                self.assertIsNone(manifest["dataset"]["hash"])
                self.assertEqual(manifest["final_loss"], metrics["final_loss"])
        finally:
            unregister("probe_goal")

    @unittest.skipUnless(_CAN_TRAIN and _HAS_ORT, "onnxruntime is not installed")
    def test_cartpole_exports_an_onnx_actor_matching_torch(self) -> None:
        import numpy as np
        import onnxruntime as ort
        import torch

        from ohho.train.ppo import load_actor

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code, events, err = _run(
                example_path("ppo"),
                [
                    f"output.dir={out}",
                    "device=cpu",
                    "hyperparameters.num_envs=2",
                    "hyperparameters.rollout_steps=8",
                    "hyperparameters.updates=2",
                    "hyperparameters.num_epochs=2",
                    "hyperparameters.batch_size=8",
                    "hyperparameters.hidden_size=8",
                    "hyperparameters.episode_steps=5",
                    "hyperparameters.seed=1",
                ],
            )
            self.assertEqual(code, EXIT_OK, err + "\n" + _dump(events))
            self.assertTrue(all(event["stream"] == STREAM for event in events))
            self.assertEqual(_events_of(events, "train_start")[0]["task"], "cartpole")
            self.assertEqual(_events_of(events, "hook")[0]["status"], "skipped")
            live = [
                event
                for event in _events_of(events, "metrics")
                if "episode_reward" in event
            ]
            self.assertEqual(len(live), 2)
            for event in live:
                self.assertIsInstance(event["episode_reward"], float)
                self.assertGreaterEqual(event["success"], 0.0)
                self.assertLessEqual(event["success"], 1.0)
            final = _events_of(events, "metrics")[-1]
            self.assertIn("final_loss", final)
            self.assertIn("path", final)
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metrics["intermediate"]), 2)
            self.assertEqual(final["final_loss"], metrics["final_loss"])
            manifest = json.loads(
                (out / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validate_run_manifest(manifest), [])
            self.assertEqual(manifest["schema"], RUN_SCHEMA_ID)
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["device"], "cpu")
            self.assertEqual(manifest["seeds"]["recipe"], 1)
            self.assertEqual(manifest["seeds"]["torch"], 1)
            self.assertTrue(str(manifest["metrics_path"]).endswith("metrics.json"))
            config = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["task"], "cartpole")
            self.assertEqual(config["reward_threshold"], CARTPOLE_REWARD_THRESHOLD)
            self.assertTrue((out / "actor.onnx").is_file())
            self.assertGreater((out / "actor.onnx").stat().st_size, 0)
            actor = load_actor(out / "actor.pt")
            actor.eval()
            obs = torch.randn(1000, actor.observation_dim)
            with torch.no_grad():
                expected = actor(obs).detach().cpu().numpy()
            session = ort.InferenceSession(
                str(out / "actor.onnx"), providers=["CPUExecutionProvider"]
            )
            got = session.run(None, {"observation": obs.numpy()})[0]
            np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
            self.assertEqual(got.shape, (1000, actor.action_dim))

    @unittest.skipUnless(_CAN_TRAIN, "torch and onnx are not installed")
    def test_export_failure_after_train_start_is_a_failed_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with patch(
                "ohho.train.ppo.export_actor_onnx",
                side_effect=RuntimeError("export broke"),
            ):
                code, events, err = _run(
                    example_path("ppo"),
                    [
                        f"output.dir={out}",
                        "hyperparameters.num_envs=1",
                        "hyperparameters.rollout_steps=2",
                        "hyperparameters.updates=1",
                        "hyperparameters.num_epochs=1",
                        "hyperparameters.hidden_size=4",
                        "hyperparameters.episode_steps=2",
                    ],
                )
            self.assertEqual(code, EXIT_TRAIN_FAILED, err)
            self.assertIn("export broke", err)
            self.assertTrue(_events_of(events, "train_start"))
            self.assertFalse(_events_of(events, "done"))
            manifest = json.loads(
                (out / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")

    @unittest.skipUnless(
        _CAN_TRAIN and _HAS_MUJOCO, "mujoco, torch, and onnx are not installed"
    )
    def test_mujoco_task_trains_on_the_builtin_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code, events, err = _run(
                example_path("ppo"),
                [
                    f"output.dir={out}",
                    "hyperparameters.task=mujoco-inverted-pendulum",
                    "hyperparameters.num_envs=1",
                    "hyperparameters.rollout_steps=4",
                    "hyperparameters.updates=1",
                    "hyperparameters.num_epochs=1",
                    "hyperparameters.batch_size=4",
                    "hyperparameters.hidden_size=8",
                    "hyperparameters.episode_steps=8",
                ],
            )
            self.assertEqual(code, EXIT_OK, err)
            config = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["task"], "mujoco-inverted-pendulum")
            self.assertEqual(
                config["reward_threshold"], MUJOCO_INVERTED_PENDULUM_REWARD_THRESHOLD
            )
            self.assertFalse(config["discrete"])
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("episode_reward", metrics)
            self.assertIn("success", metrics)
            self.assertTrue(_events_of(events, "metrics"))
            self.assertTrue((out / "actor.onnx").is_file())


class TestExternalBackends(unittest.TestCase):
    def test_configs_carry_the_exposed_params(self) -> None:
        params = resolve_ppo_params(
            {
                "hyperparameters": {
                    "num_envs": 4,
                    "rollout_steps": 16,
                    "updates": 5,
                    "num_epochs": 3,
                    "batch_size": 8,
                    "lr": 0.002,
                    "clip_coef": 0.15,
                    "entropy_coef": 0.01,
                    "gae_lambda": 0.9,
                    "gamma": 0.98,
                    "hidden_size": 12,
                }
            }
        )
        rsl = rsl_rl_train_config(params)
        self.assertEqual(rsl["num_steps_per_env"], 16)
        self.assertEqual(rsl["algorithm"]["class_name"], "PPO")
        self.assertEqual(rsl["algorithm"]["learning_rate"], 0.002)
        self.assertEqual(rsl["algorithm"]["clip_param"], 0.15)
        self.assertEqual(rsl["algorithm"]["entropy_coef"], 0.01)
        self.assertEqual(rsl["algorithm"]["lam"], 0.9)
        self.assertEqual(rsl["algorithm"]["gamma"], 0.98)
        self.assertEqual(rsl["algorithm"]["num_learning_epochs"], 3)
        self.assertEqual(rsl["algorithm"]["num_mini_batches"], 8)
        self.assertEqual(rsl["policy"]["actor_hidden_dims"], [12, 12])
        skrl = skrl_algorithm_config(params)
        self.assertEqual(skrl["rollouts"], 16)
        self.assertEqual(skrl["learning_rate"], 0.002)
        self.assertEqual(skrl["ratio_clip"], 0.15)
        self.assertEqual(skrl["entropy_loss_scale"], 0.01)
        self.assertEqual(skrl["lambda"], 0.9)
        self.assertEqual(skrl["discount_factor"], 0.98)

    @unittest.skipUnless(_CAN_TRAIN, "torch and onnx are not installed")
    def test_rsl_rl_wrapper_receives_the_task_and_params(self) -> None:
        import torch

        from ohho.train.ppo import PolicyActor

        recorded: dict[str, object] = {}

        class FakeRunner:
            def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
                recorded["cfg"] = train_cfg
                recorded["num_envs"] = env.num_envs
                recorded["log_dir"] = log_dir
                recorded["device"] = device
                self.env = env
                self.train_cfg = train_cfg
                self.alg = SimpleNamespace(
                    policy=PolicyActor(
                        env.task.observation_dim,
                        env.task.action_dim,
                        8,
                        discrete=False,
                    )
                )

            def learn(self, num_learning_iterations, init_at_random_ep_len=False):
                recorded["iterations"] = num_learning_iterations
                recorded["init_at_random_ep_len"] = init_at_random_ep_len
                self.env.get_observations()
                steps = num_learning_iterations * int(
                    self.train_cfg["num_steps_per_env"]
                )
                actions = torch.zeros(self.env.num_envs, self.env.num_actions)
                for _ in range(steps):
                    self.env.step(actions)

        register(PROBE, source="test")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out = root / "out"
                recipe = _probe_recipe(root, backend="rsl_rl")
                with patch(
                    "ohho.train.ppo._import_rsl_rl_runner", return_value=FakeRunner
                ):
                    code, events, err = _run(recipe, [f"output.dir={out}"])
                self.assertEqual(code, EXIT_OK, err)
                cfg = recorded["cfg"]
                assert isinstance(cfg, dict)
                self.assertEqual(cfg["algorithm"]["learning_rate"], 0.001)
                self.assertEqual(cfg["algorithm"]["clip_param"], 0.2)
                self.assertEqual(cfg["algorithm"]["lam"], 0.95)
                self.assertEqual(cfg["num_steps_per_env"], 4)
                self.assertEqual(recorded["iterations"], 1)
                self.assertEqual(recorded["num_envs"], 2)
                self.assertIsNone(recorded["log_dir"])
                live = [
                    event
                    for event in _events_of(events, "metrics")
                    if "episode_reward" in event
                ]
                self.assertEqual(live[0]["episode_reward"], 2.5)
                self.assertEqual(live[0]["success"], 1.0)
                self.assertTrue((out / "actor.onnx").is_file())
        finally:
            unregister("probe_goal")

    @unittest.skipUnless(_CAN_TRAIN, "torch and onnx are not installed")
    def test_skrl_wrapper_receives_the_task_and_params(self) -> None:
        import torch

        from ohho.train.ppo import PolicyActor

        recorded: dict[str, object] = {}
        actor = PolicyActor(2, 1, 8, discrete=False)

        class FakeMemory:
            def __init__(self, **kwargs: object) -> None:
                recorded["memory"] = kwargs

        class FakeAgent:
            def __init__(self, **kwargs: object) -> None:
                recorded["agent"] = kwargs

            def init(self, trainer_cfg: dict | None = None) -> None:
                recorded["init"] = trainer_cfg

        class FakeTrainer:
            def __init__(self, cfg: dict, env: object, agents: object) -> None:
                recorded["trainer"] = cfg
                self.cfg = cfg
                self.env = env
                self.agents = agents

            def train(self) -> None:
                vector_steps = int(self.cfg["timesteps"]) // int(self.env.num_envs)
                self.env.reset()
                actions = torch.zeros(self.env.num_envs, self.env.task.action_dim)
                for _ in range(vector_steps):
                    self.env.step(actions)

        parts = {
            "PPO": FakeAgent,
            "PPO_DEFAULT_CONFIG": {"experiment": {"write_interval": "auto"}},
            "RandomMemory": FakeMemory,
            "SequentialTrainer": FakeTrainer,
        }
        register(PROBE, source="test")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out = root / "out"
                recipe = _probe_recipe(root, backend="skrl")
                with (
                    patch("ohho.train.ppo._import_skrl_agent", return_value=parts),
                    patch(
                        "ohho.train.ppo.build_skrl_models",
                        return_value={"policy": actor, "value": actor},
                    ),
                ):
                    code, events, err = _run(recipe, [f"output.dir={out}"])
                self.assertEqual(code, EXIT_OK, err)
                agent = recorded["agent"]
                assert isinstance(agent, dict)
                cfg = agent["cfg"]
                assert isinstance(cfg, dict)
                self.assertEqual(cfg["learning_rate"], 0.001)
                self.assertEqual(cfg["ratio_clip"], 0.2)
                self.assertEqual(cfg["entropy_loss_scale"], 0.0)
                self.assertEqual(cfg["discount_factor"], 0.99)
                self.assertEqual(cfg["lambda"], 0.95)
                self.assertEqual(cfg["rollouts"], 4)
                self.assertEqual(cfg["experiment"]["write_interval"], 0)
                memory = recorded["memory"]
                assert isinstance(memory, dict)
                self.assertEqual(memory["num_envs"], 2)
                self.assertEqual(memory["memory_size"], 4)
                live = [
                    event
                    for event in _events_of(events, "metrics")
                    if "episode_reward" in event
                ]
                self.assertEqual(live[0]["success"], 1.0)
                self.assertAlmostEqual(live[0]["episode_reward"], 2.5)
        finally:
            unregister("probe_goal")


class TestEpisodeTracker(unittest.TestCase):
    def test_flush_emits_intermediate_metrics(self) -> None:
        rows: list[dict] = []
        tracker = EpisodeTracker(1, rows.append, 2, auto_flush=False)
        tracker.observe([1.0], [False], [False], [False])
        tracker.observe([1.5], [False], [True], [True])
        tracker.flush(0.25)
        self.assertEqual(rows[0]["event"], "metrics")
        self.assertEqual(rows[0]["episode_reward"], 2.5)
        self.assertEqual(rows[0]["success"], 1.0)
        self.assertEqual(tracker.history[0]["policy_loss"], 0.25)


def _probe_recipe(root: Path, *, backend: str) -> Path:
    path = root / f"probe-{backend}.yaml"
    path.write_text(
        "\n".join(
            [
                "schema: ohho.recipe/v1",
                "family: ppo",
                "base_weights:",
                "  uri: local:ppo-init",
                "dataset:",
                "  path: datasets/does-not-exist",
                "hyperparameters:",
                "  task: probe_goal",
                f"  backend: {backend}",
                "  num_envs: 2",
                "  rollout_steps: 4",
                "  updates: 1",
                "  num_epochs: 1",
                "  batch_size: 8",
                "  lr: 0.001",
                "  clip_coef: 0.2",
                "  entropy_coef: 0.0",
                "  gae_lambda: 0.95",
                "  gamma: 0.99",
                "  hidden_size: 8",
                "  episode_steps: 2",
                "  reward_weights:",
                "    goal: 1.25",
                "  seed: 2",
                "precision: fp32",
                "device: cpu",
                "output:",
                "  dir: out",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _dump(events: list[dict]) -> str:
    return "\n".join(json.dumps(event) for event in events)


if __name__ == "__main__":
    unittest.main()
