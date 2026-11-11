"""PPO recipe runner.

The default backend is a small PyTorch PPO that trains on CPU and writes
``actor.onnx``. ``rsl_rl`` and ``skrl`` are optional imports used when a
recipe selects them. Isaac Lab is refused with a GPU follow-up message.

``actor.onnx`` is the deterministic actor: action logits for a discrete
task, and the Gaussian mean for a continuous task. Sampling stays in the
trainer. This export is local to the recipe.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .recipe_registry import TrainResult
from .recipe_schema import DatasetUnavailable, RecipeConfigError, TrainFailed
from .rl_tasks import (
    RlTaskError,
    VecEnv,
    get_task,
    task_ids,
    task_requires_gpu,
)

Emit = Callable[[dict[str, Any]], None]

BUILTIN_BACKEND = "builtin"
RSL_RL_BACKEND = "rsl_rl"
SKRL_BACKEND = "skrl"
ISAAC_BACKEND = "isaac"
_BACKENDS = (BUILTIN_BACKEND, RSL_RL_BACKEND, SKRL_BACKEND, ISAAC_BACKEND)

_PPO_KEYS = frozenset(
    {
        "task",
        "backend",
        "num_envs",
        "rollout_steps",
        "updates",
        "train_steps",
        "num_epochs",
        "batch_size",
        "lr",
        "clip_coef",
        "entropy_coef",
        "gae_lambda",
        "gamma",
        "reward_weights",
        "seed",
        "hidden_size",
        "episode_steps",
    }
)

_TORCH_MESSAGE = (
    "PPO training needs torch. Install it with: pip install 'ohho-os[train]'"
)
_ONNX_MESSAGE = (
    "PPO actor export needs the onnx package. "
    "Install it with: pip install 'ohho-os[humanoid-sim]'"
)
_ISAAC_BACKEND = (
    "PPO backend 'isaac' needs a GPU and Isaac Lab. "
    "GPU training is a follow-up in this build."
)
_VALUE_COEF = 0.5
_MAX_GRAD_NORM = 0.5

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by patching TORCH_AVAILABLE
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class PpoParams:
    """Resolved PPO hyperparameters."""

    task: str
    backend: str
    num_envs: int
    rollout_steps: int
    updates: int
    num_epochs: int
    batch_size: int
    lr: float
    clip_coef: float
    entropy_coef: float
    gae_lambda: float
    gamma: float
    hidden_size: int
    reward_weights: dict[str, float]
    episode_steps: int | None
    seed: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "backend": self.backend,
            "num_envs": self.num_envs,
            "rollout_steps": self.rollout_steps,
            "updates": self.updates,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "clip_coef": self.clip_coef,
            "entropy_coef": self.entropy_coef,
            "gae_lambda": self.gae_lambda,
            "gamma": self.gamma,
            "hidden_size": self.hidden_size,
            "reward_weights": dict(self.reward_weights),
            "episode_steps": self.episode_steps,
            "seed": self.seed,
        }


class EpisodeTracker:
    """Episode reward and success for one PPO run.

    ``flush`` appends one row and emits an intermediate ``metrics`` event.
    A bridge with ``auto_flush`` does that once per rollout. The built-in
    trainer flushes itself at the end of each update.
    """

    def __init__(
        self,
        num_envs: int,
        emit: Emit,
        rollout_steps: int,
        *,
        auto_flush: bool,
    ) -> None:
        self.num_envs = num_envs
        self.emit = emit
        self.rollout_steps = rollout_steps
        self.auto_flush = auto_flush
        self.running = [0.0] * num_envs
        self.history: list[dict[str, Any]] = []
        self._finished_rewards: list[float] = []
        self._finished_success: list[float] = []
        self._vector_steps = 0
        self.update = 0
        self._closed = False

    def observe(
        self,
        rewards: list[float],
        terminated: list[bool],
        truncated: list[bool],
        successes: list[bool],
    ) -> None:
        for index, reward in enumerate(rewards):
            self.running[index] += float(reward)
            if bool(terminated[index] or truncated[index]):
                self._finished_rewards.append(self.running[index])
                self._finished_success.append(1.0 if successes[index] else 0.0)
                self.running[index] = 0.0
        self._vector_steps += 1
        if (
            self.auto_flush
            and self.rollout_steps > 0
            and self._vector_steps % self.rollout_steps == 0
        ):
            self.flush(None)

    def flush(self, policy_loss: float | None) -> dict[str, Any]:
        self.update += 1
        if self._finished_rewards:
            episode_reward = sum(self._finished_rewards) / len(self._finished_rewards)
            success = sum(self._finished_success) / len(self._finished_success)
        elif self.running:
            episode_reward = sum(self.running) / len(self.running)
            success = 0.0
        else:
            episode_reward = 0.0
            success = 0.0
        row: dict[str, Any] = {
            "update": self.update,
            "episode_reward": float(episode_reward),
            "success": float(success),
        }
        if policy_loss is not None:
            row["policy_loss"] = float(policy_loss)
        self.history.append(row)
        self.emit(
            {
                "event": "metrics",
                "family": "ppo",
                "update": self.update,
                "episode_reward": row["episode_reward"],
                "success": row["success"],
            }
        )
        self._finished_rewards.clear()
        self._finished_success.clear()
        return row

    def close(self, policy_loss: float | None) -> None:
        """Flush a trailing partial rollout, if the backend stopped mid-way."""
        if self._closed:
            return
        self._closed = True
        partial = (
            self.auto_flush
            and self.rollout_steps > 0
            and self._vector_steps % self.rollout_steps != 0
        )
        if partial or not self.history:
            self.flush(policy_loss)


def resolve_ppo_params(recipe: Mapping[str, Any]) -> PpoParams:
    """Read and check ``hyperparameters`` for the PPO family."""
    raw = recipe.get("hyperparameters") or {}
    if not isinstance(raw, dict):
        raise RecipeConfigError("hyperparameters must be a mapping")
    unknown = sorted(set(raw) - _PPO_KEYS)
    if unknown:
        listed = ", ".join(unknown)
        raise RecipeConfigError(f"unknown hyperparameters for family ppo: {listed}")
    if "updates" in raw:
        updates = _positive_int("updates", raw["updates"])
    elif "train_steps" in raw:
        updates = _positive_int("train_steps", raw["train_steps"])
    else:
        updates = 8
    weights = raw.get("reward_weights", {})
    if not isinstance(weights, dict):
        raise RecipeConfigError("hyperparameters.reward_weights must be a mapping")
    cleaned: dict[str, float] = {}
    for key, value in weights.items():
        if not isinstance(key, str):
            raise RecipeConfigError("reward weight names must be strings")
        cleaned[key] = _finite("reward_weights." + key, value)
    episode_steps = None
    if "episode_steps" in raw:
        episode_steps = _positive_int("episode_steps", raw["episode_steps"])
    seed = None
    if "seed" in raw:
        seed = _seed(raw["seed"])
    return PpoParams(
        task=str(raw.get("task", "cartpole")),
        backend=str(raw.get("backend", BUILTIN_BACKEND)),
        num_envs=_positive_int("num_envs", raw.get("num_envs", 4)),
        rollout_steps=_positive_int("rollout_steps", raw.get("rollout_steps", 32)),
        updates=updates,
        num_epochs=_positive_int("num_epochs", raw.get("num_epochs", 4)),
        batch_size=_positive_int("batch_size", raw.get("batch_size", 64)),
        lr=_positive_float("lr", raw.get("lr", 3e-4)),
        clip_coef=_unit_float("clip_coef", raw.get("clip_coef", 0.2)),
        entropy_coef=_nonnegative("entropy_coef", raw.get("entropy_coef", 0.0)),
        gae_lambda=_closed_unit("gae_lambda", raw.get("gae_lambda", 0.95)),
        gamma=_closed_unit("gamma", raw.get("gamma", 0.99)),
        hidden_size=_positive_int("hidden_size", raw.get("hidden_size", 32)),
        reward_weights=cleaned,
        episode_steps=episode_steps,
        seed=seed,
    )


def mini_batch_count(params: PpoParams) -> int:
    """How many minibatches cover one rollout. 1 when the batch does not divide."""
    total = params.num_envs * params.rollout_steps
    batch = params.batch_size
    if batch < 1 or total < batch or total % batch != 0:
        return 1
    return total // batch


def rsl_rl_train_config(params: PpoParams) -> dict[str, Any]:
    """``rsl_rl`` 2.3 ``OnPolicyRunner`` config from the recipe params.

    The runner reads ``num_steps_per_env`` at the top level and pops
    ``class_name`` from ``algorithm`` and ``policy``.
    """
    hidden = params.hidden_size
    return {
        "num_steps_per_env": params.rollout_steps,
        "save_interval": params.updates,
        "empirical_normalization": False,
        "logger": "tensorboard",
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": params.clip_coef,
            "entropy_coef": params.entropy_coef,
            "num_learning_epochs": params.num_epochs,
            "num_mini_batches": mini_batch_count(params),
            "learning_rate": params.lr,
            "schedule": "fixed",
            "gamma": params.gamma,
            "lam": params.gae_lambda,
            "desired_kl": 0.01,
            "max_grad_norm": _MAX_GRAD_NORM,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "actor_hidden_dims": [hidden, hidden],
            "critic_hidden_dims": [hidden, hidden],
            "activation": "elu",
        },
    }


def skrl_algorithm_config(params: PpoParams) -> dict[str, Any]:
    """Fields merged over skrl's ``PPO_DEFAULT_CONFIG`` (1.4)."""
    return {
        "rollouts": params.rollout_steps,
        "learning_epochs": params.num_epochs,
        "mini_batches": mini_batch_count(params),
        "discount_factor": params.gamma,
        "lambda": params.gae_lambda,
        "learning_rate": params.lr,
        "ratio_clip": params.clip_coef,
        "value_clip": params.clip_coef,
        "clip_predicted_values": True,
        "entropy_loss_scale": params.entropy_coef,
        "value_loss_scale": 1.0,
        "grad_norm_clip": _MAX_GRAD_NORM,
    }


def run_ppo(recipe: Mapping[str, Any], emit: Emit) -> TrainResult:
    """Train the recipe and write the actor, metrics, and ONNX file."""
    _check_precision(recipe)
    params = resolve_ppo_params(recipe)
    _reject_version_only(recipe)
    task = _require_task(params.task)
    _require_backend(params, task)
    if not TORCH_AVAILABLE:
        raise RecipeConfigError(_TORCH_MESSAGE)
    if importlib.util.find_spec("onnx") is None:
        raise RecipeConfigError(_ONNX_MESSAGE)
    device = _resolve_device(str(recipe.get("device", "cpu")))
    try:
        env = task.make(
            num_envs=params.num_envs,
            reward_weights=params.reward_weights,
            seed=0 if params.seed is None else params.seed,
            episode_steps=params.episode_steps,
        )
    except RlTaskError as exc:
        raise RecipeConfigError(str(exc)) from exc
    _check_env(env, params)
    _emit_declared(recipe, emit)
    emit(
        {
            "event": "train_start",
            "family": "ppo",
            "device": device,
            "task": params.task,
            "backend": params.backend,
            "updates": params.updates,
        }
    )
    return _run_with_torch(recipe, params, env, device, emit)


def _import_rsl_rl_runner() -> Any:
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:
        raise RecipeConfigError(
            "PPO backend 'rsl_rl' is not installed. "
            "Install rsl_rl, or set hyperparameters.backend to builtin."
        ) from exc
    return OnPolicyRunner


def _import_skrl_agent() -> dict[str, Any]:
    try:
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
        from skrl.memories.torch import RandomMemory
        from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
        from skrl.trainers.torch import SequentialTrainer
    except ImportError as exc:
        raise RecipeConfigError(
            "PPO backend 'skrl' is not installed. "
            "Install skrl, or set hyperparameters.backend to builtin."
        ) from exc
    return {
        "PPO": PPO,
        "PPO_DEFAULT_CONFIG": PPO_DEFAULT_CONFIG,
        "RandomMemory": RandomMemory,
        "SequentialTrainer": SequentialTrainer,
        "Model": Model,
        "GaussianMixin": GaussianMixin,
        "DeterministicMixin": DeterministicMixin,
    }


def _require_task(task_id: str) -> Any:
    if not task_id or not isinstance(task_id, str):
        raise RecipeConfigError("hyperparameters.task must be a task id")
    task = get_task(task_id)
    if task is None:
        known = ", ".join(task_ids()) or "(none)"
        raise RecipeConfigError(
            f"unknown PPO task {task_id!r}. Registered tasks: {known}"
        )
    if task_requires_gpu(task) or not task.available():
        reason = task.unavailable_reason()
        if not isinstance(reason, str) or not reason.strip():
            reason = f"PPO task {task_id!r} is not available in this build"
        raise RecipeConfigError(reason)
    return task


def _require_backend(params: PpoParams, task: Any) -> None:
    if params.backend not in _BACKENDS:
        known = ", ".join(_BACKENDS)
        raise RecipeConfigError(
            f"unknown PPO backend {params.backend!r}. Known backends: {known}"
        )
    if params.backend == ISAAC_BACKEND:
        raise RecipeConfigError(_ISAAC_BACKEND)
    if params.backend in {RSL_RL_BACKEND, SKRL_BACKEND} and bool(task.discrete()):
        raise RecipeConfigError(
            f"PPO backend {params.backend!r} trains continuous actions. "
            f"Task {params.task!r} is discrete; use hyperparameters.backend=builtin."
        )
    if params.backend == RSL_RL_BACKEND:
        _import_rsl_rl_runner()
    elif params.backend == SKRL_BACKEND:
        _import_skrl_agent()


def _check_env(env: VecEnv, params: PpoParams) -> None:
    if env.num_envs != params.num_envs:
        raise RecipeConfigError(
            f"task {params.task!r} built {env.num_envs} envs, not {params.num_envs}"
        )
    if env.observation_dim < 1 or env.action_dim < 1:
        raise RecipeConfigError(f"task {params.task!r} returned an empty space")


def _check_precision(recipe: Mapping[str, Any]) -> None:
    precision = recipe.get("precision", "fp32")
    if precision != "fp32":
        raise RecipeConfigError(
            f"precision {precision!r} is not available for family 'ppo' in this build"
        )


def _reject_version_only(recipe: Mapping[str, Any]) -> None:
    dataset = recipe.get("dataset")
    if not isinstance(dataset, dict) or "path" not in dataset:
        raise DatasetUnavailable(
            "dataset version ids are not resolved in this build; "
            "set dataset.path to a local dataset directory"
        )


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if not TORCH_AVAILABLE:
        raise RecipeConfigError(_TORCH_MESSAGE)
    if requested == "auto":
        from ohho.hardware import resolve_device

        return resolve_device("auto")
    assert torch is not None
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RecipeConfigError("device 'cuda' is not available on this machine")
        return "cuda"
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RecipeConfigError("device 'mps' is not available on this machine")
        return "mps"
    raise RecipeConfigError(f"device {requested!r} is not available in this build")


def _emit_declared(recipe: Mapping[str, Any], emit: Emit) -> None:
    weights = recipe.get("base_weights")
    if isinstance(weights, dict) and weights:
        emit(
            {
                "event": "base_weights",
                "status": "not_loaded",
                "uri": weights.get("uri", ""),
                "reason": (
                    "the PPO trainer in this build starts from the task "
                    "and does not load base weights"
                ),
            }
        )
    for hook_type in ("compression_hooks", "eval_hooks"):
        hooks = recipe.get(hook_type) or []
        if not isinstance(hooks, list):
            continue
        for hook in hooks:
            if not isinstance(hook, dict):
                continue
            emit(
                {
                    "event": "hook",
                    "hook_type": hook_type,
                    "name": hook.get("name", ""),
                    "status": "skipped",
                    "reason": "hooks are not executed in this build",
                }
            )


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RecipeConfigError(f"hyperparameters.{name} must be an integer >= 1")
    return value


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecipeConfigError("hyperparameters.seed must be an integer")
    return value


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeConfigError(f"hyperparameters.{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise RecipeConfigError(f"hyperparameters.{name} must be finite")
    return number


def _positive_float(name: str, value: Any) -> float:
    number = _finite(name, value)
    if number <= 0:
        raise RecipeConfigError(f"hyperparameters.{name} must be > 0")
    return number


def _nonnegative(name: str, value: Any) -> float:
    number = _finite(name, value)
    if number < 0:
        raise RecipeConfigError(f"hyperparameters.{name} must be >= 0")
    return number


def _unit_float(name: str, value: Any) -> float:
    """Clip range in (0, 1]."""
    number = _finite(name, value)
    if number <= 0 or number > 1:
        raise RecipeConfigError(f"hyperparameters.{name} must be > 0 and <= 1")
    return number


def _closed_unit(name: str, value: Any) -> float:
    number = _finite(name, value)
    if number < 0 or number > 1:
        raise RecipeConfigError(f"hyperparameters.{name} must be between 0 and 1")
    return number


if TORCH_AVAILABLE:
    assert torch is not None and nn is not None

    class PolicyActor(nn.Module):
        """Deterministic actor used for training and ONNX export.

        ``forward`` returns logits for a discrete task and the action mean
        for a continuous task. The log-std parameter is not part of the
        exported graph.
        """

        def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            hidden_size: int,
            discrete: bool,
        ) -> None:
            super().__init__()
            self.observation_dim = observation_dim
            self.action_dim = action_dim
            self.hidden_size = hidden_size
            self.discrete = discrete
            self.net = nn.Sequential(
                nn.Linear(observation_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.head = nn.Linear(hidden_size, action_dim)
            if not discrete:
                self.log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, observation: torch.Tensor) -> torch.Tensor:
            return self.head(self.net(observation))

        def distribution(self, observation: torch.Tensor) -> Any:
            mean_or_logits = self.forward(observation)
            if self.discrete:
                return torch.distributions.Categorical(logits=mean_or_logits)
            std = self.log_std.exp().expand_as(mean_or_logits)
            return torch.distributions.Normal(mean_or_logits, std)

        def action_log_prob(self, dist: Any, action: torch.Tensor) -> torch.Tensor:
            log_prob = dist.log_prob(action)
            if self.discrete:
                return log_prob
            return log_prob.sum(dim=-1)

        def action_entropy(self, dist: Any) -> torch.Tensor:
            entropy = dist.entropy()
            if self.discrete:
                return entropy
            return entropy.sum(dim=-1)

    class ValueCritic(nn.Module):
        """State-value MLP. Not exported."""

        def __init__(self, observation_dim: int, hidden_size: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(observation_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, observation: torch.Tensor) -> torch.Tensor:
            return self.net(observation)

    class _SkrlMeanActor(nn.Module):
        """ONNX wrapper around a skrl policy ``compute`` method."""

        def __init__(self, policy: nn.Module) -> None:
            super().__init__()
            self.policy = policy

        def forward(self, observation: torch.Tensor) -> torch.Tensor:
            mean, _log_std, _extra = self.policy.compute(
                {"states": observation}, "policy"
            )
            return mean

    class RslRlBridge:
        """``rsl_rl`` 2.3 ``VecEnv`` over an ohho task."""

        def __init__(self, env: VecEnv, device: str, tracker: EpisodeTracker) -> None:
            self.task = env
            self.num_envs = env.num_envs
            self.num_actions = env.action_dim
            self.max_episode_length = env.episode_steps
            self.device = torch.device(device)
            self.cfg: dict[str, Any] = {}
            self.episode_length_buf = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._tracker = tracker
            self._obs: torch.Tensor | None = None

        def get_observations(self) -> tuple[torch.Tensor, dict[str, Any]]:
            if self._obs is None:
                self._obs = self._as_tensor(self.task.reset())
            return self._obs, {"observations": {}}

        def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
            self._obs = self._as_tensor(self.task.reset())
            self.episode_length_buf.zero_()
            return self._obs, {"observations": {}}

        def step(
            self, actions: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
            rows = actions.detach().to("cpu").tolist()
            result = self.task.step(rows)
            self._obs = self._as_tensor(result.observations)
            rewards = torch.tensor(
                result.rewards, dtype=torch.float32, device=self.device
            )
            terminated = torch.tensor(
                result.terminated, dtype=torch.float32, device=self.device
            )
            truncated = torch.tensor(
                result.truncated, dtype=torch.float32, device=self.device
            )
            dones = torch.clamp(terminated + truncated, max=1.0)
            self.episode_length_buf += 1
            self.episode_length_buf[dones > 0] = 0
            self._tracker.observe(
                result.rewards, result.terminated, result.truncated, result.successes
            )
            infos = {"observations": {}, "time_outs": truncated}
            return self._obs, rewards, dones, infos

        def _as_tensor(self, rows: list[list[float]]) -> torch.Tensor:
            return torch.tensor(rows, dtype=torch.float32, device=self.device)

    class _Space:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    class SkrlBridge:
        """Gymnasium-style vector env over an ohho task, for skrl 1.4."""

        def __init__(self, env: VecEnv, device: str, tracker: EpisodeTracker) -> None:
            self.task = env
            self.num_envs = env.num_envs
            self.device = torch.device(device)
            self.observation_space = _Space((env.observation_dim,))
            self.action_space = _Space((env.action_dim,))
            self._tracker = tracker
            self._obs: torch.Tensor | None = None

        def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
            self._obs = torch.tensor(
                self.task.reset(), dtype=torch.float32, device=self.device
            )
            return self._obs, {}

        def step(
            self, actions: torch.Tensor
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
        ]:
            rows = actions.detach().to("cpu").tolist()
            result = self.task.step(rows)
            self._obs = torch.tensor(
                result.observations, dtype=torch.float32, device=self.device
            )
            rewards = torch.tensor(
                result.rewards, dtype=torch.float32, device=self.device
            ).reshape(-1, 1)
            terminated = torch.tensor(
                result.terminated, dtype=torch.bool, device=self.device
            ).reshape(-1, 1)
            truncated = torch.tensor(
                result.truncated, dtype=torch.bool, device=self.device
            ).reshape(-1, 1)
            self._tracker.observe(
                result.rewards, result.terminated, result.truncated, result.successes
            )
            return self._obs, rewards, terminated, truncated, {}

    def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> torch.Tensor:
        """GAE-Lambda. ``dones`` is 1 on the transition that ends an episode.

        Tensors are shaped ``(time, env)``. ``last_value`` is shaped ``(env,)``
        and bootstraps the step after the rollout.
        """
        steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(rewards.shape[1], dtype=rewards.dtype, device=rewards.device)
        for time in reversed(range(steps)):
            next_value = last_value if time == steps - 1 else values[time + 1]
            next_nonterminal = 1.0 - dones[time]
            delta = rewards[time] + gamma * next_value * next_nonterminal - values[time]
            gae = delta + gamma * lam * next_nonterminal * gae
            advantages[time] = gae
        return advantages

    def export_actor_onnx(module: nn.Module, path: Path, observation_dim: int) -> None:
        """Write the deterministic actor. Input name is ``observation``."""
        module.eval()
        dummy = torch.zeros(1, observation_dim, dtype=torch.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "input_names": ["observation"],
            "output_names": ["action"],
            "dynamic_axes": {
                "observation": {0: "batch"},
                "action": {0: "batch"},
            },
            "opset_version": 17,
        }
        try:
            parameters = inspect.signature(torch.onnx.export).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "dynamo" in parameters:
            kwargs["dynamo"] = False
        try:
            torch.onnx.export(module, dummy, str(path), **kwargs)
        except Exception as exc:
            raise TrainFailed(
                "PPO actor ONNX export failed. Export needs torch and onnx "
                "(pip install 'ohho-os[humanoid-sim]'). "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def load_actor(path: str | Path) -> PolicyActor:
        """Load a builtin ``actor.pt`` written by this recipe."""
        file = Path(path)
        try:
            blob = torch.load(file, map_location="cpu", weights_only=True)
        except TypeError:
            blob = torch.load(file, map_location="cpu")
        if not isinstance(blob, dict) or blob.get("kind") != "policy_actor":
            raise TrainFailed(f"not a builtin PPO actor checkpoint: {file}")
        actor = PolicyActor(
            observation_dim=int(blob["observation_dim"]),
            action_dim=int(blob["action_dim"]),
            hidden_size=int(blob["hidden_size"]),
            discrete=bool(blob["discrete"]),
        )
        actor.load_state_dict(blob["state_dict"])
        actor.eval()
        return actor

    def exportable_actor(policy: nn.Module) -> nn.Module:
        """Module whose ``forward`` is the deterministic action."""
        if isinstance(policy, PolicyActor):
            return policy
        sequential = getattr(policy, "actor", None)
        if isinstance(sequential, nn.Sequential):
            return sequential
        if callable(getattr(policy, "compute", None)):
            return _SkrlMeanActor(policy)
        return policy

    def _run_with_torch(
        recipe: Mapping[str, Any],
        params: PpoParams,
        env: VecEnv,
        device: str,
        emit: Emit,
    ) -> TrainResult:
        try:
            if params.backend == BUILTIN_BACKEND:
                policy, tracker, final_loss = train_builtin(env, params, device, emit)
            elif params.backend == RSL_RL_BACKEND:
                policy, tracker, final_loss = train_rsl_rl(env, params, device, emit)
            elif params.backend == SKRL_BACKEND:
                policy, tracker, final_loss = train_skrl(env, params, device, emit)
            else:
                raise TrainFailed(f"PPO backend {params.backend!r} was not run")
        except TrainFailed:
            raise
        except Exception as exc:
            raise TrainFailed(f"PPO training failed: {exc}") from exc
        if not isinstance(policy, nn.Module):
            raise TrainFailed("PPO training did not return an actor module")
        output = Path(str(recipe["output"]["dir"])).expanduser()
        return _write_outputs(policy, tracker, final_loss, params, env, output, device)

    def train_builtin(
        env: VecEnv,
        params: PpoParams,
        device: str,
        emit: Emit,
    ) -> tuple[PolicyActor, EpisodeTracker, float]:
        actor = PolicyActor(
            env.observation_dim, env.action_dim, params.hidden_size, env.discrete
        ).to(device)
        critic = ValueCritic(env.observation_dim, params.hidden_size).to(device)
        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=params.lr
        )
        tracker = EpisodeTracker(
            env.num_envs, emit, params.rollout_steps, auto_flush=False
        )
        obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)
        final_loss = 0.0
        for _update in range(params.updates):
            batch = _rollout(actor, critic, env, obs, params, device, tracker)
            obs = batch["next_obs"]
            final_loss = _ppo_update(actor, critic, optimizer, batch, params)
            tracker.flush(final_loss)
        actor.eval()
        return actor, tracker, final_loss

    def _rollout(
        actor: PolicyActor,
        critic: ValueCritic,
        env: VecEnv,
        obs: torch.Tensor,
        params: PpoParams,
        device: str,
        tracker: EpisodeTracker,
    ) -> dict[str, Any]:
        obs_buf = []
        act_buf = []
        logp_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []
        for _step in range(params.rollout_steps):
            with torch.no_grad():
                dist = actor.distribution(obs)
                action = dist.sample()
                log_prob = actor.action_log_prob(dist, action)
                value = critic(obs).squeeze(-1)
            result = env.step(_actions_for_env(action))
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(log_prob)
            val_buf.append(value)
            rew_buf.append(
                torch.tensor(result.rewards, dtype=torch.float32, device=device)
            )
            done = [
                result.terminated[index] or result.truncated[index]
                for index in range(env.num_envs)
            ]
            done_buf.append(torch.tensor(done, dtype=torch.float32, device=device))
            tracker.observe(
                result.rewards, result.terminated, result.truncated, result.successes
            )
            obs = torch.tensor(result.observations, dtype=torch.float32, device=device)
        with torch.no_grad():
            last_value = critic(obs).squeeze(-1)
        rewards = torch.stack(rew_buf)
        values = torch.stack(val_buf)
        dones = torch.stack(done_buf)
        advantages = compute_gae(
            rewards, values, dones, last_value, params.gamma, params.gae_lambda
        )
        returns = advantages + values
        return {
            "obs": torch.stack(obs_buf),
            "actions": torch.stack(act_buf),
            "log_prob": torch.stack(logp_buf),
            "advantages": advantages,
            "returns": returns,
            "next_obs": obs,
        }

    def _ppo_update(
        actor: PolicyActor,
        critic: ValueCritic,
        optimizer: torch.optim.Optimizer,
        batch: Mapping[str, Any],
        params: PpoParams,
    ) -> float:
        obs = batch["obs"].reshape(-1, batch["obs"].shape[-1])
        actions = _flatten_actions(batch["actions"])
        old_log_prob = batch["log_prob"].reshape(-1)
        advantages = batch["advantages"].reshape(-1)
        returns = batch["returns"].reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total = obs.shape[0]
        minibatch = min(params.batch_size, total)
        losses: list[float] = []
        for _epoch in range(params.num_epochs):
            order = torch.randperm(total, device=obs.device)
            for start in range(0, total, minibatch):
                index = order[start : start + minibatch]
                dist = actor.distribution(obs[index])
                new_log_prob = actor.action_log_prob(dist, actions[index])
                entropy = actor.action_entropy(dist).mean()
                ratio = torch.exp(new_log_prob - old_log_prob[index])
                unclipped = ratio * advantages[index]
                clipped = (
                    torch.clamp(ratio, 1.0 - params.clip_coef, 1.0 + params.clip_coef)
                    * advantages[index]
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value = critic(obs[index]).squeeze(-1)
                value_loss = (value - returns[index]).pow(2).mean()
                loss = (
                    policy_loss
                    + _VALUE_COEF * value_loss
                    - params.entropy_coef * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    _MAX_GRAD_NORM,
                )
                optimizer.step()
                losses.append(float(policy_loss.detach()))
        if not losses:
            return 0.0
        return float(sum(losses) / len(losses))

    def _flatten_actions(actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 2:
            return actions.reshape(-1)
        return actions.reshape(-1, actions.shape[-1])

    def _actions_for_env(action: torch.Tensor) -> list[Any]:
        return action.detach().to("cpu").tolist()

    def train_rsl_rl(
        env: VecEnv,
        params: PpoParams,
        device: str,
        emit: Emit,
    ) -> tuple[nn.Module, EpisodeTracker, float]:
        runner_cls = _import_rsl_rl_runner()
        tracker = EpisodeTracker(
            env.num_envs, emit, params.rollout_steps, auto_flush=True
        )
        bridge = RslRlBridge(env, device, tracker)
        try:
            runner = runner_cls(
                bridge,
                rsl_rl_train_config(params),
                log_dir=None,
                device=device,
            )
            runner.learn(num_learning_iterations=params.updates)
        except TrainFailed:
            raise
        except Exception as exc:
            raise TrainFailed(
                "rsl_rl PPO failed. The installed rsl_rl version may be "
                "unsupported; use hyperparameters.backend=builtin. "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        policy = getattr(getattr(runner, "alg", None), "policy", None)
        if not isinstance(policy, nn.Module):
            raise TrainFailed(
                "rsl_rl finished without an actor module. "
                "Use hyperparameters.backend=builtin."
            )
        loss = _optional_loss(getattr(runner, "last_policy_loss", None))
        return policy, tracker, 0.0 if loss is None else loss

    def train_skrl(
        env: VecEnv,
        params: PpoParams,
        device: str,
        emit: Emit,
    ) -> tuple[nn.Module, EpisodeTracker, float]:
        parts = _import_skrl_agent()
        return _train_skrl_with_parts(parts, env, params, device, emit)

    def _train_skrl_with_parts(
        parts: Mapping[str, Any],
        env: VecEnv,
        params: PpoParams,
        device: str,
        emit: Emit,
    ) -> tuple[nn.Module, EpisodeTracker, float]:
        cfg = dict(parts["PPO_DEFAULT_CONFIG"])
        cfg.update(skrl_algorithm_config(params))
        experiment = dict(cfg.get("experiment") or {})
        experiment["experiment_name"] = params.task
        experiment["write_interval"] = 0
        experiment["checkpoint_interval"] = 0
        cfg["experiment"] = experiment
        try:
            models = build_skrl_models(parts, env, params, device)
            memory = parts["RandomMemory"](
                memory_size=params.rollout_steps,
                num_envs=env.num_envs,
                device=device,
            )
            agent = parts["PPO"](
                models=models,
                memory=memory,
                cfg=cfg,
                observation_space=(env.observation_dim,),
                action_space=(env.action_dim,),
                device=device,
            )
            timesteps = params.updates * params.rollout_steps * env.num_envs
            tracker = EpisodeTracker(
                env.num_envs, emit, params.rollout_steps, auto_flush=True
            )
            bridge = SkrlBridge(env, device, tracker)
            if hasattr(agent, "init"):
                agent.init(trainer_cfg={"timesteps": timesteps, "headless": True})
            trainer = parts["SequentialTrainer"](
                cfg={"timesteps": timesteps, "headless": True},
                env=bridge,
                agents=agent,
            )
            trainer.train()
        except TrainFailed:
            raise
        except Exception as exc:
            raise TrainFailed(
                "skrl PPO failed. The installed skrl version may be unsupported; "
                "use hyperparameters.backend=builtin. "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        policy = models.get("policy") if isinstance(models, dict) else None
        if not isinstance(policy, nn.Module):
            raise TrainFailed(
                "skrl finished without an actor module. "
                "Use hyperparameters.backend=builtin."
            )
        loss = _optional_loss(getattr(agent, "last_policy_loss", None))
        return policy, tracker, 0.0 if loss is None else loss

    def build_skrl_models(
        parts: Mapping[str, Any],
        env: VecEnv,
        params: PpoParams,
        device: str,
    ) -> dict[str, nn.Module]:
        """Gaussian policy and value model for skrl 1.4."""
        try:
            model_base = parts["Model"]
            gaussian = parts["GaussianMixin"]
            deterministic = parts["DeterministicMixin"]
        except KeyError as exc:
            raise TrainFailed(
                "skrl model classes are missing. Use hyperparameters.backend=builtin."
            ) from exc
        hidden = params.hidden_size
        obs_dim = env.observation_dim
        act_dim = env.action_dim

        class _Policy(gaussian, model_base):
            def __init__(
                self, observation_space: Any, action_space: Any, device: str
            ) -> None:
                model_base.__init__(self, observation_space, action_space, device)
                gaussian.__init__(
                    self,
                    clip_actions=False,
                    clip_log_std=True,
                    min_log_std=-20.0,
                    max_log_std=2.0,
                    reduction="sum",
                )
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, act_dim),
                )

            def compute(
                self, inputs: Mapping[str, torch.Tensor], role: str
            ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
                del role
                return self.net(inputs["states"]), self.log_std_parameter, {}

        class _Value(deterministic, model_base):
            def __init__(
                self, observation_space: Any, action_space: Any, device: str
            ) -> None:
                model_base.__init__(self, observation_space, action_space, device)
                deterministic.__init__(self)
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, 1),
                )

            def compute(
                self, inputs: Mapping[str, torch.Tensor], role: str
            ) -> tuple[torch.Tensor, dict[str, Any]]:
                del role
                return self.net(inputs["states"]), {}

        space_obs = (obs_dim,)
        space_act = (act_dim,)
        return {
            "policy": _Policy(space_obs, space_act, device),
            "value": _Value(space_obs, space_act, device),
        }

    def _optional_loss(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(float(value)):
            return None
        return float(value)

    def _write_outputs(
        policy: nn.Module,
        tracker: EpisodeTracker,
        final_loss: float,
        params: PpoParams,
        env: VecEnv,
        output: Path,
        device: str,
    ) -> TrainResult:
        tracker.close(final_loss if not tracker.history else None)
        if not tracker.history:
            raise TrainFailed("PPO training did not report episode metrics")
        actor = exportable_actor(policy)
        actor.eval()
        output.mkdir(parents=True, exist_ok=True)
        onnx_path = output / "actor.onnx"
        export_actor_onnx(actor, onnx_path, env.observation_dim)
        kind = "policy_actor" if isinstance(policy, PolicyActor) else "module"
        blob = {
            "format": "ohho.ppo.actor/v1",
            "kind": kind,
            "state_dict": policy.state_dict(),
            "observation_dim": env.observation_dim,
            "action_dim": env.action_dim,
            "hidden_size": params.hidden_size,
            "discrete": bool(env.discrete),
            "task": params.task,
            "backend": params.backend,
        }
        torch.save(blob, output / "actor.pt")
        config = {
            "family": "ppo",
            "task": params.task,
            "backend": params.backend,
            "device": device,
            "discrete": bool(env.discrete),
            "observation_dim": env.observation_dim,
            "action_dim": env.action_dim,
            "episode_steps": env.episode_steps,
            "hidden_size": params.hidden_size,
            "onnx": "actor.onnx",
            "actor": "actor.pt",
            "reward_threshold": _threshold(params.task),
            "hyperparameters": params.as_dict(),
        }
        (output / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        last = tracker.history[-1]
        metrics = {
            "final_loss": float(final_loss),
            "episode_reward": last["episode_reward"],
            "success": last["success"],
            "task": params.task,
            "backend": params.backend,
            "intermediate": tracker.history,
        }
        metrics_path = output / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return TrainResult(
            checkpoint_dir=str(output),
            metrics_path=str(metrics_path),
            device=device,
            base_weights_loaded=False,
        )

    def _threshold(task_id: str) -> float | None:
        task = get_task(task_id)
        fn = getattr(task, "reward_threshold", None)
        if not callable(fn):
            return None
        value = fn()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

else:

    def _run_with_torch(  # type: ignore[misc]
        recipe: Mapping[str, Any],
        params: PpoParams,
        env: VecEnv,
        device: str,
        emit: Emit,
    ) -> TrainResult:
        del recipe, params, env, device, emit
        raise RecipeConfigError(_TORCH_MESSAGE)
