"""Registry of PPO tasks.

A recipe names a task id. The task builds the vectorized environment.
Robot packages add tasks through :func:`register` or the ``ohho.rl_tasks``
entry point, so the PPO recipe itself stays robot-agnostic.

The reference sim task is ``mujoco-inverted-pendulum``. With the default
reward weights its per-step reward is 1, and the solved bar is a mean
episode return of 950 over a 1000-step horizon (the usual
InvertedPendulum-v4 threshold). Training that task to the threshold on a
GPU, and any Isaac Lab task, is a follow-up. ``cartpole`` is the CPU
classic-control task the tests run.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ENTRY_GROUP = "ohho.rl_tasks"
_TASK_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")

CARTPOLE_TASK_ID = "cartpole"
CARTPOLE_REWARD_THRESHOLD = 475.0
CARTPOLE_EPISODE_STEPS = 500

MUJOCO_INVERTED_PENDULUM_TASK_ID = "mujoco-inverted-pendulum"
MUJOCO_INVERTED_PENDULUM_REWARD_THRESHOLD = 950.0
MUJOCO_INVERTED_PENDULUM_EPISODE_STEPS = 1000
MUJOCO_ANGLE_LIMIT = 0.2

ISAAC_VELOCITY_TASK_ID = "isaac-velocity"
ISAAC_UNAVAILABLE = (
    "Isaac Lab task 'isaac-velocity' needs a GPU and Isaac Lab. "
    "GPU training is a follow-up in this build."
)

_GRAVITY = 9.8
_MASS_CART = 1.0
_MASS_POLE = 0.1
_TOTAL_MASS = _MASS_CART + _MASS_POLE
_LENGTH = 0.5
_POLE_MASS_LENGTH = _MASS_POLE * _LENGTH
_FORCE_MAG = 10.0
_TAU = 0.02
_THETA_LIMIT = 12.0 * math.pi / 180.0
_X_LIMIT = 2.4

_REGISTRY: dict[str, tuple[Any, str]] = {}
_LOAD_ERRORS: dict[str, str] = {}
_LOADED = False


class RlTaskError(RuntimeError):
    """A task cannot be built in this process."""


class TaskRegistryError(RuntimeError):
    """A task could not be registered."""


@dataclass
class VecStep:
    """One synchronous step of a vectorized task."""

    observations: list[list[float]]
    rewards: list[float]
    terminated: list[bool]
    truncated: list[bool]
    successes: list[bool]


class VecEnv:
    """Vectorized environment the PPO trainer steps.

    ``step`` auto-resets finished environments. The observation in that
    step is the reset observation. ``terminated`` and ``truncated`` still
    describe the transition that just ended. ``successes`` is true only on
    the step that ends a successful episode.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        observation_dim: int,
        action_dim: int,
        discrete: bool,
        episode_steps: int,
    ) -> None:
        if num_envs < 1:
            raise RlTaskError("num_envs must be >= 1")
        if observation_dim < 1 or action_dim < 1:
            raise RlTaskError("observation_dim and action_dim must be >= 1")
        if episode_steps < 1:
            raise RlTaskError("episode_steps must be >= 1")
        self.num_envs = num_envs
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.episode_steps = episode_steps

    def reset(self) -> list[list[float]]:
        raise NotImplementedError

    def step(self, actions: Sequence[Any]) -> VecStep:
        raise NotImplementedError


def shaped_reward(
    *,
    x: float,
    theta: float,
    reward_weights: Mapping[str, float],
) -> float:
    """Per-step reward used by the built-in CartPole and inverted pendulum.

    ``alive`` is added on every step, including the step that ends the
    episode. ``x_penalty`` and ``theta_penalty`` subtract the squared cart
    position and pole angle. Other keys are ignored here so a plugin task
    can read its own names from the same mapping.
    """
    alive = float(reward_weights.get("alive", 1.0))
    x_penalty = float(reward_weights.get("x_penalty", 0.0))
    theta_penalty = float(reward_weights.get("theta_penalty", 0.0))
    return alive - x_penalty * x * x - theta_penalty * theta * theta


def register(task: Any, *, source: str) -> None:
    """Register ``task``. Repeating the same implementation is a no-op."""
    task_id = _check_task(task)
    current = _REGISTRY.get(task_id)
    if current is not None:
        previous, previous_source = current
        if previous is task or _same_implementation(previous, task):
            return
        raise TaskRegistryError(
            f"rl task {task_id!r} is already registered "
            f"({previous_source}); refusing {source}"
        )
    _REGISTRY[task_id] = (task, source)


def unregister(task_id: str) -> None:
    """Drop one task. Tests use this to remove a temporary plugin."""
    _REGISTRY.pop(task_id, None)


def ensure_loaded() -> None:
    """Register built-in tasks, then ``ohho.rl_tasks`` entry points."""
    global _LOADED
    if _LOADED:
        return
    register_builtins()
    _load_entry_points()
    _LOADED = True


def get_task(task_id: str) -> Any | None:
    """Return the registered task, or ``None`` when the id is absent."""
    ensure_loaded()
    slot = _REGISTRY.get(task_id)
    if slot is None:
        return None
    return slot[0]


def task_ids() -> tuple[str, ...]:
    """Registered task ids, sorted."""
    ensure_loaded()
    return tuple(sorted(_REGISTRY))


def load_error(task_id: str) -> str:
    """Import error for an entry point that failed to load, or ``\"\"``."""
    ensure_loaded()
    return _LOAD_ERRORS.get(task_id, "")


def register_builtins() -> None:
    """Register the tasks that ship with ohho-os."""
    register(CARTPOLE, source="builtin:ohho.train.rl_tasks")
    register(MUJOCO_INVERTED_PENDULUM, source="builtin:ohho.train.rl_tasks")
    register(ISAAC_VELOCITY, source="builtin:ohho.train.rl_tasks")


def task_requires_gpu(task: Any) -> bool:
    """True when this build must refuse the task as a GPU follow-up."""
    fn = getattr(task, "requires_gpu", None)
    if not callable(fn):
        return False
    return bool(fn())


class CartPoleEnv(VecEnv):
    """Classic cart-pole, vectorized, with no gymnasium dependency.

    State is ``(x, x_dot, theta, theta_dot)``. Action ``0`` pushes left and
    ``1`` pushes right. An episode succeeds when it reaches the horizon
    without leaving the position or angle limits. The solved bar for the
    default horizon of 500 is a mean return of 475.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        reward_weights: Mapping[str, float],
        seed: int,
        episode_steps: int,
    ) -> None:
        super().__init__(
            num_envs=num_envs,
            observation_dim=4,
            action_dim=2,
            discrete=True,
            episode_steps=episode_steps,
        )
        self.reward_weights = dict(reward_weights)
        self._rngs = [random.Random(seed + index * 1000) for index in range(num_envs)]
        self._states: list[list[float]] = []
        self._steps: list[int] = []
        self.reset()

    def reset(self) -> list[list[float]]:
        self._states = [self._initial(index) for index in range(self.num_envs)]
        self._steps = [0 for _ in range(self.num_envs)]
        return [list(state) for state in self._states]

    def step(self, actions: Sequence[Any]) -> VecStep:
        if len(actions) != self.num_envs:
            raise RlTaskError(
                f"cartpole expected {self.num_envs} actions, got {len(actions)}"
            )
        observations: list[list[float]] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        successes: list[bool] = []
        for index, action in enumerate(actions):
            pushed = 1 if int(action) >= 1 else 0
            nxt = _cartpole_physics(self._states[index], pushed)
            self._steps[index] += 1
            x, _x_dot, theta, _theta_dot = nxt
            failed = abs(x) > _X_LIMIT or abs(theta) > _THETA_LIMIT
            timed_out = self._steps[index] >= self.episode_steps and not failed
            rewards.append(
                shaped_reward(x=x, theta=theta, reward_weights=self.reward_weights)
            )
            if failed or timed_out:
                terminated.append(failed)
                truncated.append(timed_out)
                successes.append(timed_out)
                self._states[index] = self._initial(index)
                self._steps[index] = 0
                observations.append(list(self._states[index]))
            else:
                self._states[index] = nxt
                terminated.append(False)
                truncated.append(False)
                successes.append(False)
                observations.append(list(nxt))
        return VecStep(observations, rewards, terminated, truncated, successes)

    def _initial(self, index: int) -> list[float]:
        rng = self._rngs[index]
        return [rng.uniform(-0.05, 0.05) for _ in range(4)]


class CartPoleTask:
    """CPU classic-control task. Always available."""

    id = CARTPOLE_TASK_ID

    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def requires_gpu(self) -> bool:
        return False

    def observation_dim(self) -> int:
        return 4

    def action_dim(self) -> int:
        return 2

    def discrete(self) -> bool:
        return True

    def reward_threshold(self) -> float:
        return CARTPOLE_REWARD_THRESHOLD

    def default_episode_steps(self) -> int:
        return CARTPOLE_EPISODE_STEPS

    def make(
        self,
        *,
        num_envs: int,
        reward_weights: Mapping[str, float],
        seed: int,
        episode_steps: int | None = None,
    ) -> CartPoleEnv:
        horizon = CARTPOLE_EPISODE_STEPS if episode_steps is None else episode_steps
        return CartPoleEnv(
            num_envs=num_envs,
            reward_weights=reward_weights,
            seed=seed,
            episode_steps=horizon,
        )


MUJOCO_INVERTED_PENDULUM_XML = """\
<mujoco model="inverted_pendulum">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.02" gravity="0 0 -9.81" integrator="Euler"/>
  <default>
    <joint damping="0.05"/>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2.4 2.4"/>
      <geom name="cart" type="box" size="0.1 0.08 0.05" mass="1"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="pole" type="capsule" fromto="0 0 0 0 0 0.6" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide" joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


class MujocoInvertedPendulumEnv(VecEnv):
    """Cart-pole in MuJoCo. One continuous force on the cart.

    Observation is ``(x, theta, x_dot, theta_dot)``. The episode ends when
    ``|theta|`` exceeds 0.2 rad. Default weights pay 1 per step, so a mean
    return of 950 on the 1000-step horizon is the reference solved bar.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        reward_weights: Mapping[str, float],
        seed: int,
        episode_steps: int,
        model: Any,
        data: list[Any],
    ) -> None:
        super().__init__(
            num_envs=num_envs,
            observation_dim=4,
            action_dim=1,
            discrete=False,
            episode_steps=episode_steps,
        )
        self.reward_weights = dict(reward_weights)
        self._model = model
        self._data = data
        self._rngs = [random.Random(seed + index * 1000) for index in range(num_envs)]
        self._steps = [0 for _ in range(num_envs)]
        self.reset()

    def reset(self) -> list[list[float]]:
        mujoco = _import_mujoco()
        for index, data in enumerate(self._data):
            mujoco.mj_resetData(self._model, data)
            self._write_initial(index, data)
            mujoco.mj_forward(self._model, data)
            self._steps[index] = 0
        return [self._observe(data) for data in self._data]

    def step(self, actions: Sequence[Any]) -> VecStep:
        if len(actions) != self.num_envs:
            raise RlTaskError(
                f"mujoco inverted pendulum expected {self.num_envs} actions, "
                f"got {len(actions)}"
            )
        mujoco = _import_mujoco()
        observations: list[list[float]] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        successes: list[bool] = []
        for index, action in enumerate(actions):
            data = self._data[index]
            command = _scalar_action(action)
            data.ctrl[0] = max(-1.0, min(1.0, command))
            mujoco.mj_step(self._model, data)
            self._steps[index] += 1
            obs = self._observe(data)
            theta = obs[1]
            failed = abs(theta) > MUJOCO_ANGLE_LIMIT or not _finite_obs(obs)
            timed_out = self._steps[index] >= self.episode_steps and not failed
            rewards.append(
                shaped_reward(x=obs[0], theta=theta, reward_weights=self.reward_weights)
            )
            if failed or timed_out:
                terminated.append(failed)
                truncated.append(timed_out)
                successes.append(timed_out)
                mujoco.mj_resetData(self._model, data)
                self._write_initial(index, data)
                mujoco.mj_forward(self._model, data)
                self._steps[index] = 0
                observations.append(self._observe(data))
            else:
                terminated.append(False)
                truncated.append(False)
                successes.append(False)
                observations.append(obs)
        return VecStep(observations, rewards, terminated, truncated, successes)

    def _write_initial(self, index: int, data: Any) -> None:
        rng = self._rngs[index]
        data.qpos[0] = rng.uniform(-0.05, 0.05)
        data.qpos[1] = rng.uniform(-0.05, 0.05)
        data.qvel[0] = rng.uniform(-0.05, 0.05)
        data.qvel[1] = rng.uniform(-0.05, 0.05)

    @staticmethod
    def _observe(data: Any) -> list[float]:
        return [
            float(data.qpos[0]),
            float(data.qpos[1]),
            float(data.qvel[0]),
            float(data.qvel[1]),
        ]


class MujocoInvertedPendulumTask:
    """Reference MuJoCo task. Available when ``mujoco`` imports."""

    id = MUJOCO_INVERTED_PENDULUM_TASK_ID

    def available(self) -> bool:
        return importlib.util.find_spec("mujoco") is not None

    def unavailable_reason(self) -> str:
        return (
            "PPO task 'mujoco-inverted-pendulum' needs mujoco. "
            "Install it with: pip install 'ohho-os[humanoid-sim]'"
        )

    def requires_gpu(self) -> bool:
        return False

    def observation_dim(self) -> int:
        return 4

    def action_dim(self) -> int:
        return 1

    def discrete(self) -> bool:
        return False

    def reward_threshold(self) -> float:
        return MUJOCO_INVERTED_PENDULUM_REWARD_THRESHOLD

    def default_episode_steps(self) -> int:
        return MUJOCO_INVERTED_PENDULUM_EPISODE_STEPS

    def make(
        self,
        *,
        num_envs: int,
        reward_weights: Mapping[str, float],
        seed: int,
        episode_steps: int | None = None,
    ) -> MujocoInvertedPendulumEnv:
        if not self.available():
            raise RlTaskError(self.unavailable_reason())
        mujoco = _import_mujoco()
        try:
            model = mujoco.MjModel.from_xml_string(MUJOCO_INVERTED_PENDULUM_XML)
        except Exception as exc:
            raise RlTaskError(
                f"could not build the mujoco inverted pendulum: {exc}"
            ) from exc
        if int(model.nq) != 2 or int(model.nv) != 2 or int(model.nu) != 1:
            raise RlTaskError(
                "mujoco inverted pendulum has an unexpected model size "
                f"(nq={model.nq}, nv={model.nv}, nu={model.nu})"
            )
        data = [mujoco.MjData(model) for _ in range(num_envs)]
        horizon = (
            MUJOCO_INVERTED_PENDULUM_EPISODE_STEPS
            if episode_steps is None
            else episode_steps
        )
        return MujocoInvertedPendulumEnv(
            num_envs=num_envs,
            reward_weights=reward_weights,
            seed=seed,
            episode_steps=horizon,
            model=model,
            data=data,
        )


class IsaacVelocityTask:
    """Isaac Lab velocity task. Refused in this build.

    Selecting it fails before training with a GPU capability message.
    """

    id = ISAAC_VELOCITY_TASK_ID

    def available(self) -> bool:
        return False

    def unavailable_reason(self) -> str:
        return ISAAC_UNAVAILABLE

    def requires_gpu(self) -> bool:
        return True

    def observation_dim(self) -> int:
        return 1

    def action_dim(self) -> int:
        return 1

    def discrete(self) -> bool:
        return False

    def reward_threshold(self) -> float | None:
        return None

    def default_episode_steps(self) -> int:
        return 1

    def make(
        self,
        *,
        num_envs: int,
        reward_weights: Mapping[str, float],
        seed: int,
        episode_steps: int | None = None,
    ) -> VecEnv:
        del num_envs, reward_weights, seed, episode_steps
        raise RlTaskError(ISAAC_UNAVAILABLE)


CARTPOLE = CartPoleTask()
MUJOCO_INVERTED_PENDULUM = MujocoInvertedPendulumTask()
ISAAC_VELOCITY = IsaacVelocityTask()


def _cartpole_physics(state: Sequence[float], action: int) -> list[float]:
    x, x_dot, theta, theta_dot = (float(value) for value in state)
    force = _FORCE_MAG if action == 1 else -_FORCE_MAG
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + _POLE_MASS_LENGTH * theta_dot**2 * sintheta) / _TOTAL_MASS
    theta_acc = (_GRAVITY * sintheta - costheta * temp) / (
        _LENGTH * (4.0 / 3.0 - _MASS_POLE * costheta**2 / _TOTAL_MASS)
    )
    x_acc = temp - _POLE_MASS_LENGTH * theta_acc * costheta / _TOTAL_MASS
    return [
        x + _TAU * x_dot,
        x_dot + _TAU * x_acc,
        theta + _TAU * theta_dot,
        theta_dot + _TAU * theta_acc,
    ]


def _scalar_action(action: Any) -> float:
    if isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
        if len(action) != 1:
            raise RlTaskError(f"expected one continuous action, got {len(action)}")
        return float(action[0])
    return float(action)


def _finite_obs(obs: Sequence[float]) -> bool:
    return all(math.isfinite(value) for value in obs)


def _import_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:
        raise RlTaskError(
            "PPO task 'mujoco-inverted-pendulum' needs mujoco. "
            "Install it with: pip install 'ohho-os[humanoid-sim]'"
        ) from exc
    return mujoco


def _check_task(task: Any) -> str:
    task_id = getattr(task, "id", None)
    if not isinstance(task_id, str) or _TASK_ID.fullmatch(task_id) is None:
        raise TaskRegistryError(f"rl task id {task_id!r} is not allowed")
    for name in (
        "available",
        "unavailable_reason",
        "observation_dim",
        "action_dim",
        "discrete",
        "make",
    ):
        if not callable(getattr(task, name, None)):
            raise TaskRegistryError(f"{task_id}: {name}() is missing")
    return task_id


def _same_implementation(previous: Any, task: Any) -> bool:
    if previous.id != task.id:
        return False
    if type(previous) is type(task):
        return True
    previous_type = type(previous)
    task_type = type(task)
    return (
        previous_type.__module__ == task_type.__module__
        and previous_type.__qualname__ == task_type.__qualname__
    )


def _load_entry_points() -> None:
    for entry in _entry_points():
        name = str(getattr(entry, "name", ""))
        try:
            loaded = entry.load()
        except Exception as exc:
            _LOAD_ERRORS[name] = f"{type(exc).__name__}: {exc}"
            continue
        if isinstance(loaded, type):
            try:
                loaded = loaded()
            except Exception as exc:
                _LOAD_ERRORS[name] = f"{type(exc).__name__}: {exc}"
                continue
        value = str(getattr(entry, "value", name))
        try:
            register(loaded, source=f"entry-point:{value}")
        except TaskRegistryError as exc:
            _LOAD_ERRORS[name] = str(exc)


def _entry_points() -> list[Any]:
    found: Any = importlib.metadata.entry_points(group=ENTRY_GROUP)
    if isinstance(found, Mapping):
        found = found.get(ENTRY_GROUP, ())
    return list(found)
