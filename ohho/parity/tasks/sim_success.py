"""Closed-loop sim success rate on a fixed seed set.

Each seed is one episode of :func:`ohho.sim_loop.evaluate_policy_sim` on the
``sim`` robot. The goal is a point ``GOAL_DISTANCE`` metres from the origin
at angle ``seed * golden angle``. The policy observation is the sim state
with ``(goal_x, goal_y, seed)`` appended. A seed succeeds when every step
completes and the final pose is within ``SUCCESS_RADIUS`` of that goal.

:func:`pursuit_action` is a deterministic policy that reaches the goal in
``SIM_STEPS`` steps by commanding body velocity ``PURSUIT_SPEED`` toward it.
The sim loop's step is 0.1 s, so eight steps at 0.5 m/s travel 0.4 m.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ohho.parity.errors import ParityError
from ohho.parity.tasks.tolerances import tolerance_for
from ohho.parity.tasks.types import Comparison, TaskContext, as_floats

SIM_STEPS = 8
SIM_DT = 0.1
PURSUIT_SPEED = 0.5
GOAL_DISTANCE = PURSUIT_SPEED * SIM_DT * SIM_STEPS
SUCCESS_RADIUS = 0.02
_GOLDEN = math.pi * (3.0 - math.sqrt(5.0))


def goal_for_seed(seed: int) -> tuple[float, float]:
    """World-frame goal for one fixed seed. The robot starts at the origin."""
    angle = seed * _GOLDEN
    return (
        GOAL_DISTANCE * math.cos(angle),
        GOAL_DISTANCE * math.sin(angle),
    )


def pursuit_action(observation: Sequence[float]) -> list[float]:
    """Body-frame velocity that reaches the goal appended to ``observation``.

    The last three values are ``goal_x``, ``goal_y``, and the seed. The
    returned action is ``[vx, vy, 0, 0]``.
    """
    values = as_floats(observation, "sim observation")
    if len(values) < 3:
        raise ParityError("sim observation is missing the goal.")
    goal_x, goal_y = values[-3], values[-2]
    distance = math.hypot(goal_x, goal_y)
    if distance < 1e-12:
        return [0.0, 0.0, 0.0, 0.0]
    scale = PURSUIT_SPEED / distance
    return [goal_x * scale, goal_y * scale, 0.0, 0.0]


class SimSuccessEvaluator:
    """Fraction of fixed seeds whose closed-loop episode reaches the goal."""

    name = "sim_success"

    def run(self, context: TaskContext) -> list[Comparison]:
        if not context.seeds:
            raise ParityError("sim success needs at least one seed.")
        steps = context.sim_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ParityError("sim steps must be an integer >= 1.")
        reference, ref_latency = _rate(
            context.reference, context.seeds, steps, context.robot_id
        )
        optimized, opt_latency = _rate(
            context.optimized, context.seeds, steps, context.robot_id
        )
        context.latencies_ms.extend(ref_latency)
        context.latencies_ms.extend(opt_latency)
        return [
            Comparison(
                task=self.name,
                metric="sim_success",
                reference=reference,
                optimized=optimized,
                tolerance=tolerance_for(context.tolerances, "sim_success"),
                direction="higher_is_better",
                unit="fraction",
                samples=len(context.seeds),
                seeds=tuple(context.seeds),
            )
        ]


def _rate(
    policy, seeds: tuple[int, ...], steps: int, robot_id: str
) -> tuple[float, list[float]]:
    hits = 0
    latencies: list[float] = []
    for seed in seeds:
        ok, samples = _one(policy, seed, steps, robot_id)
        hits += int(ok)
        latencies.extend(samples)
    return hits / len(seeds), latencies


def _one(policy, seed: int, steps: int, robot_id: str) -> tuple[bool, list[float]]:
    from ohho.sim_loop import evaluate_policy_sim

    goal_x, goal_y = goal_for_seed(seed)
    pose = _Pose()
    client = _Client(policy, seed, goal_x, goal_y)
    completed, latencies = evaluate_policy_sim(
        client,
        robot_id=robot_id,
        eval_steps=steps,
        instruction=f"task-accuracy seed={seed}",
        probe=pose,
    )
    if pose.xy is None:
        return False, list(latencies)
    error = math.hypot(pose.xy[0] - goal_x, pose.xy[1] - goal_y)
    return completed == steps and error <= SUCCESS_RADIUS, list(latencies)


class _Pose:
    def __init__(self) -> None:
        self.xy: tuple[float, float] | None = None

    def __call__(self, step: int, robot: Any, action: list[float]) -> None:
        del step, action
        odom = robot.telemetry().odom
        if odom is None:
            return
        self.xy = (float(odom.x), float(odom.y))


class _Client:
    """Minimal ``/predict`` client for :func:`evaluate_policy_sim`."""

    def __init__(self, policy, seed: int, goal_x: float, goal_y: float) -> None:
        self._policy = policy
        self._seed = seed
        self._goal_x = goal_x
        self._goal_y = goal_y

    def post(self, path: str, json: dict[str, Any] | None = None) -> _Response:
        del path
        body = json or {}
        config = body.get("config") if isinstance(body, dict) else None
        state = config.get("state") if isinstance(config, dict) else None
        if not isinstance(state, list):
            raise ParityError("sim loop state is missing.")
        observation = as_floats(state, "sim state")
        observation.extend((self._goal_x, self._goal_y, float(self._seed)))
        action = _drive_action(self._policy(observation))
        return _Response({"action": {"vector": action}})


class _Response:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


def _drive_action(raw: object) -> list[float]:
    if not isinstance(raw, list):
        raise ParityError("sim policy must return a list of numbers.")
    values = as_floats(raw, "sim action")
    if len(values) < 3:
        values = values + [0.0] * (3 - len(values))
    return values


evaluator = SimSuccessEvaluator()
