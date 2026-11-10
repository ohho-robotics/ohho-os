"""Seeded checkpoint evaluation on the in-process kinematic sim.

Each episode places the ``sim`` robot on a Proof scenario start pose, drives
toward the goal with a pursuit controller, and scores contact and timeout.
Gains come from the checkpoint digest and the episode seed, so the same
checkpoint and seed repeat the same success rate. Scenario geometry is read
from the suite file at run time.

Scoring distances match the Proof nav regression checker: 0.30 m in XY and
0.50 rad in yaw. The goal is scored after every integration, including the
last. Precedence is collision, then success, then timeout. This runner does
not launch an external simulator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ohho.adapters.sim import SimTransport
from ohho.registry import get_spec
from ohho.schema import Odometry, Velocity

from .errors import EvalError
from .report import SCHEMA_ID, build_report
from .scenarios import ProofScenario, load_scenarios, resolve_scenario_file

JOB_TYPE = "eval"
BACKEND = "kinematic-sim"

# Published by the Proof nav regression checker, not copied from a scenario.
GOAL_XY_TOLERANCE_M = 0.30
GOAL_YAW_TOLERANCE_RAD = 0.50
BODY_RADIUS_M = 0.18
DT_S = 0.05
MAX_STEPS = 400
CRUISE_MPS = 0.6


def evaluate_after_train(
    checkpoint: str | os.PathLike[str],
    suite: str,
    *,
    enabled: bool = False,
    episodes: int = 5,
    seed: int = 0,
    scenarios: str | os.PathLike[str] | None = None,
    videos: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Eval hook for the end of a training run or sweep trial.

    Returns ``None`` when ``enabled`` is false, so a caller can always invoke
    this and turn it on with a flag. This module does not import training or
    sweep code; those packages call this function.
    """
    if not enabled:
        return None
    return run_eval(
        checkpoint,
        suite,
        episodes=episodes,
        seed=seed,
        scenarios=scenarios,
        videos=videos,
    )


def run_eval_job(job: Mapping[str, Any]) -> dict[str, Any] | None:
    """Run one sweep job whose ``type`` is ``eval`` (see :data:`JOB_TYPE`).

    Required keys are ``checkpoint`` and ``suite``. Optional keys: ``episodes``,
    ``seed``, ``scenarios``, ``videos``, and ``enabled`` (default true).
    """
    kind = job.get("type", JOB_TYPE)
    if kind not in {JOB_TYPE, SCHEMA_ID}:
        raise EvalError(f"eval job type must be {JOB_TYPE!r}, got {kind!r}")
    try:
        checkpoint = job["checkpoint"]
        suite = job["suite"]
    except KeyError as exc:
        raise EvalError(f"eval job is missing {exc.args[0]!r}") from exc
    return evaluate_after_train(
        checkpoint,
        str(suite),
        enabled=bool(job.get("enabled", True)),
        episodes=int(job.get("episodes", 5)),
        seed=int(job.get("seed", 0)),
        scenarios=job.get("scenarios"),
        videos=job.get("videos"),
    )


def run_eval(
    checkpoint: str | os.PathLike[str],
    suite: str,
    *,
    episodes: int = 5,
    seed: int = 0,
    scenarios: str | os.PathLike[str] | None = None,
    videos: str | os.PathLike[str] | None = None,
    out: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run ``episodes`` seeded rollouts per scenario and return an ohho.eval/v1 report."""
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise EvalError("episodes must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvalError("seed must be an integer")
    checkpoint_path = Path(checkpoint).expanduser()
    digest = checkpoint_sha256(checkpoint_path)
    gains = controller_gains(checkpoint_path, digest)
    path, source = resolve_scenario_file(suite, scenarios)
    loaded = load_scenarios(path)
    video_dir = Path(videos).expanduser() if videos else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    spec = get_spec("sim")
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(loaded):
        for episode_index in range(episodes):
            episode_seed = seed + scenario_index * episodes + episode_index
            outcome = _rollout(
                scenario,
                spec=spec,
                digest=digest,
                seed=episode_seed,
                base_speed=gains[0],
                heading_gain=gains[1],
                max_lin=float(spec.max_lin),
                max_ang=float(spec.max_ang),
            )
            video_path = None
            if video_dir is not None:
                video_path = _write_trace(
                    video_dir, scenario.id, episode_seed, outcome["trace"]
                )
            rows.append(
                {
                    "scenario_id": scenario.id,
                    "seed": episode_seed,
                    "success": outcome["success"],
                    "length": outcome["length"],
                    "failure_mode": outcome["failure_mode"],
                    "video_path": video_path,
                }
            )
    report = build_report(
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest,
        suite=suite,
        scenario_source=source,
        scenario_ids=[scenario.id for scenario in loaded],
        seed=seed,
        backend=BACKEND,
        episodes_per_scenario=episodes,
        episodes=rows,
    )
    if out is not None:
        destination = Path(out).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def checkpoint_sha256(path: Path) -> str:
    """Digest a checkpoint file, or a directory of checkpoint files."""
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise EvalError(f"checkpoint not found: {path}")
    files = [item for item in path.rglob("*") if item.is_file()]
    if not files:
        raise EvalError(f"checkpoint directory is empty: {path}")
    digest = hashlib.sha256()
    for child in sorted(files, key=lambda item: item.relative_to(path).as_posix()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def controller_gains(path: Path, digest: str) -> tuple[float, float]:
    """Return ``(speed_mps, heading_gain)`` before the per-episode seed scale.

    A JSON checkpoint may set ``eval.speed_mps`` and ``eval.heading_gain``.
    Any other checkpoint uses a cruise speed derived from its digest.
    """
    speed = CRUISE_MPS * (0.5 + 0.5 * _unit(digest, 0))
    gain = 2.5
    block = _eval_block(path)
    if block is not None:
        if "speed_mps" in block:
            speed = _finite_float(block["speed_mps"], "eval.speed_mps")
        if "heading_gain" in block:
            gain = _finite_float(block["heading_gain"], "eval.heading_gain")
    if not 0.05 <= speed <= 2.0:
        raise EvalError("eval.speed_mps must be between 0.05 and 2.0")
    if not 0.1 <= gain <= 8.0:
        raise EvalError("eval.heading_gain must be between 0.1 and 8.0")
    return speed, gain


def _eval_block(path: Path) -> dict[str, Any] | None:
    candidate = path if path.is_file() else path / "checkpoint.json"
    if not candidate.is_file() or candidate.suffix.lower() != ".json":
        return None
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get("eval")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise EvalError("checkpoint eval block must be an object")
    return block


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvalError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise EvalError(f"{label} must be finite")
    return number


def _rollout(
    scenario: ProofScenario,
    *,
    spec: Any,
    digest: str,
    seed: int,
    base_speed: float,
    heading_gain: float,
    max_lin: float,
    max_ang: float,
) -> dict[str, Any]:
    scale = 0.85 + 0.15 * _unit(digest, seed)
    speed = min(max_lin, base_speed * scale)
    sim = SimTransport(spec)
    sim._odom = Odometry(
        scenario.start_x,
        scenario.start_y,
        scenario.start_yaw,
        0.0,
        0.0,
        0.0,
    )
    trace: list[list[float]] = [
        [_round(scenario.start_x), _round(scenario.start_y), _round(scenario.start_yaw)]
    ]
    # Score the pose after every integration, including the last. A step whose
    # predicted pose intersects the expanded obstacle is collision and is not
    # applied, so contact wins over a goal that the same pose would have
    # entered. Otherwise an integrated pose inside the goal is success.
    # Reaching the step horizon with neither result is timeout.
    # length is the integration count on success (0 when the start pose is
    # already inside the goal), the contacting step on collision, and
    # MAX_STEPS on timeout.
    failure: str | None = None
    length = 0
    if not _reached_goal(sim._odom.x, sim._odom.y, sim._odom.theta, scenario):
        for step in range(1, MAX_STEPS + 1):
            x = sim._odom.x
            y = sim._odom.y
            theta = sim._odom.theta
            vx, wz, _ = _command(
                x,
                y,
                theta,
                scenario,
                speed=speed,
                heading_gain=heading_gain,
                max_ang=max_ang,
            )
            nxt_x, nxt_y, _nxt_theta = _integrate(x, y, theta, vx, wz, DT_S)
            if _hits_obstacle(nxt_x, nxt_y, scenario):
                length = step
                failure = "collision"
                break
            sim.send_velocity(Velocity(vx, 0.0, wz))
            sim.step(DT_S)
            trace.append(
                [_round(sim._odom.x), _round(sim._odom.y), _round(sim._odom.theta)]
            )
            if _reached_goal(sim._odom.x, sim._odom.y, sim._odom.theta, scenario):
                length = step
                break
        else:
            length = MAX_STEPS
            failure = "timeout"
    return {
        "success": failure is None,
        "length": length,
        "failure_mode": failure,
        "trace": trace,
    }


def _reached_goal(
    x: float,
    y: float,
    theta: float,
    scenario: ProofScenario,
) -> bool:
    """True when XY and yaw are inside the Proof goal tolerance."""
    dx = scenario.goal_x - x
    dy = scenario.goal_y - y
    if math.hypot(dx, dy) > GOAL_XY_TOLERANCE_M:
        return False
    return abs(_wrap(scenario.goal_yaw - theta)) <= GOAL_YAW_TOLERANCE_RAD


def _command(
    x: float,
    y: float,
    theta: float,
    scenario: ProofScenario,
    *,
    speed: float,
    heading_gain: float,
    max_ang: float,
) -> tuple[float, float, bool]:
    dx = scenario.goal_x - x
    dy = scenario.goal_y - y
    if _reached_goal(x, y, theta, scenario):
        return 0.0, 0.0, True
    if math.hypot(dx, dy) <= GOAL_XY_TOLERANCE_M:
        yaw_err = _wrap(scenario.goal_yaw - theta)
        return 0.0, _clamp(heading_gain * yaw_err, -max_ang, max_ang), False
    heading_err = _wrap(math.atan2(dy, dx) - theta)
    vx = speed * max(0.0, math.cos(heading_err))
    wz = _clamp(heading_gain * heading_err, -max_ang, max_ang)
    return vx, wz, False


def _integrate(
    x: float,
    y: float,
    theta: float,
    vx: float,
    wz: float,
    dt: float,
) -> tuple[float, float, float]:
    x += vx * math.cos(theta) * dt
    y += vx * math.sin(theta) * dt
    theta = _wrap(theta + wz * dt)
    return x, y, theta


def _hits_obstacle(x: float, y: float, scenario: ProofScenario) -> bool:
    dx = x - scenario.obstacle_x
    dy = y - scenario.obstacle_y
    yaw = scenario.obstacle_yaw
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    half_x = scenario.size_x / 2.0 + BODY_RADIUS_M
    half_y = scenario.size_y / 2.0 + BODY_RADIUS_M
    return abs(local_x) <= half_x and abs(local_y) <= half_y


def _write_trace(
    directory: Path,
    scenario_id: str,
    seed: int,
    trace: list[list[float]],
) -> str:
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scenario_id)
    path = directory / f"{safe_id}-seed{seed}.trace.json"
    payload = {"schema": "ohho.eval.trace/v1", "poses": trace}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path.resolve().as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit(digest: str, seed: int) -> float:
    mixed = int(digest[:8], 16) ^ (seed & 0xFFFFFFFF)
    raw = (mixed * 1103515245 + 12345) & 0x7FFFFFFF
    return raw / float(0x7FFFFFFF)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _round(value: float) -> float:
    return round(value, 6)
