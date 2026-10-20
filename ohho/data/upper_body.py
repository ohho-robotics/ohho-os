"""Scripted upper-body episodes on a headless humanoid simulator.

The arm group comes from the spec. Leg joints are not commanded. Each
episode ramps one arm joint toward a target and records the result with
:class:`~ohho.data.recorder.Recorder`. The sim link is wrapped in
:class:`~ohho.safety.SafetyGate` so those joint targets pass
:class:`~ohho.humanoid_policy.HumanoidPolicy`: ``custom`` mode, a sim
target, declared arm limits, and the 0.15 rad step.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Optional, Union

from ..humanoid import HumanoidMode, apply_posture, mode_of
from ..humanoid_sim import resolve_sim_config
from ..registry import RobotSpec, get_spec
from ..robot import Robot
from ..runtime.native import NativeRuntime
from ..safety import AuditLog, SafetyGate
from .humanoid_features import arm_group_name, joint_groups
from .recorder import Recorder

SpecLike = Union[str, RobotSpec]

# Position window for the scripted demo. The class policy still refuses a
# joint that is absent from the session profile, and it still caps the step.
DEMO_ARM_LIMIT = (-1.2, 1.2)


@dataclass
class UpperBodySession:
    """One armed sim humanoid in ``custom`` mode."""

    robot: Robot
    spec: RobotSpec
    arm_group: str
    arm_joints: tuple[str, ...]
    dt: float
    _gate: SafetyGate
    _runtime: NativeRuntime
    _audit_dir: Optional[tempfile.TemporaryDirectory[str]]

    def step(self, dt: Optional[float] = None) -> None:
        sim_step(self.robot, self.dt if dt is None else dt)

    def close(self) -> None:
        try:
            self._gate.disconnect()
        finally:
            self._runtime.stop()
            if self._audit_dir is not None:
                self._audit_dir.cleanup()
                self._audit_dir = None


def primary_arm(spec: RobotSpec) -> tuple[str, tuple[str, ...]]:
    """The first arm group on ``spec`` (left, then right)."""
    groups = joint_groups(spec)
    for side in ("left", "right"):
        name = arm_group_name(spec, side)
        if name and groups.get(name):
            return name, groups[name]
    raise ValueError("spec has no arm group")


def arm_joint_limits(
    spec: RobotSpec,
    span: tuple[float, float] = DEMO_ARM_LIMIT,
) -> dict[str, tuple[float, float]]:
    """Limits for every joint in an arm group. Leg joints are omitted."""
    lo, hi = float(span[0]), float(span[1])
    if not (lo <= hi):
        raise ValueError("arm limit low must be <= high")
    groups = joint_groups(spec)
    limits: dict[str, tuple[float, float]] = {}
    for side in ("left", "right"):
        name = arm_group_name(spec, side)
        if not name:
            continue
        for joint in groups.get(name, ()):
            limits[joint] = (lo, hi)
    return limits


@contextmanager
def open_upper_body_sim(
    robot: SpecLike,
    *,
    joint_limits: Optional[Mapping[str, tuple[float, float]]] = None,
    fps: float = 20.0,
    audit_dir: Optional[str] = None,
) -> Iterator[UpperBodySession]:
    """Connect a headless humanoid sim and arm it for upper-body targets."""
    spec = _resolve_spec(robot)
    group, joints = primary_arm(spec)
    limits = _session_limits(spec, joint_limits)
    if fps <= 0:
        raise ValueError("fps must be > 0")
    owned: Optional[tempfile.TemporaryDirectory[str]] = None
    if audit_dir is None:
        owned = tempfile.TemporaryDirectory(prefix="ohho-vla-")
        audit_dir = owned.name
    from ..adapters.humanoid_mujoco import HumanoidMujocoTransport
    from ..humanoid_policy import caps_from_spec, class_profile

    # Modest gains. Torque is integrated explicitly; a stiff derivative on a
    # light arm makes qacc blow up at the model timestep.
    config = replace(resolve_sim_config(spec), kp=30.0, kd=1.5)
    inner = HumanoidMujocoTransport(spec, config=config)
    profile = replace(class_profile(caps_from_spec(spec)), joint_limits=limits)
    gate = SafetyGate(
        inner,
        spec,
        profile=profile,
        audit=AuditLog(audit_dir),
        environ={},
        isatty=lambda: False,
        input_fn=_refuse_prompt,
        auto_tick=False,
    )
    runtime = NativeRuntime()
    session: Optional[UpperBodySession] = None
    try:
        gate.connect()
        bot = Robot(spec, gate, runtime)
        sim_step(bot, 1.0 / fps)
        if not gate.arm():
            reason = gate.latch_reason or "refused"
            raise RuntimeError(f"upper-body sim did not arm ({reason})")
        apply_posture(bot, "custom")
        sim_step(bot, 1.0 / fps)
        if mode_of(bot.telemetry()) is not HumanoidMode.CUSTOM:
            raise RuntimeError("upper-body sim did not enter custom mode")
        session = UpperBodySession(
            robot=bot,
            spec=spec,
            arm_group=group,
            arm_joints=joints,
            dt=1.0 / fps,
            _gate=gate,
            _runtime=runtime,
            _audit_dir=owned,
        )
        yield session
    finally:
        if session is not None:
            session.close()
        else:
            try:
                gate.disconnect()
            finally:
                runtime.stop()
                if owned is not None:
                    owned.cleanup()


def record_upper_body_episodes(
    robot: SpecLike,
    output_dir: str,
    *,
    episodes: int = 2,
    steps: int = 8,
    fps: float = 20.0,
    repo_id: str = "local/humanoid-upper-body",
    task: str = "reach",
    target: float = 0.25,
    target_step: float = 0.1,
    rate: float = 0.1,
    joint_limits: Optional[Mapping[str, tuple[float, float]]] = None,
) -> str:
    """Record ``episodes`` scripted reaches and return the dataset directory.

    The commanded joint moves by at most ``rate`` radians per frame, which
    stays inside the humanoid step cap. Body velocity stays zero. The other
    arm joints and every leg joint stay at the pose the simulator is holding.
    """
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if rate <= 0:
        raise ValueError("rate must be > 0")
    with open_upper_body_sim(robot, joint_limits=joint_limits, fps=fps) as session:
        bot = session.robot
        recorder = Recorder(
            bot,
            repo_id=repo_id,
            fps=fps,
            camera_fn=lambda: None,
            arm_action="arm_joints",
        )
        width = len(session.arm_joints)
        for index in range(episodes):
            goal = float(target) + float(index) * float(target_step)
            _settle(session, width)
            recorder.start_episode(task=task)
            for frame in range(steps):
                commanded = _approach(frame, goal, rate)
                positions = [commanded] + [0.0] * (width - 1)
                bot.move_joints(session.arm_group, positions)
                session.step()
                recorder.capture_frame()
            recorder.stop_episode()
        return recorder.save(output_dir)


def sim_step(robot: Robot, dt: float) -> None:
    """Advance the simulator under a gate, or the transport itself."""
    transport = robot.transport
    inner = getattr(transport, "_inner", None)
    target = inner if inner is not None else transport
    step = getattr(target, "step", None)
    if not callable(step):
        raise RuntimeError("this link has no step()")
    step(dt)


def _settle(session: UpperBodySession, width: int) -> None:
    zeros = [0.0] * width
    for _ in range(8):
        session.robot.move_joints(session.arm_group, zeros)
        session.step()


def _approach(frame: int, goal: float, rate: float) -> float:
    walked = rate * float(frame + 1)
    if goal >= 0:
        return min(goal, walked)
    return max(goal, -walked)


def _session_limits(
    spec: RobotSpec,
    joint_limits: Optional[Mapping[str, tuple[float, float]]],
) -> dict[str, tuple[float, float]]:
    allowed = arm_joint_limits(spec)
    if not allowed:
        raise ValueError("spec has no arm group")
    if joint_limits is None:
        return allowed
    selected: dict[str, tuple[float, float]] = {}
    for name, lim in joint_limits.items():
        if name not in allowed:
            continue
        lo, hi = float(lim[0]), float(lim[1])
        if lo > hi:
            raise ValueError(f"joint limit for {name} is reversed")
        selected[name] = (lo, hi)
    missing = [joint for joint in allowed if joint not in selected]
    if missing:
        raise ValueError(
            "arm joint limits must cover every arm joint; missing " + ", ".join(missing)
        )
    return selected


def _resolve_spec(robot: SpecLike) -> RobotSpec:
    if isinstance(robot, RobotSpec):
        return robot
    return get_spec(robot)


def _refuse_prompt(_text: str) -> str:
    raise RuntimeError("upper-body sim does not prompt")


__all__ = [
    "DEMO_ARM_LIMIT",
    "UpperBodySession",
    "arm_joint_limits",
    "open_upper_body_sim",
    "primary_arm",
    "record_upper_body_episodes",
    "sim_step",
]
