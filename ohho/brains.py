"""Agent-engine wiring — bridges ``ohho.Robot`` to ``agent_engine.AgentHarness``.

This module is the M2 "real brain": it builds a :class:`ToolRegistry` from the
robot's capabilities, a :class:`RobotPerceptor` that turns ``Robot.telemetry()``
into a :class:`WorldState`, and a :class:`ClaudeToolCallingReasoner` over a
:class:`ReasoningRouter` (cloud Claude when ``ANTHROPIC_API_KEY`` + ``anthropic``
are present, an echo fallback otherwise). Everything is wrapped behind the
:class:`Brain` protocol so ``Agent(bot).run(goal)`` works unchanged.

Heavy imports (``agent_engine`` needs numpy, ``anthropic`` is optional) are lazy:
importing ``ohho.agent`` never touches this module. It loads only when
``_default_brain()`` tries to upgrade from ``ScriptedBrain``.
"""

from __future__ import annotations

import json
from typing import List, Optional

from . import capabilities as caps
from .planner import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TRAVEL_M,
    PLANNER_SYSTEM_TEMPLATE,
    agent_max_linear_speed,
    build_robot_card,
    goal_blockers,
    wrap_drive_with_travel_limit,
)
from .robot import Robot


def harness_available() -> bool:
    """True when the full agent stack can actually be imported.

    ``agent_engine`` lives in this repo and needs numpy — merely *finding* the
    package is not enough (its import explodes without numpy), so this probe
    performs the real imports and treats any failure as "not available".
    """
    try:
        import agent_engine.core.harness  # noqa: F401
        import agent_engine.core.tools  # noqa: F401
        import agent_engine.reasoners.claude_tool_caller  # noqa: F401
    except Exception:
        return False
    return True


def build_tool_registry(
    robot: Robot,
    *,
    navigator=None,
    memory=None,
    perceptor=None,
    logger=None,
    travel_origin: Optional[tuple[float, float]] = None,
    max_travel_m: float = DEFAULT_MAX_TRAVEL_M,
):
    """Build an ``agent_engine.ToolRegistry`` from the robot's capabilities.

    Each capability maps to a callable tool the LLM can invoke:
      - ``base.drive`` → ``drive(vx, vy, w)``
      - ``manipulation`` → ``move_joints(j0, j1, …)``
      - always → ``stop``, ``emergency_stop``, ``get_telemetry``, ``get_status``
      - with a ``navigator`` → ``navigate_to(x, y)`` (+ ``explore()`` when the
        robot has ``perception.scan``)
      - with a ``memory`` → ``where_is``, ``objects_near``, ``remember_note``
      - with a ``perceptor`` + ``memory`` → ``look_around()``
    """
    from agent_engine.core.tools import ToolParam, ToolRegistry, ToolSpec

    reg = ToolRegistry()

    if robot.has(caps.BASE_DRIVE):

        def _drive(vx=0.0, vy=0.0, w=0.0):
            return robot.drive(vx=float(vx), vy=float(vy), w=float(w))

        drive_handler = _drive
        if travel_origin is not None:
            drive_handler = wrap_drive_with_travel_limit(
                robot,
                drive_handler,
                origin=travel_origin,
                max_travel_m=max_travel_m,
                log=logger,
            )
        lin_cap = agent_max_linear_speed(robot)
        reg.register(
            ToolSpec(
                name="drive",
                description=(
                    "Drive the robot base. vx=forward m/s, vy=left m/s "
                    f"(holonomic only), w=angular rad/s. Clamped to {lin_cap} m/s."
                    + (
                        " Quadruped: call stand first; motion is ignored while lying."
                        if robot.has(caps.LEGGED_POSE)
                        else ""
                    )
                ),
                params=[
                    ToolParam("vx", "forward velocity m/s", type="number"),
                    ToolParam(
                        "vy", "lateral velocity m/s (0 if not holonomic)", type="number"
                    ),
                    ToolParam("w", "angular velocity rad/s", type="number"),
                ],
                handler=drive_handler,
                low_level=True,
            )
        )

    if robot.has(caps.LEGGED_POSE):
        from .safety import EULER_LIMIT_RAD

        lim = EULER_LIMIT_RAD
        for name, desc, method in (
            ("stand", "Stand up on four feet (required before driving).", robot.stand),
            (
                "lie_down",
                "Lie flat on the ground (use when the goal is done).",
                robot.lie_down,
            ),
            ("sit", "Sit pose.", robot.sit),
            ("recover", "Recovery stand after a fall or awkward pose.", robot.recover),
        ):
            reg.register(
                ToolSpec(
                    name=name,
                    description=desc,
                    params=[],
                    handler=lambda m=method, n=name: (
                        (m() and f"{n}: ok") or f"{n}: refused"
                    ),
                )
            )
        reg.register(
            ToolSpec(
                name="set_body_pose",
                description=(f"Set body roll/pitch/yaw (rad), each clamped to ±{lim}."),
                params=[
                    ToolParam("roll", "roll rad", type="number"),
                    ToolParam("pitch", "pitch rad", type="number"),
                    ToolParam("yaw", "yaw rad", type="number"),
                ],
                handler=lambda roll=0.0, pitch=0.0, yaw=0.0: (
                    "set_body_pose: ok"
                    if robot.set_body_pose(float(roll), float(pitch), float(yaw))
                    else "set_body_pose: refused"
                ),
            )
        )

    if robot.has(caps.MANIPULATION):
        n = len(robot.spec.joint_names)
        reg.register(
            ToolSpec(
                name="move_joints",
                description=(
                    f"Command the {n}-DOF arm joints (radians, in spec order: "
                    f"{', '.join(robot.spec.joint_names)})."
                ),
                params=[
                    ToolParam(
                        f"j{i}",
                        f"joint {i} target (rad)",
                        type="number",
                    )
                    for i in range(n)
                ],
                handler=lambda **kwargs: robot.move_joints(
                    [kwargs[f"j{i}"] for i in range(n)]
                ),
            )
        )

    reg.register(
        ToolSpec(
            name="stop",
            description="Stop the base (zero velocity).",
            params=[],
            handler=lambda: robot.stop(),
        )
    )
    reg.register(
        ToolSpec(
            name="emergency_stop",
            description="Emergency stop — zero all motion and disable actuators.",
            params=[],
            handler=lambda: robot.emergency_stop(),
            low_level=True,
        )
    )
    reg.register(
        ToolSpec(
            name="get_telemetry",
            description="Read the current telemetry (odometry, joints, battery).",
            params=[],
            handler=lambda: _telemetry_str(robot),
        )
    )
    reg.register(
        ToolSpec(
            name="get_status",
            description="Read the transport connection status.",
            params=[],
            handler=lambda: _status_str(robot),
        )
    )

    if navigator is not None and robot.has(caps.BASE_DRIVE):
        reg.register(
            ToolSpec(
                name="navigate_to",
                description=(
                    "Autonomously navigate the base to a world-frame point "
                    "(A* over the live obstacle map, replanning on the way)."
                ),
                params=[
                    ToolParam("x", "goal x, metres (world frame)", type="number"),
                    ToolParam("y", "goal y, metres (world frame)", type="number"),
                ],
                handler=lambda x=0.0, y=0.0: str(
                    navigator.navigate_to(float(x), float(y), timeout=20.0)
                ),
            )
        )
        if robot.has(caps.PERCEPTION_SCAN):
            reg.register(
                ToolSpec(
                    name="explore",
                    description=(
                        "Frontier exploration: spin-scan, then drive to unknown "
                        "space repeatedly, building the map. Returns a log."
                    ),
                    params=[],
                    handler=lambda: "\n".join(
                        navigator.explore(max_frontiers=3, timeout_per=15.0)
                    ),
                )
            )

    if memory is not None:
        reg.register(
            ToolSpec(
                name="where_is",
                description=(
                    "Recall an object from spatio-temporal memory: last known "
                    "position and how long ago it was seen."
                ),
                params=[ToolParam("label", "object label, e.g. 'cup'", type="string")],
                handler=lambda label="": _where_str(memory, str(label)),
            )
        )
        reg.register(
            ToolSpec(
                name="objects_near",
                description="List remembered objects within a radius of a point.",
                params=[
                    ToolParam("x", "x, metres", type="number"),
                    ToolParam("y", "y, metres", type="number"),
                    ToolParam("radius", "search radius, metres", type="number"),
                ],
                handler=lambda x=0.0, y=0.0, radius=1.5: _near_str(
                    memory, float(x), float(y), float(radius)
                ),
            )
        )
        reg.register(
            ToolSpec(
                name="remember_note",
                description="Store a free-form note in the robot's memory timeline.",
                params=[ToolParam("text", "the note", type="string")],
                handler=lambda text="": memory.note(str(text)) or "noted",
            )
        )

    if perceptor is not None and memory is not None:
        reg.register(
            ToolSpec(
                name="look_around",
                description=(
                    "Perceive the surroundings now, store what is seen into "
                    "spatial memory, and report the detections."
                ),
                params=[],
                handler=lambda: _look_str(perceptor, memory),
            )
        )

    return reg


def _where_str(memory, label: str) -> str:
    e = memory.where_is(label)
    if e is None:
        return f"no memory of '{label}'"
    return (
        f"{e.label} last seen at ({e.x:.2f}, {e.y:.2f}), {e.age():.0f}s ago "
        f"(seen {e.count}×)"
    )


def _near_str(memory, x: float, y: float, radius: float) -> str:
    found = memory.near(x, y, radius)
    if not found:
        return f"nothing remembered within {radius:.1f} m of ({x:.2f}, {y:.2f})"
    return "; ".join(f"{e.label} at ({e.x:.2f}, {e.y:.2f})" for e in found)


def _look_str(perceptor, memory) -> str:
    from .perception import remember_detections

    detections = perceptor.look()
    if not detections:
        return "nothing detected"
    remember_detections(memory, detections)
    return "; ".join(str(d) for d in detections)


def _telemetry_str(robot: Robot) -> str:
    t = robot.telemetry()
    parts = []
    if t.odom:
        parts.append(
            f"odom(x={t.odom.x:.3f},y={t.odom.y:.3f},th={t.odom.theta:.3f},"
            f"vx={t.odom.vx:.3f},vy={t.odom.vy:.3f},w={t.odom.omega:.3f})"
        )
    if t.joints:
        parts.append(
            "joints(" + ",".join(f"{j.name}={j.position:.3f}" for j in t.joints) + ")"
        )
    if t.battery is not None:
        parts.append(f"battery={t.battery * 100:.0f}%")
    return " ".join(parts) or "no telemetry"


def _status_str(robot: Robot) -> str:
    s = robot.status()
    return f"state={s.state.value} protocol={s.protocol} label={s.label}"


class RobotPerceptor:
    """Reads ``Robot.telemetry()`` and builds an ``agent_engine.WorldState``."""

    def __init__(self, robot: Robot) -> None:
        self._robot = robot

    def perceive(self):
        from agent_engine.core.blackboard import WorldState

        t = self._robot.telemetry()
        odom = t.odom
        arm = [j.position for j in t.joints] if t.joints else []
        return WorldState(
            base_pose=(odom.x, odom.y, odom.theta) if odom else (0.0, 0.0, 0.0),
            base_velocity=(odom.vx, odom.vy, odom.omega) if odom else (0.0, 0.0, 0.0),
            arm_joint_positions=arm,
            battery_pct=t.battery,
            mission_phase="active",
        )


def build_reasoner(
    tools,
    robot: Robot,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_travel_m: float = DEFAULT_MAX_TRAVEL_M,
):
    """Build a quadruped-aware tool-calling reasoner over a ``ReasoningRouter``."""
    from agent_engine.reasoners.claude_tool_caller import ClaudeToolCallingReasoner
    from agent_engine.reasoning.router import EchoBackend, ReasoningRouter

    card = build_robot_card(robot)
    system = PLANNER_SYSTEM_TEMPLATE.format(
        robot_card=card,
        max_steps=max_steps,
        max_travel_m=max_travel_m,
        tools=json.dumps(tools.to_anthropic_schema(), indent=1),
    )

    class _OhhoReasoner(ClaudeToolCallingReasoner):
        def plan(self, goal, world, memory_context, history):
            reply = self.router.complete(
                self._user_prompt(goal, world, memory_context, history),
                system=system,
                kind=self.call_kind,
            )
            return self._parse(reply)

    backends = []
    try:
        from agent_engine.reasoning.cloud_claude import CloudClaudeBackend

        cloud = CloudClaudeBackend()
        if cloud.available():
            backends.append(cloud)
    except Exception:
        pass

    echo = EchoBackend(
        responder=lambda prompt, system: (
            '{"rationale":"no LLM available — completing goal",'
            '"tool_calls":[],"goal_complete":true,"give_up":false,"ask_human":""}'
        )
    )
    backends.append(echo)

    router = ReasoningRouter(backends)
    return _OhhoReasoner(router, tools)


class HarnessBrain:
    """Real agent brain — wraps ``agent_engine.AgentHarness`` behind ``Brain``.

    Falls back to the echo backend (goal-complete stub) when no LLM is
    available, so ``Agent(bot).run(goal)`` always returns a log. With
    ``ANTHROPIC_API_KEY`` + ``[agent]`` extra installed, it performs a real
    perceive → reason → act → reflect loop.
    """

    def __init__(self, max_steps: int = 12) -> None:
        self.max_steps = max_steps

    def run(self, robot: Robot, goal: str, max_steps: int = 20) -> List[str]:
        try:
            from agent_engine.core.harness import AgentHarness
        except Exception as e:
            # agent_engine (or its numpy dependency) is missing — degrade to
            # the zero-dependency scripted brain instead of crashing the CLI.
            from .agent import ScriptedBrain

            log = [
                f"harness brain unavailable ({e.__class__.__name__}: {e}) — "
                "falling back to the scripted brain. Install numpy + the "
                "repo's agent_engine (pip install 'ohho-os[agent]') for the "
                "full perceive→reason→act→reflect loop."
            ]
            return log + ScriptedBrain().run(robot, goal, max_steps)

        blockers = goal_blockers(robot, goal)
        log: list[str] = [
            f"goal: {goal}",
            f"robot: {robot.spec.name} ({robot.spec.category})",
            f"capabilities: {', '.join(robot.spec.capabilities) or 'none'}",
        ]
        if blockers:
            log.append(f"refused: {'; '.join(blockers)}")
            log.append("done (harness brain — goal infeasible)")
            return log

        from .memory import SpatialMemory, default_memory_path
        from .nav import Navigator
        from .perception import SimPerceptor

        memory = SpatialMemory(path=default_memory_path(robot.spec.id))
        navigator = Navigator(robot) if robot.has(caps.BASE_DRIVE) else None
        sim_perceptor = SimPerceptor(robot) if robot.simulated else None
        t0 = robot.telemetry()
        origin = (t0.odom.x, t0.odom.y) if t0.odom else (0.0, 0.0)
        step_budget = min(max_steps, self.max_steps)
        tools = build_tool_registry(
            robot,
            navigator=navigator,
            memory=memory,
            perceptor=sim_perceptor,
            logger=log.append,
            travel_origin=origin,
            max_travel_m=DEFAULT_MAX_TRAVEL_M,
        )
        perceptor = RobotPerceptor(robot)
        reasoner = build_reasoner(
            tools,
            robot,
            max_steps=step_budget,
            max_travel_m=DEFAULT_MAX_TRAVEL_M,
        )
        log += [
            build_robot_card(robot),
            f"tools: {', '.join(tools.names())}",
            memory.describe(),
        ]
        harness = AgentHarness(
            perceptor=perceptor,
            reasoner=reasoner,
            tools=tools,
            max_steps_per_goal=step_budget,
            logger=log.append,
        )
        harness.submit_goal(goal)
        reports = harness.run_until_idle(max_ticks=max_steps * 4)
        for r in reports:
            log.append(f"  [{r.phase.value}→{r.next_phase.value}] {r.detail}")
        refl = harness.last_reflection
        if refl is not None:
            log.append(
                f"reflection: success={refl.success} "
                f"confidence={refl.confidence:.2f} — {refl.summary}"
            )
        try:
            memory.save()
        except Exception:
            pass  # memory persistence is best-effort
        log.append("done (harness brain)")
        return log
