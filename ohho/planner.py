"""Quadruped-aware planning helpers for the agent brain (OHH-127).

Robot card text for the LLM, goal feasibility checks, and scripted fallbacks
that respect stand-before-drive / lie-down-at-end for legged platforms.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from . import capabilities as caps
from .safety import EULER_LIMIT_RAD

if TYPE_CHECKING:
    from .robot import Robot

# Agent-facing drive cap for a quadruped (SafetyGate may clamp further on hardware).
LEGGED_AGENT_MAX_LIN_M_S = 0.5
DEFAULT_MAX_TRAVEL_M = 6.0
DEFAULT_MAX_STEPS = 20

_MANIPULATION_RE = re.compile(
    r"\b(pick\s*up|pickup|grasp|grab|lift\s+the|carry\s+the|hold\s+the)\b",
    re.I,
)
_STAIRS_RE = re.compile(r"\b(stairs?|staircase|upstairs|downstairs)\b", re.I)


def agent_max_linear_speed(robot: Robot) -> float:
    if robot.has(caps.LEGGED_POSE):
        return LEGGED_AGENT_MAX_LIN_M_S
    return robot.spec.max_lin


def build_robot_card(robot: Robot) -> str:
    """Short platform card injected into the planner system prompt."""
    lines = [
        f"Robot: {robot.spec.name} ({robot.spec.id})",
        f"Category: {robot.spec.category}",
        f"Capabilities: {', '.join(robot.spec.capabilities) or 'none'}",
    ]
    if robot.has(caps.LEGGED_POSE):
        lines += [
            "Form factor: quadruped — no arm, no manipulation.",
            f"Max safe drive speed: {LEGGED_AGENT_MAX_LIN_M_S} m/s forward/lateral.",
            "You must call stand before any drive or navigate_to.",
            "When the goal is complete, call lie_down before goal_complete.",
            "Stairs / uneven vertical terrain: not supported in v0 — refuse such goals.",
        ]
    elif robot.has(caps.MANIPULATION):
        lines.append(
            f"Mobile manipulator: {len(robot.spec.joint_names)}-DOF arm "
            f"({', '.join(robot.spec.joint_names)})."
        )
    if robot.has(caps.BASE_DRIVE):
        lines.append(
            f"Holonomic base drive: {robot.has(caps.BASE_HOLONOMIC)}; "
            f"planner drive cap {agent_max_linear_speed(robot):.2f} m/s."
        )
    return "\n".join(lines)


def goal_blockers(robot: Robot, goal: str) -> List[str]:
    """Human-readable reasons to refuse a goal before planning."""
    reasons: List[str] = []
    if _MANIPULATION_RE.search(goal) and not robot.has(caps.MANIPULATION):
        reasons.append("no manipulation capability (cannot pick up or grasp)")
    if _STAIRS_RE.search(goal):
        reasons.append("stairs not supported in v0")
    return reasons


def travel_distance_m(robot: Robot, origin: Tuple[float, float]) -> float:
    t = robot.telemetry()
    if not t.odom:
        return 0.0
    return math.hypot(t.odom.x - origin[0], t.odom.y - origin[1])


def wrap_drive_with_travel_limit(
    robot: Robot,
    handler: Callable[..., object],
    *,
    origin: Tuple[float, float],
    max_travel_m: float,
    log: Optional[Callable[[str], None]] = None,
) -> Callable[..., object]:
    """Return a drive handler that refuses when the base exceeds ``max_travel_m``."""

    def _drive(**kwargs):
        dist = travel_distance_m(robot, origin)
        if dist > max_travel_m:
            msg = (
                f"refused: travel budget exceeded ({dist:.2f} m > {max_travel_m:.2f} m)"
            )
            if log:
                log(msg)
            return msg
        return handler(**kwargs)

    return _drive


def euler_limits() -> float:
    return EULER_LIMIT_RAD


PLANNER_SYSTEM_TEMPLATE = """You are the planning module for an embodied robot agent.
Each cycle you receive world state, memory, and recent tool results; choose the
next step toward the goal.

ROBOT CARD:
{robot_card}

PLANNER GUARDRAILS:
- Max planner steps per goal: {max_steps}
- Max base travel from start of goal: {max_travel_m:.1f} m (stop and give_up if exceeded)
- Only use tools listed below; never invent raw motor or sport-mode API calls
- For quadrupeds: stand before drive/navigate; lie_down when finished
- Refuse (give_up=true) goals that need missing capabilities

Respond with a SINGLE JSON object and nothing else:
{{"rationale": "<one sentence>",
  "tool_calls": [{{"tool": "<name>", "args": {{...}}}}],
  "goal_complete": <true when done>,
  "give_up": <true if impossible/unsafe>,
  "ask_human": "<question or empty>"}}

Rules:
- Prefer one tool call per cycle.
- Set goal_complete=true with empty tool_calls when the goal is achieved.

Available tools (JSON schema):
{tools}"""
