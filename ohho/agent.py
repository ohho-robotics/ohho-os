"""A minimal agent loop.

``Agent(robot).run(goal)`` runs a pluggable "brain". The default is a
deterministic scripted brain that demonstrates the perceive → act → reflect loop
and exercises the robot. The real reasoning stack (an LLM-driven
perceive → reason → act → reflect loop) plugs in here via ``agent_engine``.
"""

from __future__ import annotations

import re
import time
from typing import List, Optional, Protocol

from . import capabilities as caps
from .planner import goal_blockers


class Brain(Protocol):
    def run(self, robot, goal: str, max_steps: int) -> List[str]: ...


_STATUS_GOAL_RE = re.compile(r"\b(status|connection|connected)\b", re.I)
_LIE_DOWN_GOAL_RE = re.compile(r"\b(lie\s*down|lying\s+down|lay\s+down)\b", re.I)


class ScriptedBrain:
    """Deterministic placeholder brain — no LLM, no external calls."""

    def run(self, robot, goal: str, max_steps: int = 20) -> List[str]:
        log: List[str] = [
            f"goal: {goal}",
            f"robot: {robot.spec.name} ({robot.spec.category})",
            f"capabilities: {', '.join(robot.spec.capabilities) or 'none'}",
        ]
        blockers = goal_blockers(robot, goal)
        if blockers:
            log.append(f"refused: {'; '.join(blockers)}")
            log.append("done (scripted brain)")
            return log

        if robot.has(caps.LEGGED_POSE):
            return log + self._run_quadruped(robot, goal)

        return log + self._run_wheeled(robot, goal)

    def _run_quadruped(self, robot, goal: str) -> List[str]:
        log: List[str] = []
        if _STATUS_GOAL_RE.search(goal):
            st = robot.status()
            log.append(
                f"status: state={st.state.value} protocol={st.protocol} label={st.label}"
            )
            log.append("done (scripted brain)")
            return log

        if _LIE_DOWN_GOAL_RE.search(goal):
            robot.lie_down()
            log.append(f"posture: {robot.legged_posture()}")
            log.append("done (scripted brain)")
            return log

        t0 = robot.telemetry()
        x0 = t0.odom.x if (t0 and t0.odom) else 0.0
        robot.stand()
        robot.drive(vx=0.1)
        time.sleep(0.2)
        robot.drive(vx=0.1, w=0.2)
        time.sleep(0.2)
        robot.stop()
        robot.lie_down()
        t1 = robot.telemetry()
        x1 = t1.odom.x if (t1 and t1.odom) else x0
        log.append(f"moved from x={x0:.3f} to x={x1:.3f}")
        log.append(f"posture: {robot.legged_posture()}")
        log.append("done (scripted brain — install agent_engine for full autonomy)")
        return log

    def _run_wheeled(self, robot, goal: str) -> List[str]:
        log: List[str] = []
        t0 = robot.telemetry()
        x0 = t0.odom.x if (t0 and t0.odom) else 0.0
        robot.drive(vx=0.1)
        time.sleep(0.2)
        robot.drive(vx=0.1, w=0.3)
        time.sleep(0.2)
        robot.stop()
        t1 = robot.telemetry()
        x1 = t1.odom.x if (t1 and t1.odom) else x0
        log.append(f"moved from x={x0:.3f} to x={x1:.3f}")
        log.append("done (scripted brain — install agent_engine for full autonomy)")
        return log


def _default_brain() -> Brain:
    """Prefer the real reasoning stack if available; otherwise scripted.

    Tries to load ``ohho.brains.HarnessBrain`` (which lazily imports
    ``agent_engine`` + ``anthropic``). Falls back to ``ScriptedBrain`` when
    ``agent_engine`` is not installed (the common case without the ``[agent]``
    extra).
    """
    try:
        from . import brains

        if brains.harness_available():
            return brains.HarnessBrain()
    except Exception:
        pass
    return ScriptedBrain()


class Agent:
    def __init__(self, robot, brain: Optional[Brain] = None) -> None:
        self.robot = robot
        self.brain: Brain = brain or _default_brain()

    def run(self, goal: str, max_steps: int = 20) -> List[str]:
        return self.brain.run(self.robot, goal, max_steps)
