"""Sim goal eval harness (OHH-127)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from .agent import Agent, ScriptedBrain
from .robot import Robot

_EVAL_PATH = Path(__file__).resolve().parent.parent / "evals" / "go2_sim_goals.yaml"


@dataclass(frozen=True)
class SimGoal:
    id: int
    goal: str
    scripted_pass: bool = False
    expect_refuse: bool = False
    expect_log_substrings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SimEvalSpec:
    robot_id: str
    transport: str
    max_steps: int
    max_travel_m: float
    goals: tuple[SimGoal, ...]


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal YAML reader for the fixed eval file (no PyYAML required)."""
    lines = text.splitlines()
    root: dict[str, Any] = {}
    goals: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    list_key: Optional[str] = None

    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - "):
            rest = line[4:].strip()
            if list_key == "goals":
                current = {}
                goals.append(current)
                if rest and ":" in rest:
                    key, val = rest.split(":", 1)
                    key, val = key.strip(), val.strip()
                    if val in ("true", "false"):
                        current[key] = val == "true"
                    elif val.isdigit():
                        current[key] = int(val)
                    else:
                        current[key] = val.strip('"').strip("'")
            continue
        if line.startswith("    "):
            if current is None:
                continue
            m = re.match(r"\s+(\w+):\s*(.*)$", line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                current[key] = [
                    s.strip().strip('"').strip("'")
                    for s in inner.split(",")
                    if s.strip()
                ]
            elif val in ("true", "false"):
                current[key] = val == "true"
            elif val.isdigit():
                current[key] = int(val)
            else:
                current[key] = val.strip('"').strip("'")
            continue
        m = re.match(r"^(\w+):\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val == "":
            list_key = key
            root[key] = []
            continue
        list_key = None
        if val in ("true", "false"):
            root[key] = val == "true"
        elif re.fullmatch(r"\d+", val):
            root[key] = int(val)
        elif re.fullmatch(r"\d+\.\d+", val):
            root[key] = float(val)
        else:
            root[key] = val.strip('"').strip("'")

    root["goals"] = goals
    return root


def load_go2_sim_eval(path: Optional[Path] = None) -> SimEvalSpec:
    p = path or _EVAL_PATH
    doc = _parse_simple_yaml(p.read_text(encoding="utf-8"))
    goals = tuple(
        SimGoal(
            id=int(g["id"]),
            goal=str(g["goal"]),
            scripted_pass=bool(g.get("scripted_pass", False)),
            expect_refuse=bool(g.get("expect_refuse", False)),
            expect_log_substrings=tuple(g.get("expect_log_substrings") or ()),
        )
        for g in doc.get("goals", [])
    )
    return SimEvalSpec(
        robot_id=str(doc.get("robot_id", "unitree-go2")),
        transport=str(doc.get("transport", "sim://")),
        max_steps=int(doc.get("max_steps", 20)),
        max_travel_m=float(doc.get("max_travel_m", 6.0)),
        goals=goals,
    )


def _log_ok(log: List[str], needles: tuple[str, ...]) -> bool:
    blob = "\n".join(log)
    return all(n in blob for n in needles)


def run_scripted_go2_eval(
    spec: SimEvalSpec,
    *,
    only_scripted_pass: bool = True,
) -> list[tuple[SimGoal, bool, List[str]]]:
    """Run goals with :class:`ScriptedBrain` (no API key)."""
    results: list[tuple[SimGoal, bool, List[str]]] = []
    with Robot.connect(spec.robot_id, spec.transport, runtime="native") as bot:
        brain = ScriptedBrain()
        for g in spec.goals:
            if only_scripted_pass and not g.scripted_pass:
                continue
            log = Agent(bot, brain=brain).run(g.goal, max_steps=spec.max_steps)
            ok = _log_ok(log, g.expect_log_substrings)
            if g.expect_refuse:
                ok = ok and any("refused:" in line for line in log)
            results.append((g, ok, log))
    return results
