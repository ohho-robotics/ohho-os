"""Load Proof scenario suites by reference.

ohho-os does not vendor scenario geometry. A suite name points at a file
listed in ``suites.json`` (repository URL plus a path inside that checkout).
Set ``OHHO_PROOF_ROOT`` to the checkout root, or pass the file path.

The reader accepts the subset of YAML that the Proof nav regression file
uses (block maps, block lists, and one-line flow maps). PyYAML is not
required.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from .errors import EvalError

_SCALAR_INT = re.compile(r"-?\d+")
_SCALAR_FLOAT = re.compile(r"-?(?:\d+\.\d+|\d+[eE][+-]?\d+)")


@dataclass(frozen=True)
class SuiteRef:
    """A named pointer at a scenario file. The file is not copied here."""

    name: str
    title: str
    repository: str
    relative_path: str

    @property
    def source(self) -> str:
        """Stable reference string stored on a report when this file is used."""
        return f"{self.repository}#{self.relative_path}"


def _load_suites() -> dict[str, SuiteRef]:
    raw = files("ohho.eval").joinpath("suites.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or not data:
        raise RuntimeError("eval suites.json must be a non-empty object")
    suites: dict[str, SuiteRef] = {}
    for name, item in data.items():
        if not isinstance(name, str) or not isinstance(item, dict):
            raise RuntimeError("eval suites.json entries must be objects")
        suites[name] = SuiteRef(
            name=name,
            title=str(item["title"]),
            repository=str(item["repository"]),
            relative_path=str(item["relative_path"]),
        )
    return suites


# Pointers only. Scenario bodies stay in the repository named by each entry.
SUITES: dict[str, SuiteRef] = _load_suites()


@dataclass(frozen=True)
class ProofScenario:
    """One scenario read from a Proof YAML file."""

    id: str
    start_x: float
    start_y: float
    start_yaw: float
    goal_x: float
    goal_y: float
    goal_yaw: float
    obstacle_name: str
    obstacle_x: float
    obstacle_y: float
    obstacle_yaw: float
    size_x: float
    size_y: float
    size_z: float
    world: str
    frame_id: str


def list_suites() -> tuple[SuiteRef, ...]:
    """Return the known suite pointers, in name order."""
    return tuple(SUITES[name] for name in sorted(SUITES))


def resolve_scenario_file(
    suite: str,
    scenarios: str | os.PathLike[str] | None = None,
) -> tuple[Path, str]:
    """Return ``(path, scenario_source)`` for a suite.

    An explicit ``scenarios`` path, then ``OHHO_EVAL_SCENARIOS``, wins over
    the checkout lookup. Those runs record the file path as the source.
    A named suite with ``OHHO_PROOF_ROOT`` records the repository reference,
    not a copy of the scenarios.
    """
    if scenarios is not None and str(scenarios).strip():
        path = Path(scenarios).expanduser()
        if not path.is_file():
            raise EvalError(f"scenario file not found: {path}")
        return path, path.resolve().as_posix()

    env_file = os.environ.get("OHHO_EVAL_SCENARIOS", "").strip()
    if env_file:
        path = Path(env_file).expanduser()
        if not path.is_file():
            raise EvalError(f"OHHO_EVAL_SCENARIOS file not found: {path}")
        return path, path.resolve().as_posix()

    ref = SUITES.get(suite)
    if ref is None:
        known = ", ".join(sorted(SUITES)) or "(none)"
        raise EvalError(
            f"unknown suite {suite!r}; known suites: {known}. "
            "Pass --scenarios with a Proof scenario YAML file."
        )
    root = os.environ.get("OHHO_PROOF_ROOT", "").strip()
    if not root:
        raise EvalError(
            f"suite {suite!r} references {ref.relative_path} in {ref.repository}. "
            "Set OHHO_PROOF_ROOT to that checkout or pass --scenarios. "
            "Scenario definitions are not vendored in ohho-os."
        )
    path = Path(root).expanduser() / ref.relative_path
    if not path.is_file():
        raise EvalError(f"suite {suite!r} scenario file missing: {path}")
    return path, ref.source


def load_scenarios(path: str | os.PathLike[str]) -> tuple[ProofScenario, ...]:
    """Parse a Proof scenario YAML file. Raises :class:`EvalError` on a bad file."""
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"could not read scenario file {file}: {exc}") from exc
    try:
        doc = parse_scenario_yaml(text)
    except EvalError:
        raise
    except Exception as exc:
        raise EvalError(f"could not parse scenario file {file}: {exc}") from exc
    return scenarios_from_document(doc, source=str(file))


def scenarios_from_document(
    doc: Any,
    *,
    source: str,
) -> tuple[ProofScenario, ...]:
    """Turn a parsed mapping into scenarios. ``source`` is only used in errors."""
    if not isinstance(doc, dict) or "scenarios" not in doc:
        raise EvalError(f"{source} must be a mapping with a 'scenarios' list")
    frame_id = str(doc.get("frame_id", "map"))
    world = str(doc.get("world", ""))
    raw_list = doc["scenarios"]
    if not isinstance(raw_list, list) or not raw_list:
        raise EvalError(f"{source} has no scenarios")
    found: dict[str, ProofScenario] = {}
    ordered: list[ProofScenario] = []
    for index, entry in enumerate(raw_list):
        if not isinstance(entry, dict):
            raise EvalError(f"{source} scenario #{index} must be a mapping")
        scenario = _scenario_from_entry(entry, index, frame_id, world, source)
        if scenario.id in found:
            raise EvalError(f"{source} duplicate scenario id {scenario.id!r}")
        found[scenario.id] = scenario
        ordered.append(scenario)
    return tuple(ordered)


def parse_scenario_yaml(text: str) -> dict[str, Any]:
    """Parse the Proof scenario YAML subset into a mapping."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if "#" in raw:
            raw = raw.split("#", 1)[0]
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    if not lines:
        raise EvalError("scenario YAML is empty")
    doc, index = _parse_map(lines, 0, lines[0][0])
    if index != len(lines):
        raise EvalError("scenario YAML has trailing content that was not parsed")
    if not isinstance(doc, dict):
        raise EvalError("scenario YAML root must be a mapping")
    return doc


def _parse_map(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines) and lines[index][0] == indent:
        content = lines[index][1]
        if content.startswith("- "):
            break
        key, rest = _split_key(content)
        index += 1
        if rest == "":
            if index < len(lines) and lines[index][0] > indent:
                child, index = _parse_at(lines, index)
            else:
                child = None
            result[key] = child
        elif rest.startswith("{") and rest.endswith("}"):
            result[key] = _parse_flow(rest)
        elif rest in {">", "|", ">-", "|-", ">+", "|+"}:
            raise EvalError(f"unsupported YAML scalar for {key!r}")
        else:
            result[key] = _scalar(rest)
    return result, index


def _parse_list(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    items: list[Any] = []
    while (
        index < len(lines)
        and lines[index][0] == indent
        and lines[index][1].startswith("- ")
    ):
        content = lines[index][1][2:].strip()
        index += 1
        if content == "":
            if index < len(lines) and lines[index][0] > indent:
                child, index = _parse_at(lines, index)
                items.append(child)
            else:
                items.append(None)
            continue
        if ":" not in content:
            items.append(_scalar(content))
            continue
        key, rest = _split_key(content)
        item: dict[str, Any] = {}
        if rest == "":
            if index < len(lines) and lines[index][0] > indent:
                child, index = _parse_at(lines, index)
                item[key] = child
            else:
                item[key] = None
        elif rest.startswith("{") and rest.endswith("}"):
            item[key] = _parse_flow(rest)
        else:
            item[key] = _scalar(rest)
        if (
            index < len(lines)
            and lines[index][0] > indent
            and not lines[index][1].startswith("- ")
        ):
            extra, index = _parse_map(lines, index, lines[index][0])
            item.update(extra)
        items.append(item)
    return items, index


def _parse_at(
    lines: list[tuple[int, str]],
    index: int,
) -> tuple[Any, int]:
    if index >= len(lines):
        raise EvalError("unexpected end of scenario YAML")
    indent, content = lines[index]
    if content.startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_map(lines, index, indent)


def _split_key(content: str) -> tuple[str, str]:
    if ":" not in content:
        raise EvalError(f"expected key: value, got {content!r}")
    key, rest = content.split(":", 1)
    key = key.strip()
    if not key:
        raise EvalError(f"empty key in {content!r}")
    return key, rest.strip()


def _parse_flow(text: str) -> dict[str, Any]:
    inner = text.strip()[1:-1].strip()
    if not inner:
        return {}
    result: dict[str, Any] = {}
    for part in inner.split(","):
        if ":" not in part:
            raise EvalError(f"expected key: value in flow map {text!r}")
        key, rest = part.split(":", 1)
        result[key.strip()] = _scalar(rest.strip())
    return result


def _scalar(value: str) -> Any:
    if (value.startswith('"') and value.endswith('"') and len(value) >= 2) or (
        value.startswith("'") and value.endswith("'") and len(value) >= 2
    ):
        return value[1:-1]
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "~"}:
        return None
    if _SCALAR_INT.fullmatch(value):
        return int(value)
    if _SCALAR_FLOAT.fullmatch(value):
        return float(value)
    return value


def _scenario_from_entry(
    entry: dict[str, Any],
    index: int,
    frame_id: str,
    world: str,
    source: str,
) -> ProofScenario:
    scenario_id = entry.get("id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise EvalError(f"{source} scenario #{index} has no id")
    start_x, start_y, start_yaw = _pose(
        entry.get("start"), f"{source} scenario {scenario_id!r} start"
    )
    goal_x, goal_y, goal_yaw = _pose(
        entry.get("goal"), f"{source} scenario {scenario_id!r} goal"
    )
    obstacle = entry.get("obstacle")
    if not isinstance(obstacle, dict):
        raise EvalError(f"{source} scenario {scenario_id!r} is missing obstacle")
    name = obstacle.get("name")
    if not isinstance(name, str) or not name.strip():
        raise EvalError(f"{source} scenario {scenario_id!r} obstacle has no name")
    obs_x, obs_y, obs_yaw = _pose(
        obstacle.get("pose"), f"{source} scenario {scenario_id!r} obstacle"
    )
    size = obstacle.get("size")
    if not isinstance(size, dict):
        raise EvalError(f"{source} scenario {scenario_id!r} obstacle is missing size")
    try:
        size_x = float(size["x"])
        size_y = float(size["y"])
        size_z = float(size["z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvalError(
            f"{source} scenario {scenario_id!r} obstacle size is incomplete"
        ) from exc
    for label, dim in (("x", size_x), ("y", size_y), ("z", size_z)):
        if dim <= 0.0:
            raise EvalError(
                f"{source} scenario {scenario_id!r} obstacle size {label} must be > 0"
            )
    return ProofScenario(
        id=scenario_id,
        start_x=start_x,
        start_y=start_y,
        start_yaw=start_yaw,
        goal_x=goal_x,
        goal_y=goal_y,
        goal_yaw=goal_yaw,
        obstacle_name=name,
        obstacle_x=obs_x,
        obstacle_y=obs_y,
        obstacle_yaw=obs_yaw,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        world=world,
        frame_id=frame_id,
    )


def _pose(mapping: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(mapping, dict):
        raise EvalError(f"{label} must be a mapping with x, y, yaw_deg")
    try:
        x = float(mapping["x"])
        y = float(mapping["y"])
        yaw_deg = float(mapping["yaw_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvalError(f"{label} must include numeric x, y, yaw_deg") from exc
    return x, y, math.radians(yaw_deg)
