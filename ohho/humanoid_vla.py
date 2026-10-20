"""Closed-loop eval of an upper-body policy in the humanoid simulator.

A scripted checkpoint reaches a joint target. An ACT checkpoint is loaded
through the same serve adapter the mobile base uses. The success rate is
written as JSON. A GPU fine-tune has not been run, so this rate belongs to
the checkpoint that was loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .data.recorder import _telemetry_to_state
from .data.upper_body import SpecLike, open_upper_body_sim
from .policy_contract import (
    PolicyContract,
    apply_policy_action,
    contract_for_spec,
    contract_from_names,
)

_DEFAULT_TARGET = 0.3


def write_scripted_checkpoint(path: str, *, target: float = _DEFAULT_TARGET) -> str:
    """Write a trivial reach checkpoint and return its directory."""
    out = Path(path).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": "scripted",
        "task": "reach",
        "target_joint": float(target),
    }
    with (out / "checkpoint.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(out)


def evaluate_upper_body(
    robot: SpecLike,
    checkpoint: str,
    output_path: str,
    *,
    episodes: int = 2,
    steps: int = 16,
    tolerance: float = 0.12,
    target: Optional[float] = None,
    fps: float = 20.0,
) -> dict[str, Any]:
    """Run ``episodes`` in sim and write a success-rate report.

    Success is the first joint of the first arm group finishing within
    ``tolerance`` radians of the target. Leg joints are not part of the
    action. ``gpu_finetune`` stays ``not_run``.
    """
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    loaded = _load_checkpoint(checkpoint)
    policy_name = str(loaded.get("policy") or "act")
    goal = float(target) if target is not None else _target_of(loaded)
    with open_upper_body_sim(robot, fps=fps) as session:
        spec = session.spec
        robot_id = spec.id
        contract = _contract_for_checkpoint(spec, loaded)
        joint_name = _target_joint(contract)
        rows: list[dict[str, Any]] = []
        successes = 0
        for index in range(episodes):
            _settle(session)
            final = _run_episode(
                session,
                contract,
                loaded,
                policy_name,
                goal,
                steps,
            )
            ok = abs(final - goal) <= tolerance
            if ok:
                successes += 1
            rows.append(
                {
                    "index": index,
                    "success": ok,
                    "joint": joint_name,
                    "target": goal,
                    "final": final,
                }
            )
    report = {
        "task": "reach",
        "policy": policy_name,
        "checkpoint": str(checkpoint),
        "robot_id": robot_id,
        "episodes": episodes,
        "steps_per_episode": steps,
        "successes": successes,
        "success_rate": successes / episodes,
        "tolerance_rad": tolerance,
        "gpu_finetune": "not_run",
        "note": (
            "success_rate is the loaded checkpoint in simulation. "
            "A GPU fine-tune has not been run."
        ),
        "results": rows,
    }
    destination = _report_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(
        f"upper-body eval: {successes}/{episodes} "
        f"success_rate={report['success_rate']:.3f} policy={policy_name}"
    )
    print(f"wrote {destination}")
    return report


def _run_episode(
    session: Any,
    contract: PolicyContract,
    loaded: dict[str, Any],
    policy_name: str,
    goal: float,
    steps: int,
) -> float:
    model = None
    if policy_name == "act":
        from .serve.act import ACTModel

        model = ACTModel(device="cpu")
        model.load_model(str(loaded["path"]))
    joint_name = _target_joint(contract).split(".", 1)[-1]
    final = 0.0
    for _ in range(steps):
        if policy_name == "scripted":
            vector = _scripted_action(contract, goal)
        elif model is not None:
            state = _telemetry_to_state(session.robot)
            prediction = model.predict_action(None, "reach", state=state)
            vector = list(prediction["vector"])
        else:
            raise ValueError(
                f"policy {policy_name!r} has no eval runner; use scripted or act"
            )
        apply_policy_action(session.robot, vector, contract)
        session.step()
        final = _reading(session.robot, joint_name)
    return final


def _scripted_action(contract: PolicyContract, goal: float) -> list[float]:
    action = [0.0] * contract.action_dim
    spans = contract.spans_of("arm_joints")
    if not spans:
        raise ValueError("contract has no arm group")
    action[spans[0].indices[0]] = goal
    return action


def _target_joint(contract: PolicyContract) -> str:
    spans = contract.spans_of("arm_joints")
    if not spans or not spans[0].indices:
        raise ValueError("contract has no arm joint")
    return contract.action_names[spans[0].indices[0]]


def _settle(session: Any) -> None:
    width = len(session.arm_joints)
    zeros = [0.0] * width
    for _ in range(8):
        session.robot.move_joints(session.arm_group, zeros)
        session.step()


def _reading(robot: Any, name: str) -> float:
    for joint in robot.telemetry().joints:
        if joint.name == name:
            return float(joint.position)
    return 0.0


def _contract_for_checkpoint(spec: Any, loaded: dict[str, Any]) -> PolicyContract:
    names = loaded.get("action_names")
    if isinstance(names, list) and names:
        state = loaded.get("state_names") or ()
        if not isinstance(state, list):
            state = ()
        arm = loaded.get("arm_action")
        return contract_from_names(
            state,
            names,
            arm_action=arm if isinstance(arm, str) else None,
            kind="humanoid",
        )
    form = loaded.get("arm_action")
    return contract_for_spec(spec, form if isinstance(form, str) else "arm_joints")


def _load_checkpoint(path: str) -> dict[str, Any]:
    root = Path(path).expanduser()
    if root.is_file():
        data = _read_json(root)
        data["path"] = str(root.parent if root.suffix == ".json" else root)
        return data
    for name in ("checkpoint.json", "config.json"):
        candidate = root / name
        if candidate.is_file():
            data = _read_json(candidate)
            data["path"] = str(root)
            return data
    raise FileNotFoundError(f"no checkpoint at {path}")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not a JSON object")
    return loaded


def _target_of(loaded: dict[str, Any]) -> float:
    raw = loaded.get("target_joint", _DEFAULT_TARGET)
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_joint must be a number") from exc


def _report_path(output_path: str) -> Path:
    path = Path(output_path).expanduser()
    if path.suffix.lower() == ".json":
        return path
    return path / "eval.json"


__all__ = [
    "evaluate_upper_body",
    "write_scripted_checkpoint",
]
