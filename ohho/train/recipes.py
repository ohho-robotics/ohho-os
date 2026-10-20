"""Train recipes for a recorded dataset.

:func:`training_recipe` reads the dataset feature names and describes an ACT
or SmolVLA run. :func:`smoke_finetune` is the CPU check: a couple of ACT
steps when torch is installed, otherwise the mock checkpoint. SmolVLA is
validated and not launched. Both go through :func:`ohho.train.finetune`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from ..policy_contract import contract_from_reader

_SMOKE_CHUNK = 2
_SMOKE_BATCH = 2
_SMOKE_WIDTH = 32


def training_recipe(
    dataset: str,
    policy: str = "act",
    *,
    reader: Any = None,
) -> dict[str, Any]:
    """Hyperparameters and tensor shapes for ``dataset``.

    The shapes are the dataset's state and action names. A humanoid recipe
    keeps the arm-action form the recorder stored. The GPU epoch count is
    recorded here and is not executed.
    """
    from . import SUPPORTED_POLICIES
    from ..data.reader import DatasetReader

    if policy not in SUPPORTED_POLICIES:
        raise ValueError(
            f"unknown policy '{policy}'. Supported: {', '.join(SUPPORTED_POLICIES)}"
        )
    loaded = reader if reader is not None else DatasetReader(dataset)
    contract = contract_from_reader(loaded)
    state_dim = contract.state_dim or loaded.state_dim
    action_dim = contract.action_dim or loaded.action_dim
    if state_dim < 1 or action_dim < 1:
        raise ValueError("dataset needs a non-empty state and action")
    if contract.state_names and len(contract.state_names) != loaded.state_dim:
        raise ValueError("state names do not match observation.state shape")
    if contract.action_names and len(contract.action_names) != loaded.action_dim:
        raise ValueError("action names do not match the action shape")
    recipe: dict[str, Any] = {
        "policy": policy,
        "kind": contract.kind,
        "state_names": list(contract.state_names),
        "action_names": list(contract.action_names),
        "arm_action": contract.arm_action,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "batch_size": 8,
        "lr": 1e-4,
        "chunk_size": 8 if contract.kind == "humanoid" else 10,
        "device": "cpu",
        "gpu_finetune": "not_run",
    }
    if policy == "act":
        recipe["train_steps"] = 200
    else:
        from . import _default_checkpoint

        recipe["num_epochs"] = 100
        recipe["checkpoint"] = _default_checkpoint(policy)
    return recipe


def smoke_finetune(
    dataset: str,
    output_dir: str,
    policy: str = "act",
    *,
    steps: int = 2,
) -> dict[str, Any]:
    """Prove the train entry point on a few frames.

    ACT runs ``steps`` on CPU when torch imports, and the mock checkpoint
    otherwise. Any other policy writes the recipe and does not start
    training.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    recipe = training_recipe(dataset, policy)
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "recipe.json", recipe)
    if policy != "act":
        result = {
            "status": "dry_run",
            "policy": policy,
            "checkpoint": "",
            "ran_steps": 0,
            "recipe": recipe,
        }
        _write_json(out / "smoke.json", result)
        return result

    torch_ready = importlib.util.find_spec("torch") is not None
    from . import finetune

    checkpoint = finetune(
        dataset,
        "act",
        device="cpu",
        output_dir=str(out),
        num_epochs=1,
        batch_size=_SMOKE_BATCH,
        lr=float(recipe["lr"]),
        mock=not torch_ready,
        train_steps=steps,
        chunk_size=_SMOKE_CHUNK,
        d_model=_SMOKE_WIDTH,
        nhead=2,
        dim_feedforward=_SMOKE_WIDTH,
        latent_dim=4,
    )
    result = {
        "status": "trained" if torch_ready else "mock",
        "policy": "act",
        "checkpoint": checkpoint,
        "ran_steps": steps,
        "recipe": recipe,
    }
    _write_json(out / "smoke.json", result)
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


__all__ = ["smoke_finetune", "training_recipe"]
