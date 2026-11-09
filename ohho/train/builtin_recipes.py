"""Built-in recipe families.

ACT calls :func:`ohho.train.act.train_act` and writes a real checkpoint.
Diffusion Policy, SmolVLA, OpenVLA, and PPO are registered so a recipe can
name them, and their run step reports that the family is not available.
They do not call a mock trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .precision import PRECISIONS, prepare_precision
from .recipe_registry import Emit, TrainResult, register
from .recipe_schema import (
    DatasetUnavailable,
    FamilyUnavailable,
    RecipeConfigError,
    TrainFailed,
)

_ACT_KEYS = frozenset(
    {
        "train_steps",
        "num_epochs",
        "batch_size",
        "lr",
        "chunk_size",
        "d_model",
        "nhead",
        "num_encoder_layers",
        "num_decoder_layers",
        "dim_feedforward",
        "latent_dim",
        "kl_weight",
        "log_every",
        "seed",
    }
)
_PASSTHROUGH = (
    "batch_size",
    "lr",
    "chunk_size",
    "d_model",
    "nhead",
    "num_encoder_layers",
    "num_decoder_layers",
    "dim_feedforward",
    "latent_dim",
    "kl_weight",
    "log_every",
)


class ActRecipe:
    """Tiny ACT trainer backed by :func:`ohho.train.act.train_act`."""

    id = "act"

    def available(self) -> bool:
        return True

    def metadata(self) -> dict[str, Any]:
        """Precisions the shared runner may select for ACT."""
        return {"precisions": list(PRECISIONS)}

    def run(self, recipe: dict[str, Any], emit: Emit) -> TrainResult:
        plan = prepare_precision(recipe, family=self)
        kwargs = _act_kwargs(recipe)
        train_steps = int(kwargs.get("train_steps", 200))
        pruning = _pruning_for(recipe, train_steps)
        planned = train_steps
        if pruning is not None:
            from .compression.pruning import optimizer_steps

            planned = optimizer_steps(pruning, train_steps)
        dataset_path = _dataset_dir(recipe)
        torch = _load_torch()
        _emit_declared(recipe, emit)
        _apply_seed(torch, recipe, plan.device)
        emit(
            {
                "event": "train_start",
                "family": self.id,
                "device": plan.device,
                "precision": plan.precision,
                "train_steps": planned,
            }
        )
        try:
            from ohho.train.act import train_act

            checkpoint = train_act(
                dataset=str(dataset_path),
                output_dir=str(Path(recipe["output"]["dir"]).expanduser()),
                device=plan.device,
                precision=plan.precision,
                mock=False,
                pruning=pruning,
                **kwargs,
            )
        except Exception as exc:
            raise TrainFailed(f"ACT training failed: {exc}") from exc
        metrics = Path(checkpoint) / "metrics.json"
        if not metrics.is_file():
            raise TrainFailed(f"ACT training did not write {metrics}")
        return TrainResult(
            checkpoint_dir=str(Path(checkpoint)),
            metrics_path=str(metrics),
            device=plan.device,
            base_weights_loaded=False,
        )


class UnavailableRecipe:
    """A family id with no trainer in this build."""

    def __init__(self, family_id: str) -> None:
        self.id = family_id

    def available(self) -> bool:
        return False

    def metadata(self) -> dict[str, Any]:
        """The shared precision set. A later trainer picks it up unchanged."""
        return {"precisions": list(PRECISIONS)}

    def run(self, recipe: dict[str, Any], emit: Emit) -> TrainResult:
        del recipe, emit
        raise FamilyUnavailable(self.id)


ACT = ActRecipe()
DIFFUSION = UnavailableRecipe("diffusion")
SMOLVLA = UnavailableRecipe("smolvla")
OPENVLA = UnavailableRecipe("openvla")
PPO = UnavailableRecipe("ppo")


def register_builtins() -> None:
    """Register the families that ship with ohho-os."""
    register(ACT, source="builtin:ohho.train.builtin_recipes")
    register(DIFFUSION, source="builtin:ohho.train.builtin_recipes")
    register(SMOLVLA, source="builtin:ohho.train.builtin_recipes")
    register(OPENVLA, source="builtin:ohho.train.builtin_recipes")
    register(PPO, source="builtin:ohho.train.builtin_recipes")


def _act_kwargs(recipe: dict[str, Any]) -> dict[str, Any]:
    hyperparameters = recipe.get("hyperparameters") or {}
    if not isinstance(hyperparameters, dict):
        raise RecipeConfigError("hyperparameters must be a mapping")
    if "mock" in hyperparameters:
        raise RecipeConfigError(
            "hyperparameters.mock is not accepted; ohho train does not run a fake trainer"
        )
    unknown = sorted(set(hyperparameters) - _ACT_KEYS)
    if unknown:
        listed = ", ".join(unknown)
        raise RecipeConfigError(f"unknown hyperparameters for family act: {listed}")
    kwargs: dict[str, Any] = {}
    for key in _PASSTHROUGH:
        if key in hyperparameters:
            kwargs[key] = hyperparameters[key]
    if "train_steps" in hyperparameters:
        kwargs["train_steps"] = hyperparameters["train_steps"]
    elif "num_epochs" in hyperparameters:
        kwargs["train_steps"] = int(hyperparameters["num_epochs"]) * 2
    d_model = int(kwargs.get("d_model", 64))
    nhead = int(kwargs.get("nhead", 2))
    if nhead <= 0 or d_model % nhead != 0:
        raise RecipeConfigError(
            f"hyperparameters.d_model ({d_model}) must be divisible by nhead ({nhead})"
        )
    return kwargs


def _pruning_for(recipe: dict[str, Any], train_steps: int) -> dict[str, Any] | None:
    """Normalize ``recipe["pruning"]`` before training starts.

    A schedule that cannot fit in ``train_steps`` fails here, so the run
    manifest is not opened.
    """
    block = recipe.get("pruning")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise RecipeConfigError("pruning must be a mapping")
    from .compression.pruning import normalize_pruning

    return normalize_pruning(block, train_steps=train_steps)


def _dataset_dir(recipe: dict[str, Any]) -> Path:
    dataset = recipe["dataset"]
    if "path" not in dataset:
        raise DatasetUnavailable(
            "dataset version ids are not resolved in this build; "
            "set dataset.path to a local dataset directory"
        )
    path = Path(str(dataset["path"])).expanduser()
    if not path.is_dir():
        raise DatasetUnavailable(f"dataset path does not exist: {path}")
    return path


def _import_torch() -> Any:
    """Import torch. Tests patch this to simulate a build without the extra."""
    import torch

    return torch


def _load_torch() -> Any:
    try:
        return _import_torch()
    except ImportError as exc:
        raise RecipeConfigError(
            "ACT training needs torch. Install it with: pip install 'ohho-os[train]'"
        ) from exc


def _apply_seed(torch: Any, recipe: dict[str, Any], device: str) -> None:
    hyperparameters = recipe.get("hyperparameters") or {}
    if "seed" not in hyperparameters:
        return
    seed = int(hyperparameters["seed"])
    torch.manual_seed(seed)
    if device == "cuda" or device.startswith("cuda:"):
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "manual_seed_all"):
            cuda.manual_seed_all(seed)


def _emit_declared(recipe: dict[str, Any], emit: Emit) -> None:
    weights = recipe.get("base_weights")
    if isinstance(weights, dict) and weights:
        emit(
            {
                "event": "base_weights",
                "status": "not_loaded",
                "uri": weights.get("uri", ""),
                "reason": (
                    "the ACT trainer in this build starts from the dataset "
                    "and does not load base weights"
                ),
            }
        )
    for hook_type in ("compression_hooks", "eval_hooks"):
        hooks = recipe.get(hook_type) or []
        for hook in hooks:
            emit(
                {
                    "event": "hook",
                    "hook_type": hook_type,
                    "name": hook.get("name", ""),
                    "status": "skipped",
                    "reason": "hooks are not executed in this build",
                }
            )
