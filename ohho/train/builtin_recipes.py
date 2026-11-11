"""Built-in recipe families.

ACT calls :func:`ohho.train.act.train_act` and writes a real checkpoint.
Diffusion Policy calls :func:`ohho.train.il.diffusion_policy.train_diffusion`.
PPO calls :func:`ohho.train.ppo.run_ppo` and writes a real checkpoint.
SmolVLA and OpenVLA are registered so a recipe can name them, and their
run step reports that the family is not available. They do not call a mock trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ohho.compression.qat import QatError, prepare_qat_recipe
from .il.family import DiffusionRecipe, prepare_il_run
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

    def preflight(self, recipe: dict[str, Any]) -> None:
        from .il.family import preflight_il_recipe

        preflight_il_recipe(recipe, family_id=self.id)
        _prepare_qat(recipe)
        train_steps = int(recipe.get("hyperparameters", {}).get("train_steps", 200))
        _pruning_for(recipe, train_steps)

    def run(self, recipe: dict[str, Any], emit: Emit) -> TrainResult:
        plan = prepare_precision(recipe, family=self)
        _prepare_qat(recipe)
        prepared = prepare_il_run(
            recipe,
            emit,
            family_id=self.id,
            family_label="ACT",
            trainer_name="ACT",
        )
        train_steps = int(prepared.resolved["train_steps"])
        pruning = _pruning_for(recipe, train_steps)
        planned = train_steps
        if pruning is not None:
            from .compression.pruning import optimizer_steps

            planned = optimizer_steps(pruning, train_steps)
        emit(
            {
                "event": "train_start",
                "family": self.id,
                "device": plan.device,
                "precision": plan.precision,
                "train_steps": planned,
                "val_every": int(prepared.resolved["val_every"]),
            }
        )
        try:
            from ohho.train.act import train_act

            checkpoint = train_act(
                dataset=str(prepared.dataset),
                output_dir=str(Path(recipe["output"]["dir"]).expanduser()),
                device=plan.device,
                precision=plan.precision,
                mock=False,
                pruning=pruning,
                qat=recipe.get("qat"),
                train_steps=train_steps,
                batch_size=int(prepared.resolved["batch_size"]),
                lr=float(prepared.resolved["lr"]),
                chunk_size=int(prepared.resolved["chunk_size"]),
                d_model=int(prepared.resolved["d_model"]),
                nhead=int(prepared.resolved["nhead"]),
                num_encoder_layers=int(prepared.resolved["num_encoder_layers"]),
                num_decoder_layers=int(prepared.resolved["num_decoder_layers"]),
                dim_feedforward=int(prepared.resolved["dim_feedforward"]),
                latent_dim=int(prepared.resolved["latent_dim"]),
                kl_weight=float(prepared.resolved["kl_weight"]),
                log_every=int(prepared.resolved["log_every"]),
                val_every=int(prepared.resolved["val_every"]),
                val_fraction=float(prepared.resolved["val_fraction"]),
                early_stop_patience=int(prepared.resolved["early_stop_patience"]),
                early_stop_min_delta=float(prepared.resolved["early_stop_min_delta"]),
                vision_backbone=str(prepared.resolved["vision_backbone"]),
                on_event=_on_event(self.id, emit),
            )
        except (QatError, TrainFailed):
            raise
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


class PpoRecipe:
    """PPO on a registered task.

    The CPU path is a built-in PyTorch trainer. ``rsl_rl`` and ``skrl`` are
    optional backends. See :mod:`ohho.train.ppo`.
    """

    id = "ppo"

    def available(self) -> bool:
        return True

    def metadata(self) -> dict[str, Any]:
        """Precisions the shared runner may select for PPO."""
        return {"precisions": ["fp32"]}

    def run(self, recipe: dict[str, Any], emit: Emit) -> TrainResult:
        from .ppo import run_ppo

        return run_ppo(recipe, emit)


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
DIFFUSION = DiffusionRecipe()
SMOLVLA = UnavailableRecipe("smolvla")
OPENVLA = UnavailableRecipe("openvla")
PPO = PpoRecipe()


def register_builtins() -> None:
    """Register the families that ship with ohho-os."""
    register(ACT, source="builtin:ohho.train.builtin_recipes")
    register(DIFFUSION, source="builtin:ohho.train.builtin_recipes")
    register(SMOLVLA, source="builtin:ohho.train.builtin_recipes")
    register(OPENVLA, source="builtin:ohho.train.builtin_recipes")
    register(PPO, source="builtin:ohho.train.builtin_recipes")


def _on_event(family_id: str, emit: Emit) -> Emit:
    def _send(payload: dict[str, Any]) -> None:
        body = dict(payload)
        body.setdefault("family", family_id)
        emit(body)

    return _send


def _check_precision(recipe: dict[str, Any], family_id: str = "act") -> None:
    if family_id == "act":
        return
    precision = recipe.get("precision", "fp32")
    if precision != "fp32":
        raise RecipeConfigError(
            f"precision {precision!r} is not available for family {family_id!r} in this build"
        )


def _resolve_device(requested: str, family_label: str = "ACT") -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        from ohho.hardware import resolve_device

        device = resolve_device("auto")
        return device if device in ("cpu", "cuda", "mps") else "cpu"
    torch = _load_torch(family_label)
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RecipeConfigError("device 'cuda' is not available on this machine")
        return requested
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RecipeConfigError("device 'mps' is not available on this machine")
        return "mps"
    raise RecipeConfigError(f"device {requested!r} is not available in this build")


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
    block = recipe.get("qat")
    if isinstance(block, dict):
        kwargs["qat"] = block
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


def _prepare_qat(recipe: dict[str, Any]) -> None:
    """Refuse a QAT block the ACT trainer cannot run. Absent block is a no-op."""
    block = recipe.get("qat")
    if not isinstance(block, dict):
        return
    prepare_qat_recipe(block, family_id="act")

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


def _load_torch(family_label: str = "ACT") -> Any:
    try:
        return _import_torch()
    except ImportError as exc:
        raise RecipeConfigError(
            f"{family_label} training needs torch. Install it with: pip install 'ohho-os[train]'"
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


def _emit_declared(
    recipe: dict[str, Any], emit: Emit, trainer_name: str = "ACT"
) -> None:
    weights = recipe.get("base_weights")
    if isinstance(weights, dict) and weights:
        emit(
            {
                "event": "base_weights",
                "status": "not_loaded",
                "uri": weights.get("uri", ""),
                "reason": (
                    f"the {trainer_name} trainer in this build starts from the dataset "
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
