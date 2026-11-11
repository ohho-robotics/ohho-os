"""Preflight and the Diffusion Policy recipe family.

ACT uses the same preflight from :mod:`ohho.train.builtin_recipes` and then
calls :func:`ohho.train.act.train_act`. Both families register there.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..recipe_registry import TrainResult
from ..recipe_schema import RecipeConfigError, TrainFailed
from .data import DatasetRef, resolve_dataset
from .params import resolve_hyperparameters

Emit = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class PreparedRun:
    """Checked recipe values, local dataset directory, and device name."""

    resolved: dict[str, Any]
    dataset: Path
    device: str


class DiffusionRecipe:
    """Diffusion Policy on a LeRobot dataset. The tiny trainer is state-only."""

    id = "diffusion"

    def available(self) -> bool:
        return True

    def metadata(self) -> dict[str, Any]:
        """Precisions the shared runner may select for Diffusion."""
        return {"precisions": ["fp32"]}

    def preflight(self, recipe: dict[str, Any]) -> None:
        preflight_il_recipe(recipe, family_id=self.id)

    def run(self, recipe: dict[str, Any], emit: Emit) -> TrainResult:
        prepared = prepare_il_run(
            recipe,
            emit,
            family_id=self.id,
            family_label="Diffusion",
            trainer_name="diffusion",
        )
        return _finish_diffusion(recipe, emit, prepared)


def preflight_il_recipe(recipe: dict[str, Any], *, family_id: str) -> None:
    """Validate hyperparameters and reject untrained backbones before training."""
    resolved = resolve_hyperparameters(family_id, _hyperparameters(recipe))
    _reject_untrained_backbone(str(resolved["vision_backbone"]))


def prepare_il_run(
    recipe: dict[str, Any],
    emit: Emit,
    *,
    family_id: str,
    family_label: str,
    trainer_name: str,
) -> PreparedRun:
    """Validate parameters, dataset, and torch. Emit declared hooks. Do not train.

    Hyperparameter errors and a missing dataset happen before any progress event.
    ``train_start`` is emitted by the caller after this returns.
    """
    from ohho.train.builtin_recipes import (
        _apply_seed,
        _check_precision,
        _emit_declared,
        _load_torch,
        _resolve_device,
    )

    _check_precision(recipe, family_id)
    resolved = resolve_hyperparameters(family_id, _hyperparameters(recipe))
    _reject_untrained_backbone(str(resolved["vision_backbone"]))
    ref = resolve_dataset(recipe)
    device = _resolve_device(str(recipe.get("device", "cpu")), family_label)
    torch_module = _load_torch(family_label)
    dataset = _local_dataset(ref)
    _emit_declared(recipe, emit, trainer_name=trainer_name)
    _apply_seed(torch_module, recipe, device)
    return PreparedRun(resolved=resolved, dataset=dataset, device=device)


def _hyperparameters(recipe: dict[str, Any]) -> dict[str, Any]:
    value = recipe.get("hyperparameters")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RecipeConfigError("hyperparameters must be a mapping")
    return value


def _reject_untrained_backbone(backbone: str) -> None:
    """``none`` is the stem this build trains. Other legal names stop here."""
    if backbone == "none":
        return
    raise RecipeConfigError(
        f"hyperparameters.vision_backbone {backbone!r} is not trainable in this build; "
        "ACT and Diffusion Policy recipes train on state and action vectors. "
        "Use vision_backbone 'none'."
    )


def _local_dataset(ref: DatasetRef) -> Path:
    if ref.kind == "hub":
        # Attribute lookup so tests can patch data.materialize_hub_dataset.
        from . import data as dataset_data

        return dataset_data.materialize_hub_dataset(ref.repo_id)
    if ref.path is None:
        raise RecipeConfigError("dataset path does not exist")
    return ref.path


def _finish_diffusion(
    recipe: dict[str, Any], emit: Emit, prepared: PreparedRun
) -> TrainResult:
    train_steps = int(prepared.resolved["train_steps"])
    emit(
        {
            "event": "train_start",
            "family": "diffusion",
            "device": prepared.device,
            "train_steps": train_steps,
            "val_every": int(prepared.resolved["val_every"]),
        }
    )
    try:
        from ohho.train.il.diffusion_policy import train_diffusion

        checkpoint = train_diffusion(
            dataset=str(prepared.dataset),
            output_dir=str(Path(recipe["output"]["dir"]).expanduser()),
            device=prepared.device,
            train_steps=train_steps,
            batch_size=int(prepared.resolved["batch_size"]),
            lr=float(prepared.resolved["lr"]),
            chunk_size=int(prepared.resolved["chunk_size"]),
            vision_backbone=str(prepared.resolved["vision_backbone"]),
            diffusion_steps=int(prepared.resolved["diffusion_steps"]),
            scheduler=str(prepared.resolved["scheduler"]),
            num_inference_steps=int(prepared.resolved["num_inference_steps"]),
            d_model=int(prepared.resolved["d_model"]),
            log_every=int(prepared.resolved["log_every"]),
            val_every=int(prepared.resolved["val_every"]),
            val_fraction=float(prepared.resolved["val_fraction"]),
            early_stop_patience=int(prepared.resolved["early_stop_patience"]),
            early_stop_min_delta=float(prepared.resolved["early_stop_min_delta"]),
            on_event=_family_event("diffusion", emit),
        )
    except TrainFailed:
        raise
    except Exception as exc:
        raise TrainFailed(f"Diffusion training failed: {exc}") from exc
    metrics = Path(checkpoint) / "metrics.json"
    if not metrics.is_file():
        raise TrainFailed(f"Diffusion training did not write {metrics}")
    return TrainResult(
        checkpoint_dir=str(Path(checkpoint)),
        metrics_path=str(metrics),
        device=prepared.device,
        base_weights_loaded=False,
    )


def _family_event(family_id: str, emit: Emit) -> Emit:
    def _on_event(payload: dict[str, Any]) -> None:
        body = dict(payload)
        body.setdefault("family", family_id)
        emit(body)

    return _on_event
