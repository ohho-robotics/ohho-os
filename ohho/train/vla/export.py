"""Merge a LoRA adapter and write the Serve checkpoint directory.

The directory is ``config.json`` plus ``policy.pt``. ``policy.pt`` is a plain
state dict (adapter tensors are folded in when ``tune_mode`` is ``lora``).
Nothing in this module downloads weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..recipe_schema import RecipeConfigError
from .licence import checkpoint_metadata
from .params import resolve_hyperparameters


def merge_lora_for_serve(
    model: Any,
    output_dir: str | Path,
    *,
    family: str,
    hyperparameters: Mapping[str, Any] | None = None,
    base_weights_uri: str = "",
) -> Path:
    """Write a Serve checkpoint for ``model``.

    A LoRA config wraps a plain module with the recipe rank, alpha, dropout,
    and target modules, then merges the adapter. A module that already has a
    PEFT adapter is merged as it stands. A full fine-tune writes the module
    state dict unchanged. ``config.json`` records the family licence metadata.
    """
    resolved = resolve_hyperparameters(
        family, {} if hyperparameters is None else hyperparameters
    )
    torch = _load_torch()
    merged, did_merge = _merged_module(model, resolved)
    folder = Path(output_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().clone() for key, value in merged.state_dict().items()
    }
    torch.save(state, folder / "policy.pt")
    metadata = checkpoint_metadata(family, resolved, base_weights_uri=base_weights_uri)
    metadata["merged_lora"] = did_merge
    (folder / "config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return folder


def _merged_module(model: Any, resolved: Mapping[str, Any]) -> tuple[Any, bool]:
    if _is_peft_model(model):
        return model.merge_and_unload(), True
    if resolved["tune_mode"] != "lora":
        return model, False
    lora_config, get_peft_model = _load_peft()
    config = lora_config(
        r=int(resolved["lora_rank"]),
        lora_alpha=int(resolved["lora_alpha"]),
        lora_dropout=float(resolved["lora_dropout"]),
        target_modules=_peft_targets(tuple(resolved["lora_target_modules"])),
        bias="none",
    )
    return get_peft_model(model, config).merge_and_unload(), True


def _peft_targets(targets: tuple[str, ...]) -> str | list[str]:
    if targets == ("all-linear",):
        return "all-linear"
    return list(targets)


def _is_peft_model(model: Any) -> bool:
    return hasattr(model, "merge_and_unload") and hasattr(model, "peft_config")


def _import_torch() -> Any:
    import torch

    return torch


def _load_torch() -> Any:
    try:
        return _import_torch()
    except ImportError as exc:
        raise RecipeConfigError(
            "LoRA export needs torch. Install it with: pip install 'ohho-os[train]'"
        ) from exc


def _import_peft() -> tuple[Any, Any]:
    from peft import LoraConfig, get_peft_model

    return LoraConfig, get_peft_model


def _load_peft() -> tuple[Any, Any]:
    try:
        return _import_peft()
    except ImportError as exc:
        raise RecipeConfigError(
            "LoRA merge needs peft. Install it with: pip install peft"
        ) from exc
