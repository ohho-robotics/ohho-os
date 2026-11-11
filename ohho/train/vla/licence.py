"""Published base-model licence recorded in VLA checkpoint metadata.

The strings are the licences published with the checkpoints. This module
does not fetch a model card and does not download weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..recipe_schema import RecipeConfigError
from .params import resolve_hyperparameters

SMOLVLA_MODEL_ID = "lerobot/smolvla_base"
OPENVLA_MODEL_ID = "openvla/openvla-7b"
SMOLVLA_PARAMETERS = 450_000_000
OPENVLA_PARAMETERS = 7_500_000_000

_SMOLVLA_NOTE = "The lerobot/smolvla_base model card lists Apache-2.0."
_OPENVLA_NOTE = (
    "The OpenVLA README states that openvla-7b is derived from Llama-2 and "
    "those weights are subject to the Llama 2 Community License. The Hugging "
    "Face model card also lists MIT."
)
_SMOLVLA_COUNT_NOTE = (
    "Planning constant for the published SmolVLA 450M checkpoint "
    "(lerobot/smolvla_base). Weights are not loaded to count parameters."
)
_OPENVLA_COUNT_NOTE = (
    "Planning constant. The OpenVLA README describes this checkpoint as "
    "7.5 billion parameters. Weights are not loaded to count parameters."
)


@dataclass(frozen=True)
class BaseModelRecord:
    """Licence and identity of the published checkpoint a family starts from."""

    family: str
    model_id: str
    licence: str
    card_licence: str
    note: str
    parameter_count: int
    parameter_count_note: str


_RECORDS = {
    "smolvla": BaseModelRecord(
        family="smolvla",
        model_id=SMOLVLA_MODEL_ID,
        licence="Apache-2.0",
        card_licence="Apache-2.0",
        note=_SMOLVLA_NOTE,
        parameter_count=SMOLVLA_PARAMETERS,
        parameter_count_note=_SMOLVLA_COUNT_NOTE,
    ),
    "openvla": BaseModelRecord(
        family="openvla",
        model_id=OPENVLA_MODEL_ID,
        licence="Llama 2 Community License",
        card_licence="MIT",
        note=_OPENVLA_NOTE,
        parameter_count=OPENVLA_PARAMETERS,
        parameter_count_note=_OPENVLA_COUNT_NOTE,
    ),
}


def base_model_record(family: str) -> BaseModelRecord:
    """Return the published base-model record for ``family``."""
    try:
        return _RECORDS[family]
    except KeyError as exc:
        raise RecipeConfigError(f"no base-model licence for family {family!r}") from exc


def recipe_base_weights_uri(recipe: Mapping[str, Any]) -> str:
    """Return ``base_weights.uri`` when the recipe sets one, else ``\"\"``."""
    weights = recipe.get("base_weights")
    if isinstance(weights, Mapping):
        uri = weights.get("uri")
        if isinstance(uri, str):
            return uri
    return ""


def checkpoint_metadata(
    family: str,
    hyperparameters: Mapping[str, Any],
    *,
    base_weights_uri: str = "",
) -> dict[str, Any]:
    """Metadata written beside an exported Serve checkpoint.

    ``base_model_licence`` is the published weight licence for this family.
    ``base_weights_uri`` records the recipe uri and does not change that licence.
    ``merged_lora`` is true when ``tune_mode`` is ``lora``; an export that
    actually folds an adapter sets the same flag on the file it writes.
    """
    resolved = resolve_hyperparameters(family, hyperparameters)
    record = base_model_record(family)
    return {
        "policy": family,
        "model_type": family,
        "model_family": family,
        "mock": False,
        "precision": resolved["precision"],
        "tune_mode": resolved["tune_mode"],
        "gradient_checkpointing": resolved["gradient_checkpointing"],
        "lora_rank": resolved["lora_rank"],
        "lora_alpha": resolved["lora_alpha"],
        "lora_dropout": resolved["lora_dropout"],
        "lora_target_modules": list(resolved["lora_target_modules"]),
        "merged_lora": resolved["tune_mode"] == "lora",
        "base_model": record.model_id,
        "base_model_licence": record.licence,
        "base_model_card_licence": record.card_licence,
        "base_model_licence_note": record.note,
        "base_weights_uri": base_weights_uri,
        "parameter_count": record.parameter_count,
        "parameter_count_note": record.parameter_count_note,
    }
