"""SmolVLA and OpenVLA recipe families.

``available()`` is true so ``ohho train`` runs parameter preflight. ``run``
then raises :class:`~ohho.train.recipe_schema.FamilyUnavailable`. This build
does not download base weights and does not launch the fine-tune. Call
:func:`estimate_memory` before a GPU launch, and :func:`merge_lora_for_serve`
when a trained adapter should be folded for Serve.

BF16 is ``hyperparameters.precision``. The shared recipe ``precision`` field
is not read here; OHH-524 will unify it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..recipe_registry import Emit, TrainResult
from ..recipe_schema import FamilyUnavailable
from .licence import checkpoint_metadata, recipe_base_weights_uri
from .memory import MemoryEstimate, estimate_memory
from .params import recipe_hyperparameters, resolve_hyperparameters


@dataclass(frozen=True)
class PreparedVLA:
    """Checked hyperparameters, memory warning, and licence metadata."""

    resolved: dict[str, Any]
    memory: MemoryEstimate
    metadata: dict[str, Any]


class _VLARecipe:
    """Parameter preflight for one VLA family. The weight loader is not in this build."""

    id = ""

    def available(self) -> bool:
        """True so preflight runs. The trainer that loads base weights is absent."""
        return True

    def preflight(self, recipe: dict[str, Any]) -> None:
        """Validate parameters, then stop before any weight download."""
        prepare(self.id, recipe)
        raise FamilyUnavailable(self.id)

    def run(self, recipe: dict[str, Any], emit: Emit) -> TrainResult:
        """Validate parameters, then stop before any weight download."""
        del emit
        prepare(self.id, recipe)
        raise FamilyUnavailable(self.id)


class SmolVLARecipe(_VLARecipe):
    """SmolVLA fine-tune. The default launch class is one 24 GB GPU."""

    id = "smolvla"


class OpenVLARecipe(_VLARecipe):
    """OpenVLA LoRA or full fine-tune.

    LoRA's launch class is one A100 80GB. Full fine-tune is multi-GPU/cloud only.
    """

    id = "openvla"


SMOLVLA = SmolVLARecipe()
OPENVLA = OpenVLARecipe()


def prepare(family: str, recipe: Mapping[str, Any]) -> PreparedVLA:
    """Resolve parameters, estimate memory, and build licence metadata.

    Does not train and does not download weights.
    """
    raw = recipe_hyperparameters(recipe)
    resolved = resolve_hyperparameters(family, raw)
    memory = estimate_memory(family, resolved)
    metadata = checkpoint_metadata(
        family,
        resolved,
        base_weights_uri=recipe_base_weights_uri(recipe),
    )
    return PreparedVLA(resolved=resolved, memory=memory, metadata=metadata)
