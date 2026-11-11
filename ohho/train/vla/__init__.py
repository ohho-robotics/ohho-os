"""SmolVLA and OpenVLA fine-tune recipes.

Importing this package does not import torch, peft, or a base-model checkpoint.
"""

from __future__ import annotations

from .export import merge_lora_for_serve
from .family import OPENVLA, SMOLVLA, OpenVLARecipe, PreparedVLA, SmolVLARecipe, prepare
from .licence import base_model_record, checkpoint_metadata
from .memory import MemoryEstimate, estimate_memory
from .params import documented_parameter_lines, resolve_hyperparameters

__all__ = [
    "MemoryEstimate",
    "OPENVLA",
    "OpenVLARecipe",
    "PreparedVLA",
    "SMOLVLA",
    "SmolVLARecipe",
    "base_model_record",
    "checkpoint_metadata",
    "documented_parameter_lines",
    "estimate_memory",
    "merge_lora_for_serve",
    "prepare",
    "resolve_hyperparameters",
]
