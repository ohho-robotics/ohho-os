"""Planning memory estimate for SmolVLA and OpenVLA fine-tunes.

The OpenVLA README's LoRA example (rank 32, ``all-linear``, batch 16, one
A100 80GB) is published at about 72 GB, and a reduced batch is published at
about 27 GB. Those two points anchor the OpenVLA LoRA estimate: 24 GiB fixed
plus 3 GiB per sample at BF16 with gradient checkpointing. SmolVLA's launch
class is one 24 GB GPU. Full OpenVLA fine-tune is multi-GPU/cloud only
(upstream uses a node of 8 A100 80GB with FSDP).

The result is a warning a UI or CLI can show before launch. It does not
allocate a model and does not download weights. Gradient accumulation does
not scale the estimate: ``batch_size`` is the microbatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .licence import OPENVLA_PARAMETERS, base_model_record
from .params import resolve_hyperparameters

_GIB = 1024**3
CLASS_24GB = "24gb"
CLASS_A100_80GB = "a100-80gb"
CLASS_MULTI_GPU = "multi-gpu-cloud"
_LIMIT_GIB = {
    CLASS_24GB: 24.0,
    CLASS_A100_80GB: 80.0,
    CLASS_MULTI_GPU: None,
}
_RECOMMENDED = {
    CLASS_24GB: "single 24 GB GPU",
    CLASS_A100_80GB: "single A100 80GB",
    CLASS_MULTI_GPU: "multi-GPU/cloud (8x A100 80GB, FSDP)",
}
_SMOLVLA_SAMPLE_BYTES = (5 * _GIB) // 4
_OPENVLA_SAMPLE_BYTES = 3 * _GIB
_SMOLVLA_OVERHEAD_BYTES = 2 * _GIB


def trainable_count(
    parameter_count: int,
    tune_mode: str,
    rank: int,
    targets: tuple[str, ...],
) -> int:
    """Trainable parameters used by the estimate.

    LoRA uses 1.4% of the checkpoint at rank 32 (the OpenVLA paper's
    all-linear figure) and scales linearly with rank. An explicit module list
    scales that by ``len(targets) / 7`` (the seven Llama projection names
    ``all-linear`` covers). Full fine-tune trains every parameter.
    """
    if tune_mode == "full":
        return parameter_count
    scaled = parameter_count * 14 * rank // (1000 * 32)
    if targets == ("all-linear",):
        return scaled
    return scaled * len(targets) // 7


def _component_bytes(
    parameter_count: int,
    tune_mode: str,
    rank: int,
    targets: tuple[str, ...],
    bytes_per_parameter: int,
) -> int:
    weight, grad, optim, adapter, _trainable = _parts(
        parameter_count, tune_mode, rank, targets, bytes_per_parameter
    )
    return weight + grad + optim + adapter


def _parts(
    parameter_count: int,
    tune_mode: str,
    rank: int,
    targets: tuple[str, ...],
    bytes_per_parameter: int,
) -> tuple[int, int, int, int, int]:
    trainable = trainable_count(parameter_count, tune_mode, rank, targets)
    weight = parameter_count * bytes_per_parameter
    grad = trainable * bytes_per_parameter
    optim = trainable * 8
    adapter = trainable * bytes_per_parameter if tune_mode == "lora" else 0
    return weight, grad, optim, adapter, trainable


# Residual that makes the OpenVLA LoRA anchor (rank 32, all-linear, BF16,
# gradient checkpointing, batch 16) land on 72 GiB. Batch 1 then lands on 27 GiB.
_OPENVLA_OVERHEAD_BYTES = (
    72 * _GIB
    - _component_bytes(OPENVLA_PARAMETERS, "lora", 32, ("all-linear",), 2)
    - 16 * _OPENVLA_SAMPLE_BYTES
)


@dataclass(frozen=True)
class MemoryEstimate:
    """One config's planning estimate and the launch-class warning."""

    family: str
    tune_mode: str
    precision: str
    batch_size: int
    grad_accumulation_steps: int
    gradient_checkpointing: bool
    lora_rank: int
    parameter_count: int
    trainable_parameters: int
    weights_bytes: int
    gradients_bytes: int
    optimizer_bytes: int
    adapters_bytes: int
    activations_bytes: int
    overhead_bytes: int
    total_bytes: int
    total_gib: float
    gpu_class: str
    recommended_gpu: str
    launch_limit_gib: float | None
    fits_launch_class: bool
    warning: str

    def as_dict(self) -> dict[str, Any]:
        """Plain mapping a UI can serialise. ``launch_limit_gib`` is null for multi-GPU."""
        return {
            "family": self.family,
            "tune_mode": self.tune_mode,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "lora_rank": self.lora_rank,
            "parameter_count": self.parameter_count,
            "trainable_parameters": self.trainable_parameters,
            "weights_bytes": self.weights_bytes,
            "gradients_bytes": self.gradients_bytes,
            "optimizer_bytes": self.optimizer_bytes,
            "adapters_bytes": self.adapters_bytes,
            "activations_bytes": self.activations_bytes,
            "overhead_bytes": self.overhead_bytes,
            "total_bytes": self.total_bytes,
            "total_gib": self.total_gib,
            "gpu_class": self.gpu_class,
            "recommended_gpu": self.recommended_gpu,
            "launch_limit_gib": self.launch_limit_gib,
            "fits_launch_class": self.fits_launch_class,
            "warning": self.warning,
        }


def estimate_memory(
    family: str, hyperparameters: Mapping[str, Any] | None = None
) -> MemoryEstimate:
    """Estimate device memory for a VLA config and warn when it misses its class.

    ``hyperparameters`` is a raw or already resolved family mapping. Unknown
    keys and out-of-range values raise :class:`RecipeConfigError`.
    """
    resolved = resolve_hyperparameters(
        family, {} if hyperparameters is None else hyperparameters
    )
    record = base_model_record(family)
    precision = str(resolved["precision"])
    tune_mode = str(resolved["tune_mode"])
    rank = int(resolved["lora_rank"])
    targets = tuple(resolved["lora_target_modules"])
    batch = int(resolved["batch_size"])
    checkpointing = bool(resolved["gradient_checkpointing"])
    bpp = 4 if precision == "fp32" else 2
    weight, grad, optim, adapter, trainable = _parts(
        record.parameter_count, tune_mode, rank, targets, bpp
    )
    sample, overhead = _sample_and_overhead(family)
    activations = sample * batch
    if precision == "fp32":
        activations *= 2
    if not checkpointing:
        activations *= 3
    total = weight + grad + optim + adapter + activations + overhead
    total_gib = round(total / _GIB, 3)
    gpu_class = _gpu_class(family, tune_mode)
    limit = _LIMIT_GIB[gpu_class]
    fits = False if limit is None else total_gib <= limit
    warning = _warning(family, tune_mode, gpu_class, total_gib, fits)
    return MemoryEstimate(
        family=family,
        tune_mode=tune_mode,
        precision=precision,
        batch_size=batch,
        grad_accumulation_steps=int(resolved["grad_accumulation_steps"]),
        gradient_checkpointing=checkpointing,
        lora_rank=rank,
        parameter_count=record.parameter_count,
        trainable_parameters=trainable,
        weights_bytes=weight,
        gradients_bytes=grad,
        optimizer_bytes=optim,
        adapters_bytes=adapter,
        activations_bytes=activations,
        overhead_bytes=overhead,
        total_bytes=total,
        total_gib=total_gib,
        gpu_class=gpu_class,
        recommended_gpu=_RECOMMENDED[gpu_class],
        launch_limit_gib=limit,
        fits_launch_class=fits,
        warning=warning,
    )


def _sample_and_overhead(family: str) -> tuple[int, int]:
    if family == "openvla":
        return _OPENVLA_SAMPLE_BYTES, _OPENVLA_OVERHEAD_BYTES
    return _SMOLVLA_SAMPLE_BYTES, _SMOLVLA_OVERHEAD_BYTES


def _gpu_class(family: str, tune_mode: str) -> str:
    if family == "openvla" and tune_mode == "full":
        return CLASS_MULTI_GPU
    if family == "openvla":
        return CLASS_A100_80GB
    return CLASS_24GB


def _warning(
    family: str, tune_mode: str, gpu_class: str, total_gib: float, fits: bool
) -> str:
    if family == "openvla" and tune_mode == "full":
        return (
            "full OpenVLA fine-tune is multi-GPU/cloud only "
            "(upstream uses a node of 8 A100 80GB with FSDP); "
            f"estimated aggregate {total_gib:.1f} GiB does not fit one GPU"
        )
    if fits:
        return ""
    if gpu_class == CLASS_24GB:
        return f"estimated {total_gib:.1f} GiB exceeds the 24 GB SmolVLA launch class"
    return (
        f"estimated {total_gib:.1f} GiB exceeds the 80 GB OpenVLA LoRA launch class "
        "(single A100 80GB)"
    )
