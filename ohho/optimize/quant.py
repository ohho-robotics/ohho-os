"""Weight-quant backends for LLM and VLM checkpoints.

AWQ, GPTQ, bitsandbytes, and FP8 share one :func:`quantize` entry. Each
backend is an optional import. A missing package or a missing CUDA device
raises :class:`OptimizeCapabilityError` and does not write a fake checkpoint.
The CLI exposes ``awq`` and ``gptq``. ``bitsandbytes`` and ``fp8`` are the
same interface for callers that already hold a module.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ohho.optimize.errors import OptimizeCapabilityError, OptimizeError
from ohho.optimize.models import LoadedModel

_BACKENDS = ("awq", "gptq", "bitsandbytes", "fp8")
_PACKAGES = {
    "awq": ("awq", "autoawq"),
    "gptq": ("auto_gptq", "gptqmodel"),
    "bitsandbytes": ("bitsandbytes",),
}
_MISSING = {
    "awq": (
        "AWQ is not installed (package awq or autoawq). "
        "INT4 and INT8 weight quant is a GPU follow-up."
    ),
    "gptq": (
        "GPTQ is not installed (package auto-gptq or gptqmodel). "
        "INT4 and INT8 weight quant is a GPU follow-up."
    ),
    "bitsandbytes": (
        "bitsandbytes is not installed. INT4 and INT8 kernels are a GPU follow-up."
    ),
    "fp8": "FP8 weight quant needs PyTorch with float8 dtypes.",
}


def probe_quant(name: str) -> tuple[bool, str]:
    """Return whether ``name`` can run in this process, and a short reason."""
    if name not in _BACKENDS:
        known = ", ".join(_BACKENDS)
        return False, f"Unknown quantizer {name!r}. Expected one of: {known}."
    if name == "fp8":
        if not _has_fp8_dtype():
            return False, _MISSING["fp8"]
        if not _cuda():
            return False, "FP8 weight quant needs a CUDA device. GPU follow-up."
        return True, "torch.float8_e4m3fn"
    for package in _PACKAGES[name]:
        if importlib.util.find_spec(package) is not None:
            if name == "bitsandbytes" and not _cuda():
                return False, "bitsandbytes needs a CUDA device. GPU follow-up."
            return True, package
    return False, _MISSING[name]


def available_quantizers() -> dict[str, tuple[bool, str]]:
    """Probe every backend. Imports stay inside the probe."""
    return {name: probe_quant(name) for name in _BACKENDS}


def quantize(
    model: LoadedModel,
    backend: str,
    out_dir: str | Path,
    precision: str,
    *,
    examples=None,
) -> Path:
    """Run one backend. Raises when the package or the checkpoint does not fit."""
    ok, detail = probe_quant(backend)
    if not ok:
        raise OptimizeCapabilityError(detail)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if backend == "awq":
        return _run_awq(model, destination, precision)
    if backend == "gptq":
        return _run_gptq(model, destination, precision, examples)
    if backend == "bitsandbytes":
        return _run_bitsandbytes(model, destination, precision)
    if backend == "fp8":
        return _run_fp8(model, destination, precision)
    raise OptimizeError(detail)


def replace_linear(module, factory) -> int:
    """Swap each ``nn.Linear`` for ``factory(in_features, out_features, bias=...)``.

    Returns the number of layers replaced. ``factory`` is the bitsandbytes
    class, or a test double. Child modules are walked, and a replaced layer
    is not walked again.
    """
    import torch.nn as nn

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            made = factory(
                int(child.in_features),
                int(child.out_features),
                bias=child.bias is not None,
            )
            setattr(module, name, made)
            replaced += 1
        else:
            replaced += replace_linear(child, factory)
    return replaced


def _run_awq(model: LoadedModel, out_dir: Path, precision: str) -> Path:
    directory = _hf_directory(model)
    try:
        from awq import AutoAWQForCausalLM
    except Exception as exc:
        raise OptimizeCapabilityError(_MISSING["awq"]) from exc
    bits = 4 if precision == "INT4" else 8
    lm = AutoAWQForCausalLM.from_pretrained(str(directory))
    lm.quantize(
        quant_config={
            "w_bit": bits,
            "q_group_size": 128,
            "zero_point": True,
            "version": "GEMM",
        }
    )
    dest = out_dir / "awq"
    dest.mkdir(parents=True, exist_ok=True)
    lm.save_quantized(str(dest))
    return dest


def _run_gptq(model: LoadedModel, out_dir: Path, precision: str, examples) -> Path:
    directory = _hf_directory(model)
    if not examples:
        raise OptimizeError(
            "GPTQ needs calibration examples. Pass examples= to quantize(). "
            "ohho optimize does not download a dataset."
        )
    bits = 4 if precision == "INT4" else 8
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except Exception as exc:
        raise OptimizeCapabilityError(_MISSING["gptq"]) from exc
    config = BaseQuantizeConfig(bits=bits, group_size=128, desc_act=False)
    lm = AutoGPTQForCausalLM.from_pretrained(str(directory), quantize_config=config)
    lm.quantize(examples)
    dest = out_dir / "gptq"
    dest.mkdir(parents=True, exist_ok=True)
    lm.save_quantized(str(dest))
    return dest


def _run_bitsandbytes(model: LoadedModel, out_dir: Path, precision: str) -> Path:
    if model.module is None:
        raise OptimizeError("bitsandbytes needs a PyTorch module.")
    try:
        import bitsandbytes as bnb
        import torch
    except Exception as exc:
        raise OptimizeCapabilityError(_MISSING["bitsandbytes"]) from exc
    if precision == "INT4":
        factory = bnb.nn.Linear4bit
    else:
        factory = bnb.nn.Linear8bitLt
    count = replace_linear(model.module, factory)
    if count < 1:
        raise OptimizeError("bitsandbytes found no Linear layers to replace.")
    dest = out_dir / "bitsandbytes.pt"
    torch.save(model.module.state_dict(), dest)
    return dest


def _run_fp8(model: LoadedModel, out_dir: Path, precision: str) -> Path:
    if precision != "FP8":
        raise OptimizeError("The fp8 backend requires precision FP8.")
    if model.module is None:
        raise OptimizeError("FP8 weight quant needs a PyTorch module.")
    import torch

    if not torch.cuda.is_available():
        raise OptimizeCapabilityError(
            "FP8 weight quant needs a CUDA device. GPU follow-up."
        )
    dtype = torch.float8_e4m3fn
    with torch.no_grad():
        for param in model.module.parameters():
            if param.is_floating_point():
                param.data = param.data.to(device="cuda", dtype=dtype)
    dest = out_dir / "fp8.pt"
    torch.save(model.module.state_dict(), dest)
    return dest


def _hf_directory(model: LoadedModel) -> Path:
    path = model.path
    if path is None:
        raise OptimizeError(
            f"{model.kind} model {model.ref!r} is not a local model directory. "
            "AWQ and GPTQ need a directory that contains config.json."
        )
    directory = path if path.is_dir() else path.parent
    if not (directory / "config.json").is_file():
        raise OptimizeError(
            "AWQ and GPTQ need a local model directory that contains config.json. "
            "No dataset is downloaded."
        )
    return directory


def _has_fp8_dtype() -> bool:
    try:
        import torch
    except Exception:
        return False
    return hasattr(torch, "float8_e4m3fn")


def _cuda() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False
