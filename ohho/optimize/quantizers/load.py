"""Load a quantized artifact in the runtime that serves it.

AWQ and GPTQ use ``from_quantized``. bitsandbytes loads the saved state
dict. ORT static INT8 opens a CPU session. TensorRT FP8 deserializes the
engine. Each import stays inside the branch that needs it.
"""

from __future__ import annotations

from pathlib import Path

from ohho.optimize.errors import OptimizeCapabilityError, OptimizeError

_INSTALL = {
    "awq": "AWQ is not installed (package awq or autoawq).",
    "gptq": "GPTQ is not installed (package auto-gptq or gptqmodel).",
    "bitsandbytes": "bitsandbytes is not installed.",
    "ort-int8": "onnxruntime is not installed.",
    "fp8": "TensorRT is not installed.",
}


def load_quantized(path: str | Path, backend: str):
    """Open ``path`` with the serving runtime for ``backend``."""
    artifact = Path(path)
    if not artifact.exists():
        raise OptimizeError(f"Quantized artifact does not exist: {artifact}")
    if backend == "awq":
        return _awq(artifact)
    if backend == "gptq":
        return _gptq(artifact)
    if backend == "bitsandbytes":
        return _bitsandbytes(artifact)
    if backend == "ort-int8":
        return _ort(artifact)
    if backend == "fp8":
        return _fp8(artifact)
    known = "awq, gptq, bitsandbytes, ort-int8, fp8"
    raise OptimizeError(f"Unknown quantizer {backend!r}. Expected one of: {known}.")


def _awq(path: Path):
    try:
        from awq import AutoAWQForCausalLM
    except Exception as exc:
        raise OptimizeCapabilityError(_INSTALL["awq"]) from exc
    return AutoAWQForCausalLM.from_quantized(str(path))


def _gptq(path: Path):
    try:
        from auto_gptq import AutoGPTQForCausalLM
    except Exception as exc:
        raise OptimizeCapabilityError(_INSTALL["gptq"]) from exc
    return AutoGPTQForCausalLM.from_quantized(str(path))


def _bitsandbytes(path: Path):
    try:
        import torch
    except Exception as exc:
        raise OptimizeCapabilityError(_INSTALL["bitsandbytes"]) from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _ort(path: Path):
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise OptimizeCapabilityError(_INSTALL["ort-int8"]) from exc
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _fp8(path: Path):
    try:
        import tensorrt as trt
    except Exception as exc:
        raise OptimizeCapabilityError(_INSTALL["fp8"]) from exc
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise OptimizeError("TensorRT did not deserialize the FP8 engine.")
    return engine
