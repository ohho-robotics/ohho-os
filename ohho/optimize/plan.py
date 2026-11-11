"""Validate an optimize request against a hardware target profile.

The profile rules live in :mod:`ohho.targets`. This module adds the engine
rules: which flag applies a precision, when INT8 needs a calibrator, and
when a workspace does not fit the memory budget. It does not import
PyTorch, ONNX Runtime, or TensorRT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ohho.targets.errors import TargetProfileError
from ohho.targets.providers import TRT_PROVIDER
from ohho.targets.schema import (
    TargetProfile,
    check_combination,
    get_profile,
    load_profile,
)

from ohho.optimize.errors import OptimizeError
from ohho.optimize.qdq import MISSING_CALIBRATION, model_has_qdq

_QUANT = frozenset({"awq", "gptq", "bitsandbytes", "fp8"})
_VERSION = re.compile(r"^(\d+\.\d+\.\d+)")


@dataclass(frozen=True)
class OptimizeRequest:
    """One ``ohho optimize`` invocation.

    ``calib_batch`` is ``None`` when the user did not pass ``--calib-batch``.
    ``workspace_gb`` is ``None`` when the profile default should be used.
    """

    model: str
    target: str
    precision: str
    out_dir: str
    onnx: bool = False
    tensorrt: bool = False
    calib_data: str | None = None
    calib_batch: int | None = None
    calib_cache: str | None = None
    calib_max_images: int | None = None
    ort_ep: str | None = None
    quant: str | None = None
    workspace_gb: float | None = None
    max_batch: int = 1
    dynamic_batch: bool = False
    submodule: str | None = None
    allow_parity_fail: bool = False
    parity_inputs: str = "random"
    parity_n: int = 256
    parity_seed: int = 0
    parity_mean_abs: float | None = None
    parity_thresholds: str | None = None
    parity_family: str | None = None


@dataclass(frozen=True)
class OptimizePlan:
    """A request that passed the target profile and the engine rules."""

    request: OptimizeRequest
    profile: TargetProfile
    precision: str
    provider: str
    export_onnx: bool
    build_tensorrt: bool
    build_session: bool
    quant: str | None
    workspace_bytes: int
    calib_batch: int | None


def installed_trt_version() -> str:
    """Dotted TensorRT version, or ``""`` when TensorRT is not installed."""
    try:
        import tensorrt as trt
    except Exception:
        return ""
    text = str(getattr(trt, "__version__", "")).strip()
    match = _VERSION.match(text)
    if match is None:
        return ""
    return match.group(1)


def resolve_target(ref: str) -> TargetProfile:
    """Load a built-in profile id or a ``.json`` / ``.yaml`` profile file."""
    text = ref.strip()
    if not text:
        raise OptimizeError("--target is required.")
    path = Path(text)
    try:
        if path.is_file():
            return load_profile(path)
        return get_profile(text)
    except TargetProfileError as exc:
        raise OptimizeError(str(exc)) from exc


def build_plan(request: OptimizeRequest) -> OptimizePlan:
    """Accept the request or raise :class:`OptimizeError`."""
    _require_text(request.model, "--model")
    _require_text(request.out_dir, "--out")
    if request.max_batch < 1:
        raise OptimizeError("--max-batch must be at least 1.")
    if request.parity_n < 1:
        raise OptimizeError("--parity-n must be at least 1.")
    if type(request.parity_seed) is not int:
        raise OptimizeError("--parity-seed must be an integer.")
    if not isinstance(request.parity_inputs, str) or not request.parity_inputs.strip():
        raise OptimizeError("--parity-inputs is required.")
    if request.parity_mean_abs is not None and (
        isinstance(request.parity_mean_abs, bool)
        or not isinstance(request.parity_mean_abs, (int, float))
        or float(request.parity_mean_abs) < 0
    ):
        raise OptimizeError("--parity-mean-abs must be >= 0.")
    if request.calib_batch is not None and request.calib_batch < 1:
        raise OptimizeError("--calib-batch must be at least 1.")
    if request.calib_max_images is not None and request.calib_max_images < 1:
        raise OptimizeError("--calib-max-images must be at least 1.")
    if request.quant is not None and request.quant not in _QUANT:
        known = ", ".join(sorted(_QUANT))
        raise OptimizeError(
            f"Unknown quantizer {request.quant!r}. Expected one of: {known}."
        )
    if request.tensorrt and request.ort_ep not in (None, "tensorrt"):
        raise OptimizeError(
            "--tensorrt builds a TensorRT engine, so --ort-ep must be "
            "tensorrt or omitted."
        )

    profile = resolve_target(request.target)
    provider_text = _provider_text(request, profile)
    version = ""
    if provider_text.strip().lower() in {"tensorrt", "trt"} or (
        provider_text.strip() == TRT_PROVIDER
    ):
        version = installed_trt_version()
    try:
        precision, provider = check_combination(
            profile,
            request.precision,
            provider_text,
            tensorrt_version=version,
        )
    except TargetProfileError as exc:
        raise OptimizeError(str(exc)) from exc

    if request.tensorrt and TRT_PROVIDER not in profile.execution_providers:
        raise OptimizeError(
            f"TensorRT is not valid on {profile.id}. Execution providers: "
            f"{', '.join(profile.execution_providers)}."
        )

    workspace_bytes = _workspace_bytes(request, profile)
    _reject_calibration_on_other_precisions(request, precision)
    _reject_unapplied_precision(request, profile, precision)
    calib_batch = _calib_batch(request, precision)

    quant_only = request.quant is not None and not request.onnx and not request.tensorrt
    export_onnx = not quant_only
    build_tensorrt = request.tensorrt
    build_session = export_onnx and not (request.tensorrt and request.ort_ep is None)
    if request.ort_ep and export_onnx:
        build_session = True

    return OptimizePlan(
        request=request,
        profile=profile,
        precision=precision,
        provider=provider,
        export_onnx=export_onnx,
        build_tensorrt=build_tensorrt,
        build_session=build_session,
        quant=request.quant,
        workspace_bytes=workspace_bytes,
        calib_batch=calib_batch,
    )


def _require_text(value: str, flag: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise OptimizeError(f"{flag} is required.")


def _provider_text(request: OptimizeRequest, profile: TargetProfile) -> str:
    if request.ort_ep:
        return request.ort_ep
    if request.tensorrt:
        return "tensorrt"
    return profile.execution_providers[0]


def _workspace_bytes(request: OptimizeRequest, profile: TargetProfile) -> int:
    if request.workspace_gb is None:
        return profile.workspace_size_mib * (1 << 20)
    if request.workspace_gb <= 0:
        raise OptimizeError("--workspace-gb must be positive.")
    mib = int(round(float(request.workspace_gb) * 1024))
    if mib < 1:
        raise OptimizeError("--workspace-gb is smaller than 1 MiB.")
    if mib > profile.memory_budget_mib:
        raise OptimizeError(
            f"Workspace {mib} MiB exceeds the {profile.id} memory budget "
            f"of {profile.memory_budget_mib} MiB."
        )
    return int(round(float(request.workspace_gb) * (1 << 30)))


def _reject_calibration_on_other_precisions(
    request: OptimizeRequest, precision: str
) -> None:
    asked = (
        request.calib_data is not None
        or request.calib_cache is not None
        or request.calib_batch is not None
        or request.calib_max_images is not None
    )
    if asked and precision != "INT8":
        raise OptimizeError(
            "Calibration data, cache, batch size, and max images apply only to "
            "INT8 TensorRT builds."
        )


def _calib_batch(request: OptimizeRequest, precision: str) -> int | None:
    if precision != "INT8" or not request.tensorrt:
        return None
    batch = request.calib_batch or 1
    if request.calib_max_images is not None and request.calib_max_images < batch:
        raise OptimizeError(
            "--calib-max-images must be at least --calib-batch "
            f"({request.calib_max_images} < {batch})."
        )
    cache_ready = bool(request.calib_cache) and Path(str(request.calib_cache)).is_file()
    if request.calib_data or cache_ready:
        return batch
    if model_has_qdq(request.model):
        return request.calib_batch
    raise OptimizeError(MISSING_CALIBRATION)


def _reject_unapplied_precision(
    request: OptimizeRequest,
    profile: TargetProfile,
    precision: str,
) -> None:
    if precision == "FP32":
        return
    if precision == "FP16":
        if request.tensorrt:
            _require_positive_workspace(request, profile)
            return
        raise OptimizeError(
            "FP16 needs --tensorrt. ONNX Runtime runs the exported FP32 graph; "
            f"it does not build an FP16 engine for {profile.id}."
        )
    if precision == "FP8":
        if request.tensorrt or request.quant == "fp8":
            if request.tensorrt:
                _require_positive_workspace(request, profile)
            return
        raise OptimizeError(
            "FP8 needs --tensorrt, or the fp8 weight quantizer. "
            f"{profile.id} lists FP8, and this engine does not rewrite the "
            "ONNX graph to FP8 by itself."
        )
    if precision == "INT8":
        if request.tensorrt:
            _require_positive_workspace(request, profile)
            return
        if request.quant in {"awq", "gptq", "bitsandbytes"}:
            return
        raise OptimizeError(
            "INT8 needs --tensorrt with --calib-data on a TensorRT target, "
            "or --quant awq, --quant gptq, or the bitsandbytes backend. "
            f"A plain ONNX export for {profile.id} stays FP32."
        )
    if precision == "INT4":
        if request.quant in {"awq", "gptq", "bitsandbytes"}:
            return
        raise OptimizeError(
            "INT4 needs --quant awq or --quant gptq "
            "(bitsandbytes is available to the Python API)."
        )


def _require_positive_workspace(
    request: OptimizeRequest, profile: TargetProfile
) -> None:
    if request.workspace_gb is None and profile.workspace_size_mib < 1:
        raise OptimizeError(
            f"{profile.id} has no TensorRT workspace. Pass --workspace-gb "
            "within the memory budget, or choose a TensorRT target."
        )
