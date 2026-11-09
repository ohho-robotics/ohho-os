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
from ohho.targets.providers import CPU_PROVIDER, TRT_PROVIDER
from ohho.targets.schema import (
    TargetProfile,
    check_combination,
    get_profile,
    load_profile,
)

from ohho.optimize.errors import OptimizeError
from ohho.optimize.qdq import MISSING_CALIBRATION, model_has_qdq

_QUANT = frozenset({"awq", "gptq", "bitsandbytes", "fp8", "ort-int8"})
_WEIGHT = frozenset({"awq", "gptq", "bitsandbytes"})
_VERSION = re.compile(r"^(\d+\.\d+\.\d+)")
PRECISION_FLAGS = (
    "fp32",
    "fp16",
    "int8",
    "fp8",
    "int4",
    "int4-awq",
    "int4-gptq",
    "int4-bnb",
    "nf4",
)
_PRECISION_ALIASES = {
    "fp32": ("FP32", None),
    "fp16": ("FP16", None),
    "int8": ("INT8", None),
    "fp8": ("FP8", "fp8"),
    "int4": ("INT4", None),
    "int4-awq": ("INT4", "awq"),
    "int4-gptq": ("INT4", "gptq"),
    "int4-bnb": ("INT4", "bitsandbytes"),
    "nf4": ("INT4", "bitsandbytes"),
}


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


def resolve_precision(value: str) -> tuple[str, str | None]:
    """Map a ``--precision`` flag to a canonical precision and a backend.

    ``int4-awq`` is INT4 plus the AWQ backend. ``fp32`` leaves the backend
    unset. The canonical token is what :func:`check_combination` accepts.
    """
    key = value.strip().lower()
    try:
        precision, quant = _PRECISION_ALIASES[key]
    except KeyError:
        known = ", ".join(PRECISION_FLAGS)
        raise OptimizeError(
            f"Unknown precision {value!r}. Expected one of: {known}."
        ) from None
    return precision, quant


def build_plan(request: OptimizeRequest) -> OptimizePlan:
    """Accept the request or raise :class:`OptimizeError`.

    The target profile is checked before a model is loaded and before a
    quantizer runs. FP8 on a GPU that is not Ada or Hopper fails here.
    """
    _require_text(request.model, "--model")
    _require_text(request.out_dir, "--out")
    if request.max_batch < 1:
        raise OptimizeError("--max-batch must be at least 1.")
    if request.calib_batch is not None and request.calib_batch < 1:
        raise OptimizeError("--calib-batch must be at least 1.")
    if request.calib_max_images is not None and request.calib_max_images < 1:
        raise OptimizeError("--calib-max-images must be at least 1.")
    if request.quant is not None and request.quant not in _QUANT:
        known = ", ".join(sorted(_QUANT))
        raise OptimizeError(
            f"Unknown quantizer {request.quant!r}. Expected one of: {known}."
        )
    canonical, implied = resolve_precision(request.precision)
    quant = _resolve_quant(request.quant, implied, request.precision)
    tensorrt = request.tensorrt or quant == "fp8"
    if request.tensorrt and request.ort_ep not in (None, "tensorrt"):
        raise OptimizeError(
            "--tensorrt builds a TensorRT engine, so --ort-ep must be "
            "tensorrt or omitted."
        )
    if quant == "fp8" and request.ort_ep not in (None, "tensorrt"):
        raise OptimizeError(
            "FP8 builds a TensorRT engine, so --ort-ep must be tensorrt or omitted."
        )

    profile = resolve_target(request.target)
    provider_text = _provider_text(request, profile, tensorrt=tensorrt)
    version = ""
    if provider_text.strip().lower() in {"tensorrt", "trt"} or (
        provider_text.strip() == TRT_PROVIDER
    ):
        version = installed_trt_version()
    try:
        precision, provider = check_combination(
            profile,
            canonical,
            provider_text,
            tensorrt_version=version,
        )
    except TargetProfileError as exc:
        text = str(exc)
        if canonical == "FP8" and "Hopper" not in text:
            text = f"{text} TensorRT FP8 needs an Ada or Hopper GPU."
        raise OptimizeError(text) from exc

    if (
        precision == "INT8"
        and quant is None
        and not request.tensorrt
        and profile.accelerator == "none"
        and CPU_PROVIDER in profile.execution_providers
    ):
        quant = "ort-int8"

    if quant in {"fp8", "ort-int8"}:
        ok, reason = _quantizer_supports(quant, profile)
        if not ok:
            raise OptimizeError(reason)

    if tensorrt and TRT_PROVIDER not in profile.execution_providers:
        raise OptimizeError(
            f"TensorRT is not valid on {profile.id}. Execution providers: "
            f"{', '.join(profile.execution_providers)}."
        )

    workspace_bytes = _workspace_bytes(request, profile)
    _reject_calibration_on_other_precisions(request, precision)
    _reject_unapplied_precision(request, profile, precision, quant)
    calib_batch = _calib_batch(request, precision)

    export_onnx = _export_onnx(request, quant)
    build_tensorrt = tensorrt
    build_session = _build_session(request, quant, export_onnx, build_tensorrt)

    return OptimizePlan(
        request=request,
        profile=profile,
        precision=precision,
        provider=provider,
        export_onnx=export_onnx,
        build_tensorrt=build_tensorrt,
        build_session=build_session,
        quant=quant,
        workspace_bytes=workspace_bytes,
        calib_batch=calib_batch,
    )


def _resolve_quant(explicit: str | None, implied: str | None, flag: str) -> str | None:
    if implied and explicit and implied != explicit:
        raise OptimizeError(
            f"--precision {flag} selects {implied}, which conflicts with --quant {explicit}."
        )
    if implied:
        return implied
    return explicit


def _quantizer_supports(name: str, profile) -> tuple[bool, str]:
    from ohho.optimize.quantizers import get_quantizer

    return get_quantizer(name).supports(profile, None)


def _export_onnx(request: OptimizeRequest, quant: str | None) -> bool:
    if quant in _WEIGHT and not request.onnx and not request.tensorrt:
        return False
    if quant in {"fp8", "ort-int8"}:
        return bool(request.onnx)
    return True


def _build_session(
    request: OptimizeRequest,
    quant: str | None,
    export_onnx: bool,
    build_tensorrt: bool,
) -> bool:
    if quant == "ort-int8":
        return True
    build = export_onnx and not (build_tensorrt and request.ort_ep is None)
    if request.ort_ep and export_onnx:
        return True
    return build


def _require_text(value: str, flag: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise OptimizeError(f"{flag} is required.")


def _provider_text(
    request: OptimizeRequest, profile: TargetProfile, *, tensorrt: bool = False
) -> str:
    if request.ort_ep:
        return request.ort_ep
    if tensorrt or request.tensorrt:
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
    quant: str | None,
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
        if request.tensorrt or quant == "fp8":
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
        if quant in {"awq", "gptq", "bitsandbytes", "ort-int8"}:
            return
        raise OptimizeError(
            "INT8 needs --tensorrt with --calib-data on a TensorRT target, "
            "or --quant awq, --quant gptq, or the bitsandbytes backend. "
            "On x86-cpu and rpi5, --precision int8 runs ORT static INT8. "
            f"A plain ONNX export for {profile.id} stays FP32."
        )
    if precision == "INT4":
        if quant in {"awq", "gptq", "bitsandbytes"}:
            return
        raise OptimizeError(
            "INT4 needs --quant awq or --quant gptq, or --precision "
            "int4-awq, int4-gptq, or int4-bnb."
        )


def _require_positive_workspace(
    request: OptimizeRequest, profile: TargetProfile
) -> None:
    if request.workspace_gb is None and profile.workspace_size_mib < 1:
        raise OptimizeError(
            f"{profile.id} has no TensorRT workspace. Pass --workspace-gb "
            "within the memory budget, or choose a TensorRT target."
        )
