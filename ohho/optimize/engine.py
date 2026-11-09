"""Run a validated optimize plan and write ``manifest.json``."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ohho.optimize.calibrator import (
    CalibrationCache,
    EmptyBatchSource,
    cuda_copy,
    declared_calib_shape,
    entropy_scale,
    load_calibration_source,
    make_entropy_calibrator,
    preprocessing_config_from_dir,
    read_all_samples,
    sha256_path,
)
from ohho.optimize.errors import OptimizeCapabilityError, OptimizeError
from ohho.optimize.exporters import (
    ONNX_OPSET,
    describe_export,
    export_onnx,
    export_torchscript,
)
from ohho.optimize.manifest import (
    build_manifest,
    file_record,
    package_versions,
    write_manifest,
)
from ohho.optimize.models import load_model
from ohho.optimize.ort_session import build_session
from ohho.optimize.plan import OptimizePlan, OptimizeRequest, build_plan
from ohho.optimize.qdq import MISSING_CALIBRATION, onnx_has_qdq, quantization_mode
from ohho.optimize.quant import probe_quant, quantize
from ohho.optimize.tensorrt import build_engine, cuda_device_available, probe_tensorrt
from ohho.targets.providers import installed_execution_providers

logger = logging.getLogger("ohho.optimize")

_ONNX_EXTRA = (
    "ONNX export needs torch and the onnx package: pip install 'ohho-os[optimize]'."
)


@dataclass
class OptimizeBackends:
    """Stand-ins used by tests. Production calls leave every field empty.

    ``trt_module`` replaces the TensorRT import. ``available_providers``
    replaces the ONNX Runtime provider probe. ``cuda`` overrides the CUDA
    device probe. ``None`` means probe the real machine.
    """

    trt_module: Any = None
    ort_module: Any = None
    available_providers: tuple[str, ...] | None = None
    cuda: bool | None = None


def optimize(
    request: OptimizeRequest, *, backends: OptimizeBackends | None = None
) -> dict[str, Any]:
    """Validate, export, and write ``manifest.json``. Returns the document."""
    plan = build_plan(request)
    standins = backends or OptimizeBackends()
    _assert_ready(plan, standins)
    out = Path(request.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    loaded = load_model(request.model, submodule=request.submodule)
    outputs: list[dict[str, Any]] = []
    quant_path: Path | None = None
    if plan.quant:
        quant_path = quantize(loaded, plan.quant, out, plan.precision)
        outputs.append(file_record(quant_path, plan.quant, out))

    onnx_path = out / "model.onnx"
    wrote_onnx = False
    if plan.export_onnx:
        _assert_export_runtime(loaded)
        if loaded.module is None and loaded.onnx_path is not None:
            if loaded.onnx_path.resolve() != onnx_path.resolve():
                shutil.copyfile(loaded.onnx_path, onnx_path)
        else:
            export_onnx(loaded, onnx_path, opset=ONNX_OPSET, dynamic_batch=True)
        outputs.append(file_record(onnx_path, "onnx", out))
        wrote_onnx = True
        if loaded.kind == "torchscript" and loaded.module is not None:
            ts_path = out / "model.ts"
            export_torchscript(loaded, ts_path)
            outputs.append(file_record(ts_path, "torchscript", out))

    quantization: str | None = None
    if plan.build_tensorrt:
        if not wrote_onnx:
            raise OptimizeError("TensorRT needs an ONNX graph to parse.")
        engine_path = out / "model.engine"
        trt_module = standins.trt_module
        calibrator = None
        activation_scale = None
        if plan.precision == "INT8":
            if trt_module is None:
                trt_module = _load_tensorrt()
            quantization = quantization_mode(trt_module)
            calibrator, activation_scale = _int8_inputs(
                plan, loaded, onnx_path, trt_module, standins, quantization
            )
        build_engine(
            onnx_path,
            engine_path,
            precision=plan.precision,
            workspace_bytes=plan.workspace_bytes,
            max_batch=request.max_batch,
            dynamic_batch=request.dynamic_batch,
            calibrator=calibrator,
            trt_module=trt_module,
            activation_scale=activation_scale,
        )
        outputs.append(file_record(engine_path, "engine", out))

    session_providers: list[str] = []
    if plan.build_session:
        if not wrote_onnx:
            raise OptimizeError("An ONNX Runtime session needs an ONNX file.")
        session = build_session(
            onnx_path,
            provider=plan.provider,
            available=standins.available_providers,
            ort_module=standins.ort_module,
        )
        session_providers = [str(item) for item in session.get_providers()]

    document = build_manifest(
        model={
            "ref": loaded.ref,
            "kind": loaded.kind,
            "submodule": loaded.submodule,
            "source": loaded.source,
        },
        target={
            "id": plan.profile.id,
            "arch": plan.profile.arch,
            "accelerator": plan.profile.accelerator,
        },
        precision=plan.precision,
        execution_provider=plan.provider,
        flags={
            "onnx": wrote_onnx,
            "tensorrt": plan.build_tensorrt,
            "quant": plan.quant,
            "ort_ep": request.ort_ep,
            "workspace_gb": request.workspace_gb,
            "workspace_bytes": plan.workspace_bytes,
            "max_batch": request.max_batch,
            "dynamic_batch": request.dynamic_batch,
            "calib_batch": plan.calib_batch,
            "calib_max_images": request.calib_max_images,
            "quantization": quantization,
            "opset": ONNX_OPSET if wrote_onnx else None,
            "submodule": request.submodule,
        },
        inputs={
            "model": request.model,
            "target": request.target,
            "precision": request.precision,
            "onnx": request.onnx,
            "tensorrt": request.tensorrt,
            "quant": request.quant,
            "ort_ep": request.ort_ep,
            "workspace_gb": request.workspace_gb,
            "max_batch": request.max_batch,
            "dynamic_batch": request.dynamic_batch,
            "calib_data": request.calib_data,
            "calib_cache": request.calib_cache,
            "calib_batch": request.calib_batch,
            "calib_max_images": request.calib_max_images,
            "submodule": request.submodule,
        },
        outputs=outputs,
        session_providers=session_providers,
        versions=package_versions(),
    )
    write_manifest(out, document)
    return document


def _assert_ready(plan: OptimizePlan, standins: OptimizeBackends) -> None:
    if plan.build_session:
        available = standins.available_providers
        if available is None:
            probed = installed_execution_providers()
            available = tuple(probed) if probed is not None else None
        if available is None:
            raise OptimizeCapabilityError(
                "onnxruntime is not installed. CPU and CUDA sessions need "
                "pip install 'ohho-os[optimize]'."
            )
        if plan.provider not in available:
            listed = ", ".join(available) if available else "(none)"
            raise OptimizeCapabilityError(
                f"{plan.provider} is not installed. Installed execution providers: {listed}."
            )
    if plan.build_tensorrt and standins.trt_module is None:
        ok, detail = probe_tensorrt()
        if not ok:
            raise OptimizeCapabilityError(detail)
        cuda = standins.cuda if standins.cuda is not None else cuda_device_available()
        if not cuda:
            raise OptimizeCapabilityError(
                "TensorRT is installed but no CUDA device is available. "
                "Engine builds are a GPU follow-up."
            )
    if plan.quant:
        ok, detail = probe_quant(plan.quant)
        if not ok:
            raise OptimizeCapabilityError(detail)


def _load_tensorrt():
    try:
        import tensorrt as trt
    except Exception as exc:
        raise OptimizeCapabilityError(
            "TensorRT is not installed. FP16, INT8 and FP8 engine builds are a GPU "
            "follow-up: install TensorRT on the target machine and rerun with --tensorrt."
        ) from exc
    return trt


def _assert_export_runtime(loaded) -> None:
    if loaded.module is None and loaded.onnx_path is not None:
        return
    if (
        importlib.util.find_spec("torch") is None
        or importlib.util.find_spec("onnx") is None
    ):
        raise OptimizeCapabilityError(_ONNX_EXTRA)


def _int8_inputs(
    plan: OptimizePlan,
    loaded,
    onnx_path: Path,
    trt_module,
    standins: OptimizeBackends,
    quantization: str,
):
    """Return ``(calibrator, activation_scale)`` for one INT8 engine build."""
    if onnx_has_qdq(onnx_path.read_bytes()):
        logger.info(
            "ONNX model already contains Q/DQ nodes; INT8 calibration data is not required."
        )
        return None, None
    if quantization == "explicit":
        return None, _explicit_activation_scale(plan, loaded, onnx_path)
    return _calibrator(plan, loaded, trt_module, standins, onnx_path), None


def _calibrator(
    plan: OptimizePlan, loaded, trt_module, standins: OptimizeBackends, onnx_path: Path
):
    batch = plan.calib_batch or 1
    shape, preprocessing, cache = _calibration_context(
        plan, loaded, onnx_path, "implicit"
    )
    source = _batch_source(plan, shape, cache, batch)
    copy = None
    cuda = standins.cuda if standins.cuda is not None else cuda_device_available()
    if standins.trt_module is None and cuda:
        copy = cuda_copy
    return make_entropy_calibrator(
        source,
        batch_size=batch,
        cache_path=plan.request.calib_cache,
        trt_module=trt_module,
        copy_to_device=copy,
        model_sha256=None if cache is None else cache.model_sha256,
        preprocessing=None if cache is None else preprocessing,
        quantization="implicit",
    )


def _explicit_activation_scale(plan: OptimizePlan, loaded, onnx_path: Path) -> float:
    """Entropy scale for Q/DQ insertion. A matching sidecar skips the samples."""
    batch = plan.calib_batch or 1
    shape, _preprocessing, cache = _calibration_context(
        plan, loaded, onnx_path, "explicit"
    )
    if cache is not None:
        reused = cache.read()
        if reused is not None:
            scale = cache.activation_scale()
            if scale is None:
                raise OptimizeError(
                    f"INT8 calibration cache {cache.path} has no activation scale. "
                    "Pass --calib-data DIR to calibrate again."
                )
            return scale
        if not plan.request.calib_data:
            detail = cache.reason or "the cache cannot be reused"
            raise OptimizeError(
                f"INT8 calibration cache {cache.path} invalidated: {detail}. "
                "Pass --calib-data DIR to calibrate again."
            )
    source = load_calibration_source(
        plan.request.calib_data,
        batch_size=batch,
        sample_shape=shape,
        max_images=plan.request.calib_max_images,
    )
    scale = entropy_scale(read_all_samples(source))
    if cache is not None:
        cache.write(b"explicit-qdq", activation_scale=scale)
    return scale


def _calibration_context(
    plan: OptimizePlan, loaded, onnx_path: Path, quantization: str
):
    shape = _sample_shape(loaded, plan.request.calib_data)
    preprocessing = preprocessing_config_from_dir(
        plan.request.calib_data,
        shape=shape,
        max_images=plan.request.calib_max_images,
        submodule=plan.request.submodule,
    )
    cache = None
    if plan.request.calib_cache:
        cache = CalibrationCache(
            plan.request.calib_cache,
            model_sha256=_model_sha256(loaded, onnx_path),
            preprocessing=preprocessing,
            quantization=quantization,
        )
    return shape, preprocessing, cache


def _batch_source(
    plan: OptimizePlan, shape, cache: CalibrationCache | None, batch: int
):
    if plan.request.calib_data:
        return load_calibration_source(
            plan.request.calib_data,
            batch_size=batch,
            sample_shape=shape,
            max_images=plan.request.calib_max_images,
        )
    status = "missing" if cache is None else cache.evaluate()
    if status == "reusable":
        return EmptyBatchSource()
    if status == "invalid" and cache is not None:
        raise OptimizeError(
            f"INT8 calibration cache {cache.path} invalidated: {cache.reason}. "
            "Pass --calib-data DIR to calibrate again."
        )
    raise OptimizeError(MISSING_CALIBRATION)


def _sample_shape(loaded, calib_data: str | None) -> tuple[int, ...]:
    if loaded.module is not None:
        spec = describe_export(loaded)
        if len(spec.inputs) != 1:
            raise OptimizeError(
                f"INT8 calibration feeds one input. This model has {len(spec.inputs)}."
            )
        return tuple(int(dim) for dim in spec.inputs[0].shape[1:])
    if loaded.inputs:
        if len(loaded.inputs) != 1:
            raise OptimizeError(
                f"INT8 calibration feeds one input. This model has {len(loaded.inputs)}."
            )
        return tuple(int(dim) for dim in loaded.inputs[0].shape[1:])
    if calib_data:
        return declared_calib_shape(calib_data)
    raise OptimizeError(
        "INT8 calibration needs one input shape. Add model.json inputs or calib.json."
    )


def _model_sha256(loaded, onnx_path: Path) -> str:
    if loaded.path is not None and loaded.path.exists():
        return sha256_path(loaded.path)
    return hashlib.sha256(onnx_path.read_bytes()).hexdigest()
