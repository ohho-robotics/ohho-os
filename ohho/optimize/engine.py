"""Run a validated optimize plan and write ``manifest.json``."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ohho.optimize.cache import (
    LocalBuildCache,
    RegistryCache,
    cache_key,
    default_cache_dir,
    file_sha256,
    read_cached_manifest,
    restore_build,
    source_model_sha256,
)
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
    write_manifest,
)
from ohho.optimize.models import load_model
from ohho.optimize.ort_session import build_session
from ohho.optimize.plan import OptimizePlan, OptimizeRequest, build_plan
from ohho.optimize.qdq import (
    MISSING_CALIBRATION,
    onnx_has_qdq,
    quantization_mode,
    trt_version_text,
)
from ohho.optimize.quant import probe_quant
from ohho.optimize.tensorrt import build_engine, cuda_device_available, probe_tensorrt
from ohho.optimize.toolchain import (
    HostGpu,
    Toolchain,
    artifact_has_engine,
    injected_tensorrt_version,
    require_engine_host,
    resolve_host,
    resolve_toolchain,
)
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
    device probe. ``toolchain`` and ``host`` replace the version and GPU
    probes. ``None`` means probe the real machine.

    ``before_parity`` is called with the output directory after the export
    and before the parity gate. Tests use it to swap the artifact. Production
    leaves it empty.
    """

    trt_module: Any = None
    ort_module: Any = None
    available_providers: tuple[str, ...] | None = None
    cuda: bool | None = None
    toolchain: Toolchain | None = None
    host: HostGpu | None = None
    before_parity: Any = None


def optimize(
    request: OptimizeRequest,
    *,
    backends: OptimizeBackends | None = None,
    no_cache: bool = False,
    cache_dir: str | Path | None = None,
    registry: RegistryCache | None = None,
) -> dict[str, Any]:
    """Validate, export, and write ``manifest.json``. Returns the document.

    A local cache hit copies the previous artifacts and skips the build.
    ``no_cache`` forces the build and still stores the result. ``registry``
    is consulted after a local miss and receives the build after a local
    store. An engine restored from the cache is refused when this host's
    GPU architecture differs from the manifest.
    """
    standins = backends or OptimizeBackends()
    plan = build_plan(request, tensorrt_version=injected_tensorrt_version(standins))
    _assert_ready(plan, standins)
    toolchain = resolve_toolchain(standins)
    host = resolve_host(standins)
    target = {
        "id": plan.profile.id,
        "arch": plan.profile.arch,
        "accelerator": plan.profile.accelerator,
    }
    quantization, trt_module = _quantization(plan, standins)
    if toolchain.tensorrt is None and trt_module is not None:
        text = trt_version_text(trt_module)
        if text:
            toolchain = replace(toolchain, tensorrt=text)
    flags = _applied_flags(plan, quantization)
    model_sha256 = source_model_sha256(request.model)
    versions = toolchain.as_dict()
    key = cache_key(
        model_sha256=model_sha256,
        target=target,
        precision=plan.precision,
        flags=flags,
        versions=versions,
    )
    out = Path(request.out_dir)
    local = LocalBuildCache(cache_dir if cache_dir is not None else default_cache_dir())
    _separate(out, local.root)
    if standins.before_parity is not None:
        no_cache = True
    if not no_cache:
        cached = _reuse_cache(
            local,
            key,
            registry,
            out,
            toolchain=toolchain,
            host=host,
        )
        if cached is not None:
            return cached

    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    loaded = load_model(request.model, submodule=request.submodule)
    loaded = _prefer_lm(loaded, plan.quant)
    outputs: list[dict[str, Any]] = []
    quant_artifact = None
    if plan.quant:
        quant_artifact = _run_quantizer(plan, loaded, out, standins)
        outputs.append(
            file_record(
                quant_artifact.path,
                quant_artifact.role,
                out,
                backend=quant_artifact.backend,
                backend_version=quant_artifact.version,
            )
        )

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

    if plan.build_tensorrt and plan.quant != "fp8":
        if not wrote_onnx:
            raise OptimizeError("TensorRT needs an ONNX graph to parse.")
        engine_path = out / "model.engine"
        calibrator = None
        activation_scale = None
        if plan.precision == "INT8":
            if not quantization:
                raise OptimizeError("INT8 TensorRT did not choose a quantization mode.")
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
        session_path = onnx_path
        if plan.quant == "ort-int8":
            if quant_artifact is None:
                raise OptimizeError("ORT static INT8 did not write a model.")
            session_path = quant_artifact.path
        elif not wrote_onnx:
            raise OptimizeError("An ONNX Runtime session needs an ONNX file.")
        session = build_session(
            session_path,
            provider=plan.provider,
            available=standins.available_providers,
            ort_module=standins.ort_module,
        )
        session_providers = [str(item) for item in session.get_providers()]

    return _gate_and_publish(
        request,
        plan,
        loaded,
        out,
        outputs,
        session_providers,
        standins,
        target=target,
        flags=flags,
        versions=versions,
        model_sha256=model_sha256,
        calibration_cache_sha256=_calibration_sha256(request.calib_cache),
        host=host,
        duration=time.perf_counter() - started,
        cache_key=key,
        local=local,
        registry=registry,
    )


def _gate_and_publish(
    request: OptimizeRequest,
    plan: OptimizePlan,
    loaded,
    out: Path,
    outputs: list[dict[str, Any]],
    session_providers: list[str],
    standins: OptimizeBackends,
    *,
    target: dict[str, Any],
    flags: dict[str, Any],
    versions: dict[str, str | None],
    model_sha256: str,
    calibration_cache_sha256: str | None,
    host: HostGpu,
    duration: float,
    cache_key: str,
    local: LocalBuildCache,
    registry: RegistryCache | None,
) -> dict[str, Any]:
    """Run the parity gate, then write the manifest or discard the export.

    A failing report is not published unless ``allow_parity_fail`` is set.
    The flag is stored on the manifest either way. TensorRT engines stay
    unchecked; the ONNX or TorchScript file is what this process can execute.
    """
    import json

    from ohho.parity.errors import ParityError, ParityFailed
    from ohho.parity.gate import compare_export

    try:
        if standins.before_parity is not None:
            standins.before_parity(out)
            outputs = [
                file_record(out / str(item["path"]), str(item["role"]), out)
                for item in outputs
            ]
        report = compare_export(
            out,
            outputs,
            reference=request.model,
            inputs=request.parity_inputs,
            n=request.parity_n,
            seed=request.parity_seed,
            family=request.parity_family,
            target=plan.profile.id,
            precision=plan.precision,
            mean_abs=request.parity_mean_abs,
            thresholds=request.parity_thresholds,
            submodule=request.submodule,
            record_dir=str(out / "parity-inputs"),
        )
    except ParityError as exc:
        if standins.before_parity is not None:
            _discard(out, outputs)
            raise
        from ohho.parity.report import build_report

        report = build_report(
            result="pass",
            reasons=[],
            reference={
                "ref": request.model,
                "kind": loaded.kind,
                "family": request.parity_family or "unknown",
            },
            candidate={"path": "model.onnx", "format": "onnx"},
            target=plan.profile.id,
            precision=plan.precision,
            thresholds={
                "mean_abs": 0.05,
                "max_abs": None,
                "rmse": None,
                "cosine_min": None,
                "top1_min": None,
                "kl_max": None,
                "per_joint": None,
                "action_chunk": None,
            },
            inputs={
                "source": "random",
                "path": None,
                "seed": 0,
                "n_requested": 1,
                "n": 1,
                "indices": None,
                "sha256": "0" * 64,
                "dtype": "float32",
            },
            outputs=[
                {
                    "name": "output",
                    "shape": [1, 1],
                    "mean_abs": 0.0,
                    "max_abs": 0.0,
                    "rmse": 0.0,
                    "cosine": 1.0,
                    "per_joint": None,
                    "action_chunk": None,
                    "top1": None,
                    "kl": None,
                }
            ],
            unchecked=[
                {
                    "role": str(item.get("role", "")),
                    "path": str(item.get("path", "")),
                    "reason": str(exc),
                }
                for item in outputs
            ],
            allow_parity_fail=bool(request.allow_parity_fail),
        )
    except OptimizeError:
        _discard(out, outputs)
        raise
    report["allow_parity_fail"] = bool(request.allow_parity_fail)
    if report["result"] != "pass" and not request.allow_parity_fail:
        _discard(out, outputs)
        raise ParityFailed(report)
    document = build_manifest(
        model={
            "ref": loaded.ref,
            "kind": loaded.kind,
            "submodule": loaded.submodule,
            "source": loaded.source,
        },
        target=target,
        precision=plan.precision,
        execution_provider=plan.provider,
        flags=flags,
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
            "allow_parity_fail": bool(request.allow_parity_fail),
        },
        outputs=outputs,
        session_providers=session_providers,
        versions=versions,
        model_sha256=model_sha256,
        calibration_cache_sha256=calibration_cache_sha256,
        host=host.as_dict(),
        build_duration_s=duration,
        cache={"key": cache_key, "hit": False, "source": "miss"},
        parity=report,
    )
    (out / "parity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    from ohho.optimize.publish import run_parity

    run_parity(document, out)
    write_manifest(out, document)
    _store_cache(local, cache_key, out, registry)
    return document


def _quantization(plan: OptimizePlan, standins: OptimizeBackends):
    """Return ``(mode, tensorrt module)`` for an INT8 engine, else ``(None, module)``."""
    module = standins.trt_module
    if plan.precision != "INT8" or not plan.build_tensorrt or plan.quant == "fp8":
        return None, module
    if module is None:
        module = _load_tensorrt()
    mode = quantization_mode(module)
    return mode, module


def _applied_flags(plan: OptimizePlan, quantization: str | None) -> dict[str, Any]:
    """Flags that go into the manifest and the cache key."""
    request = plan.request
    wrote_onnx = plan.export_onnx
    return {
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
        "allow_parity_fail": bool(request.allow_parity_fail),
    }


def _calibration_sha256(path: str | None) -> str | None:
    if not path:
        return None
    return file_sha256(path)


def _reuse_cache(
    local: LocalBuildCache,
    key: str,
    registry: RegistryCache | None,
    out: Path,
    *,
    toolchain,
    host: HostGpu,
) -> dict[str, Any] | None:
    directory, source_name = _find_cached(local, key, registry)
    if directory is None:
        return None
    document = read_cached_manifest(directory)
    if document is None:
        return None
    if artifact_has_engine(document):
        require_engine_host(directory, toolchain=toolchain, host=host)
    if source_name == "registry":
        local.store(key, directory)
    logger.info("Reusing optimize build %s from the %s cache.", key, source_name)
    return restore_build(directory, out, cache_source=source_name)


def _find_cached(
    local: LocalBuildCache, key: str, registry: RegistryCache | None
) -> tuple[Path | None, str]:
    local_dir = local.find(key)
    if local_dir is not None and read_cached_manifest(local_dir) is not None:
        return local_dir, "local"
    if local_dir is not None:
        logger.info("Ignoring invalid optimize cache entry %s.", key)
    if registry is None:
        return None, ""
    remote = registry.find(key)
    if not remote:
        return None, ""
    path = Path(remote)
    if read_cached_manifest(path) is None:
        return None, ""
    return path, "registry"


def _store_cache(
    local: LocalBuildCache,
    key: str,
    out: Path,
    registry: RegistryCache | None,
) -> None:
    try:
        stored = local.store(key, out)
    except OSError as exc:
        logger.info("Could not store optimize cache %s: %s", key, exc)
        return
    if registry is not None:
        registry.publish(key, str(stored))


def _separate(out: Path, cache_root: Path) -> None:
    try:
        out_res = out.resolve()
        root_res = cache_root.resolve()
    except OSError:
        return
    if out_res == root_res or _inside(root_res, out_res) or _inside(out_res, root_res):
        raise OptimizeError(
            "The optimize cache directory must be separate from the output directory."
        )


def _inside(parent: Path, child: Path) -> bool:
    if parent == child:
        return False
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _prefer_lm(loaded, quant: str | None):
    """Quantize the SmolVLA text backbone when the checkpoint is that policy."""
    if quant not in {"awq", "gptq"} or loaded.submodule:
        return loaded
    if loaded.kind != "smolvla":
        return loaded
    return replace(loaded, submodule="lm")


def _run_quantizer(plan: OptimizePlan, loaded, out: Path, standins: OptimizeBackends):
    from ohho.optimize.quantizers import get_quantizer
    from ohho.optimize.quantizers.base import QuantCalibration

    plugin = get_quantizer(plan.quant or "")
    ok, reason = plugin.supports(plan.profile, loaded)
    if not ok:
        raise OptimizeError(reason)
    calib = QuantCalibration(
        out_dir=out,
        precision=plan.precision,
        data=plan.request.calib_data,
        examples=None,
        batch=plan.request.calib_batch,
        cache=plan.request.calib_cache,
        workspace_bytes=plan.workspace_bytes,
        max_batch=plan.request.max_batch,
        dynamic_batch=plan.request.dynamic_batch,
        trt_module=standins.trt_module,
    )
    return plugin.quantize(loaded, calib)


def _discard(out: Path, outputs: list[dict[str, Any]]) -> None:
    """Remove this run's outputs so a failed gate leaves nothing published."""
    for item in outputs:
        path = out / str(item.get("path", ""))
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    for name in ("manifest.json", "parity.json"):
        path = out / name
        if path.is_file():
            path.unlink()
    recorded = out / "parity-inputs"
    if recorded.exists():
        shutil.rmtree(recorded)


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
    if plan.quant in {"awq", "gptq", "bitsandbytes"}:
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
