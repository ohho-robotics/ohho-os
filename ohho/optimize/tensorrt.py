"""Build a TensorRT engine from an ONNX file.

FP16 and INT8 are builder flags. INT8 always assigns the entropy calibrator
to ``config.int8_calibrator`` before ``build_serialized_network``. The
optimisation profile covers every network input. The workspace is the
TensorRT workspace memory pool.

A missing TensorRT install or a missing CUDA device raises
:class:`OptimizeCapabilityError`. Callers can pass ``trt_module`` so a unit
test can assert the calibrator was attached without a GPU.
"""

from __future__ import annotations

from pathlib import Path

from ohho.optimize.errors import OptimizeCapabilityError, OptimizeError

_GPU_INSTALL = (
    "TensorRT is not installed. FP16, INT8 and FP8 engine builds are a GPU "
    "follow-up: install TensorRT on the target machine and rerun with --tensorrt."
)


def probe_tensorrt() -> tuple[bool, str]:
    """Return whether TensorRT imports, and a version or the reason it does not."""
    try:
        import tensorrt as trt
    except Exception:
        return False, _GPU_INSTALL
    version = str(getattr(trt, "__version__", "")).strip() or "installed"
    return True, version


def cuda_device_available() -> bool:
    """Return whether PyTorch can see a CUDA device. Import errors are false."""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    precision: str,
    workspace_bytes: int,
    max_batch: int,
    dynamic_batch: bool,
    calibrator=None,
    trt_module=None,
) -> Path:
    """Serialize an engine. INT8 requires ``calibrator`` and attaches it."""
    trt = trt_module if trt_module is not None else _import_tensorrt()
    if workspace_bytes < 1:
        raise OptimizeError("TensorRT workspace must be at least 1 byte.")
    if max_batch < 1:
        raise OptimizeError("--max-batch must be at least 1.")

    source = Path(onnx_path)
    if not source.is_file():
        raise OptimizeError(f"ONNX file does not exist: {source}")
    logger_type = trt.Logger
    log = logger_type(logger_type.WARNING)
    builder = trt.Builder(log)
    explicit = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit)
    parser = trt.OnnxParser(network, log)
    if not parser.parse(source.read_bytes()):
        errors = [
            str(parser.get_error(index)) for index in range(int(parser.num_errors))
        ]
        detail = "; ".join(errors) if errors else "the parser returned false"
        raise OptimizeError(f"ONNX parse failed: {detail}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))
    _apply_precision(builder, config, trt, precision, calibrator)
    _add_profiles(
        builder,
        config,
        network,
        max_batch=max_batch,
        dynamic_batch=dynamic_batch,
    )
    # Attach again immediately before the build so a precision helper cannot
    # leave INT8 without a calibrator. OHH-467 extends the batch source, not
    # this assignment.
    if precision.upper() == "INT8":
        config.int8_calibrator = calibrator

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise OptimizeError(
            "TensorRT engine build failed. The ONNX graph or the GPU rejected the profile."
        )
    destination = Path(engine_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_as_bytes(serialized))
    return destination


def _import_tensorrt():
    try:
        import tensorrt as trt
    except Exception as exc:
        raise OptimizeCapabilityError(_GPU_INSTALL) from exc
    return trt


def _apply_precision(builder, config, trt, precision: str, calibrator) -> None:
    name = precision.upper()
    if name == "FP32":
        return
    if name == "FP16":
        if not builder.platform_has_fast_fp16:
            raise OptimizeCapabilityError(
                "This GPU does not support fast FP16. Refusing to build an FP16 engine."
            )
        config.set_flag(trt.BuilderFlag.FP16)
        return
    if name == "INT8":
        if calibrator is None:
            raise OptimizeError(
                "INT8 TensorRT requires an entropy calibrator. "
                "Pass --calib-data or an existing --calib-cache."
            )
        if not builder.platform_has_fast_int8:
            raise OptimizeCapabilityError(
                "This GPU does not support INT8. Refusing to build an INT8 engine."
            )
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
        return
    if name == "FP8":
        flag = getattr(trt.BuilderFlag, "FP8", None)
        if flag is None:
            raise OptimizeCapabilityError(
                "This TensorRT build has no FP8 builder flag. "
                "FP8 needs an Ada or Hopper GPU."
            )
        has_fp8 = getattr(builder, "platform_has_fp8", None)
        if has_fp8 is False:
            raise OptimizeCapabilityError(
                "This GPU does not support FP8. FP8 needs an Ada or Hopper GPU."
            )
        config.set_flag(flag)
        return
    raise OptimizeError(f"TensorRT cannot build precision {precision}.")


def _add_profiles(
    builder, config, network, *, max_batch: int, dynamic_batch: bool
) -> None:
    count = int(network.num_inputs)
    if count < 1:
        raise OptimizeError("The ONNX network has no inputs.")
    profile = builder.create_optimization_profile()
    for index in range(count):
        tensor = network.get_input(index)
        shape = tuple(tensor.shape)
        if len(shape) < 1:
            raise OptimizeError(f"Input {tensor.name!r} has an empty shape.")
        rest = tuple(int(dim) for dim in shape[1:])
        if any(dim < 0 for dim in rest):
            raise OptimizeError(
                f"Input {tensor.name!r} has a dynamic axis other than batch. "
                "This builder only profiles the batch axis."
            )
        lo, opt, hi = _batch_shapes(
            rest, max_batch=max_batch, dynamic_batch=dynamic_batch
        )
        profile.set_shape(tensor.name, lo, opt, hi)
    config.add_optimization_profile(profile)


def _batch_shapes(
    rest: tuple[int, ...], *, max_batch: int, dynamic_batch: bool
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if dynamic_batch:
        low = 1
        opt = 1
    else:
        low = max_batch
        opt = max_batch
    high = max_batch
    if low > high:
        raise OptimizeError("--max-batch must be at least 1.")
    return (low, *rest), (opt, *rest), (high, *rest)


def _as_bytes(serialized: object) -> bytes:
    if isinstance(serialized, (bytes, bytearray)):
        return bytes(serialized)
    to_bytes = getattr(serialized, "tobytes", None)
    if callable(to_bytes):
        return to_bytes()
    return bytes(serialized)
