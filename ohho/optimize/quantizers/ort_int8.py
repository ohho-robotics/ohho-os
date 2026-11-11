"""ONNX Runtime static INT8 (QDQ) for CPU target profiles."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ohho.optimize.errors import OptimizeCapabilityError, OptimizeError
from ohho.optimize.quantizers.base import (
    QuantArtifact,
    QuantCalibration,
    Quantizer,
    distribution_version,
)


class OrtStaticInt8Quantizer(Quantizer):
    """Static INT8 QDQ quantization on ``x86-cpu`` and ``rpi5-cpu``.

    ``--precision int8`` on a CPU profile selects this backend. A missing
    ``onnxruntime`` install fails before the graph is rewritten.
    """

    name = "ort-int8"

    def requirements(self) -> tuple[str, ...]:
        return ("onnxruntime",)

    def supports(self, target, model) -> tuple[bool, str]:
        del model
        if target is None:
            return False, "ORT static INT8 needs a target profile."
        if "INT8" not in target.precisions:
            return (
                False,
                f"INT8 is not valid on {target.id}. "
                f"Supported precisions: {', '.join(target.precisions)}.",
            )
        cpu = "CPUExecutionProvider" in target.execution_providers
        if target.accelerator != "none" or not cpu:
            return (
                False,
                f"ORT static INT8 runs on CPU profiles. {target.id} is not one.",
            )
        return True, self.name

    def version(self) -> str:
        return distribution_version("onnxruntime", module="onnxruntime")

    def quantize(self, model, calib: QuantCalibration) -> QuantArtifact:
        try:
            from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
        except Exception as exc:
            raise OptimizeCapabilityError(
                "onnxruntime is not installed. ORT static INT8 needs "
                "pip install 'ohho-os[optimize]'."
            ) from exc

        out = Path(calib.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        destination = out / "model.int8.onnx"
        with tempfile.TemporaryDirectory() as scratch:
            onnx_path = _write_fp32(model, Path(scratch))
            specs = _specs(model, onnx_path)
            reader = _reader(model, calib, specs)
            try:
                quantize_static(
                    str(onnx_path),
                    str(destination),
                    reader,
                    quant_format=QuantFormat.QDQ,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8,
                )
            except OptimizeError:
                raise
            except Exception as exc:
                raise OptimizeError(f"ORT static INT8 failed: {exc}") from exc
        if not destination.is_file() or destination.stat().st_size < 1:
            raise OptimizeError("ORT static INT8 did not write a model.")
        return QuantArtifact(destination, self.name, self.version(), self.name)


def _write_fp32(model, scratch: Path) -> Path:
    from ohho.optimize.exporters import export_onnx

    path = scratch / "model.onnx"
    if model.module is None and model.onnx_path is not None:
        import shutil

        shutil.copyfile(model.onnx_path, path)
        return path
    if model.module is None:
        raise OptimizeError("ORT static INT8 needs a PyTorch module or an ONNX file.")
    export_onnx(model, path, dynamic_batch=True)
    return path


def _specs(model, onnx_path: Path):
    if model.inputs:
        return model.inputs
    import onnx

    from ohho.optimize.models import ExportInput

    proto = onnx.load(str(onnx_path))
    specs = []
    for item in proto.graph.input:
        dims = []
        tensor = item.type.tensor_type
        if tensor is None or not tensor.HasField("shape"):
            continue
        for dim in tensor.shape.dim:
            dims.append(int(dim.dim_value) if dim.dim_value else 1)
        if dims:
            specs.append(ExportInput(item.name, tuple(dims)))
    if not specs:
        raise OptimizeError("ORT static INT8 needs at least one model input.")
    return tuple(specs)


def _reader(model, calib: QuantCalibration, specs):
    del model
    samples = _samples(calib, specs)
    if not samples:
        raise OptimizeError("ORT static INT8 needs at least one calibration sample.")

    class _Reader:
        def __init__(self) -> None:
            self._index = 0

        def get_next(self):
            if self._index >= len(samples):
                return None
            item = samples[self._index]
            self._index += 1
            return item

        def rewind(self) -> None:
            self._index = 0

    return _Reader()


def _samples(calib: QuantCalibration, specs) -> list[dict]:
    import numpy as np

    if calib.data:
        return _from_directory(calib, specs, np)
    feeds = []
    for index in range(8):
        feed = {}
        for spec in specs:
            shape = tuple(1 if int(dim) < 1 else int(dim) for dim in spec.shape)
            feed[spec.name] = np.full(shape, 0.01 * (index + 1), dtype=np.float32)
        feeds.append(feed)
    return feeds


def _from_directory(calib: QuantCalibration, specs, np) -> list[dict]:
    from ohho.optimize.calibrator import load_calibration_source

    if len(specs) != 1:
        raise OptimizeError(
            "ORT static INT8 with --calib-data feeds one input. "
            f"This model has {len(specs)}."
        )
    spec = specs[0]
    sample_shape = tuple(int(dim) for dim in spec.shape[1:]) or (1,)
    batch = calib.batch or 1
    source = load_calibration_source(
        calib.data, batch_size=batch, sample_shape=sample_shape
    )
    width = 1
    for dim in sample_shape:
        width *= dim
    feeds = []
    for index in range(source.nbatches):
        flat = source.batch(index)
        values = np.array(flat, dtype=np.float32)
        rows = values.reshape((batch, width))
        for row in rows:
            feeds.append({spec.name: row.reshape((1, *sample_shape))})
    return feeds


backend = OrtStaticInt8Quantizer()
