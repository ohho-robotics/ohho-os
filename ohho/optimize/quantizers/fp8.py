"""TensorRT FP8. Hopper and Ada targets only."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ohho.optimize.errors import OptimizeError
from ohho.optimize.quantizers.base import (
    QuantArtifact,
    QuantCalibration,
    Quantizer,
    distribution_version,
)

_FP8_IDS = frozenset({"cloud-l4", "cloud-h100"})


def ada_or_hopper(target) -> bool:
    """True when the profile is a known Ada or Hopper part, or says so."""
    if target is None:
        return False
    if target.id in _FP8_IDS:
        return True
    summary = str(getattr(target, "summary", "")).lower()
    return "ada" in summary or "hopper" in summary


class TensorRtFp8Quantizer(Quantizer):
    """Build a TensorRT engine with the FP8 builder flag.

    ``supports`` rejects any profile that is not Ada or Hopper before an
    export or a builder call. The flag itself is set in
    :func:`ohho.optimize.tensorrt.build_engine`.
    """

    name = "fp8"

    def requirements(self) -> tuple[str, ...]:
        return ("tensorrt",)

    def supports(self, target, model) -> tuple[bool, str]:
        del model
        if target is None:
            return False, "FP8 needs a target profile."
        if "FP8" not in target.precisions or not ada_or_hopper(target):
            listed = ", ".join(target.precisions)
            return (
                False,
                f"FP8 is not valid on {target.id}. "
                "TensorRT FP8 needs an Ada or Hopper GPU. "
                f"Supported precisions: {listed}.",
            )
        return True, "tensorrt fp8"

    def version(self) -> str:
        return distribution_version("tensorrt", module="tensorrt")

    def quantize(self, model, calib: QuantCalibration) -> QuantArtifact:
        from ohho.optimize.exporters import export_onnx
        from ohho.optimize.tensorrt import build_engine

        out = Path(calib.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        engine_path = out / "model.engine"
        with tempfile.TemporaryDirectory() as scratch:
            onnx_path = Path(scratch) / "model.onnx"
            if model.module is None and model.onnx_path is not None:
                shutil.copyfile(model.onnx_path, onnx_path)
            elif model.module is not None:
                export_onnx(model, onnx_path, dynamic_batch=True)
            else:
                raise OptimizeError(
                    "TensorRT FP8 needs a PyTorch module or an ONNX file."
                )
            build_engine(
                onnx_path,
                engine_path,
                precision="FP8",
                workspace_bytes=calib.workspace_bytes,
                max_batch=calib.max_batch,
                dynamic_batch=calib.dynamic_batch,
                trt_module=calib.trt_module,
            )
        return QuantArtifact(engine_path, self.name, self.version(), self.name)


backend = TensorRtFp8Quantizer()
