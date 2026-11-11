"""AWQ INT4 for a local Hugging Face causal-LM directory."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ohho.optimize.quant import _run_awq
from ohho.optimize.quantizers.backbone import prepare_backbone
from ohho.optimize.quantizers.base import (
    QuantArtifact,
    QuantCalibration,
    Quantizer,
    distribution_version,
)


class AwqQuantizer(Quantizer):
    """AWQ weight quant. ``int4-awq`` selects this backend."""

    name = "awq"

    def requirements(self) -> tuple[str, ...]:
        return ("awq", "autoawq")

    def supports(self, target, model) -> tuple[bool, str]:
        if (
            target is not None
            and "INT4" not in target.precisions
            and "INT8" not in target.precisions
        ):
            return (
                False,
                f"AWQ is not valid on {target.id}. "
                f"Supported precisions: {', '.join(target.precisions)}.",
            )
        if model is not None and model.module is None and not _has_config(model):
            return (
                False,
                "AWQ needs a local model directory that contains config.json, "
                "or a SmolVLA module with an lm backbone.",
            )
        return True, self.name

    def version(self) -> str:
        return distribution_version("awq", "autoawq", module="awq")

    def quantize(self, model, calib: QuantCalibration) -> QuantArtifact:
        with tempfile.TemporaryDirectory() as scratch:
            prepared = prepare_backbone(model, Path(scratch))
            written = _run_awq(prepared, Path(calib.out_dir), calib.precision)
        return QuantArtifact(written, self.name, self.version(), self.name)


def _has_config(model) -> bool:
    path = model.path
    if path is None:
        return False
    directory = path if path.is_dir() else path.parent
    return (directory / "config.json").is_file()


backend = AwqQuantizer()
