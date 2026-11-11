"""GPTQ INT4 for a local Hugging Face causal-LM directory."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ohho.optimize.quant import _run_gptq
from ohho.optimize.quantizers.backbone import prepare_backbone
from ohho.optimize.quantizers.base import (
    QuantArtifact,
    QuantCalibration,
    Quantizer,
    distribution_version,
)


class GptqQuantizer(Quantizer):
    """GPTQ weight quant. ``int4-gptq`` selects this backend.

    Calibration examples are required. This command does not download a
    dataset.
    """

    name = "gptq"

    def requirements(self) -> tuple[str, ...]:
        return ("auto_gptq", "gptqmodel")

    def supports(self, target, model) -> tuple[bool, str]:
        if (
            target is not None
            and "INT4" not in target.precisions
            and "INT8" not in target.precisions
        ):
            return (
                False,
                f"GPTQ is not valid on {target.id}. "
                f"Supported precisions: {', '.join(target.precisions)}.",
            )
        return True, self.name

    def version(self) -> str:
        return distribution_version("auto-gptq", "gptqmodel", module="auto_gptq")

    def quantize(self, model, calib: QuantCalibration) -> QuantArtifact:
        with tempfile.TemporaryDirectory() as scratch:
            prepared = prepare_backbone(model, Path(scratch))
            written = _run_gptq(
                prepared, Path(calib.out_dir), calib.precision, calib.examples
            )
        return QuantArtifact(written, self.name, self.version(), self.name)


backend = GptqQuantizer()
