"""bitsandbytes NF4. Research baseline, not a production serving path."""

from __future__ import annotations

from pathlib import Path

from ohho.optimize.quant import _run_bitsandbytes
from ohho.optimize.quantizers.base import (
    QuantArtifact,
    QuantCalibration,
    Quantizer,
    distribution_version,
)


class BitsAndBytesQuantizer(Quantizer):
    """NF4 via ``bitsandbytes.nn.Linear4bit``.

    This is a research baseline. ``int4-bnb`` and ``nf4`` select it. It is
    not chosen for a bare ``int4`` or ``int8`` precision.
    """

    name = "bitsandbytes"
    research_only = True

    def requirements(self) -> tuple[str, ...]:
        return ("bitsandbytes",)

    def supports(self, target, model) -> tuple[bool, str]:
        del model
        if (
            target is not None
            and "INT4" not in target.precisions
            and "INT8" not in target.precisions
        ):
            return (
                False,
                f"bitsandbytes is not valid on {target.id}. "
                f"Supported precisions: {', '.join(target.precisions)}.",
            )
        return True, "bitsandbytes NF4 is a research baseline."

    def version(self) -> str:
        return distribution_version("bitsandbytes", module="bitsandbytes")

    def quantize(self, model, calib: QuantCalibration) -> QuantArtifact:
        written = _run_bitsandbytes(model, Path(calib.out_dir), calib.precision)
        return QuantArtifact(written, self.name, self.version(), self.name)


backend = BitsAndBytesQuantizer()
