"""Publish an optimize manifest.

The parity gate is a single call. OHH-430 implements
``ohho.parity.run_parity``. This module does not score outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_parity(document: dict[str, Any], out_dir: str | Path) -> Any:
    """Hand every published output to the parity gate.

    ``document["outputs"]`` includes each quantizer artifact (AWQ, GPTQ,
    bitsandbytes, TensorRT FP8, ORT static INT8) and any ONNX or engine
    file from the same run. When ``ohho.parity`` is not installed this
    returns ``None``.
    """
    try:
        from ohho.parity import run_parity as gate
    except ImportError:
        return None
    return gate(document, out_dir)
