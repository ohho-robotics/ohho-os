"""Export, TensorRT, ONNX Runtime, and weight-quant entry points.

Import this package by path (``import ohho.optimize``). PyTorch, ONNX
Runtime, and TensorRT are imported only when a run needs them.
"""

from __future__ import annotations

from ohho.optimize.calibrator import load_calibration_source, make_entropy_calibrator
from ohho.optimize.engine import OptimizeBackends, optimize
from ohho.optimize.errors import OptimizeCapabilityError, OptimizeError
from ohho.optimize.exporters import export_onnx, export_torchscript, register_exporter
from ohho.optimize.manifest import SCHEMA_ID, schema_text
from ohho.optimize.models import list_models, load_model, register_model
from ohho.optimize.ort_session import build_session
from ohho.optimize.plan import OptimizeRequest, build_plan
from ohho.optimize.quant import probe_quant, quantize
from ohho.optimize.tensorrt import build_engine

__all__ = [
    "SCHEMA_ID",
    "OptimizeBackends",
    "OptimizeCapabilityError",
    "OptimizeError",
    "OptimizeRequest",
    "build_engine",
    "build_plan",
    "build_session",
    "export_onnx",
    "export_torchscript",
    "list_models",
    "load_calibration_source",
    "load_model",
    "make_entropy_calibrator",
    "optimize",
    "probe_quant",
    "quantize",
    "register_exporter",
    "register_model",
    "schema_text",
]
