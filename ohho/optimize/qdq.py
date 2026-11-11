"""Implicit INT8 calibration versus explicit Q/DQ.

TensorRT 10.0 deprecated implicit INT8 calibration
(``IInt8EntropyCalibrator2`` and ``config.int8_calibrator``). Those versions
build from an ONNX graph that already contains ``QuantizeLinear`` and
``DequantizeLinear`` nodes. TensorRT 8 and 9 still attach the entropy
calibrator. A missing ``__version__`` stays on the implicit path so a test
double can attach the calibrator without claiming to be TensorRT 10.
"""

from __future__ import annotations

import array
import math
import re
import sys
from pathlib import Path

from ohho.optimize.errors import OptimizeError

# First TensorRT release that deprecates implicit INT8 calibration.
EXPLICIT_QDQ_MAJOR = 10

MISSING_CALIBRATION = (
    "INT8 TensorRT needs --calib-data DIR or an existing --calib-cache file, "
    "unless the ONNX model already contains QuantizeLinear and DequantizeLinear "
    "(Q/DQ) nodes. This model has no calibration data and no Q/DQ nodes."
)

_VERSION = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?")
_QDQ_OPS = (b"QuantizeLinear", b"DequantizeLinear")
_COMPUTE_OPS = frozenset({"Conv", "ConvTranspose", "MatMul", "Gemm", "Mul", "Add"})


def trt_version_text(trt_module: object | None) -> str:
    """Dotted ``major.minor.patch`` from a TensorRT module, or ``""``."""
    if trt_module is None:
        return ""
    raw = str(getattr(trt_module, "__version__", "")).strip()
    match = _VERSION.match(raw)
    if match is None:
        return ""
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    patch = int(match.group(3) or 0)
    return f"{major}.{minor}.{patch}"


def quantization_mode(trt_module: object | None) -> str:
    """Return ``explicit`` or ``implicit`` for this TensorRT module.

    TensorRT 10 and newer return ``explicit``. Older versions, and a module
    that does not publish ``__version__``, return ``implicit``.
    """
    text = trt_version_text(trt_module)
    if not text:
        return "implicit"
    major = int(text.split(".", 1)[0])
    if major >= EXPLICIT_QDQ_MAJOR:
        return "explicit"
    return "implicit"


def onnx_has_qdq(data: bytes) -> bool:
    """Return whether ``data`` is an ONNX graph with Q/DQ nodes.

    Both op names have to appear. When the ``onnx`` package can parse the
    buffer, the node types decide. A buffer that names both ops but is not a
    readable ONNX document is treated as Q/DQ so a marked file is not forced
    through a second calibration.
    """
    if not data or any(token not in data for token in _QDQ_OPS):
        return False
    try:
        import onnx
    except Exception:
        return True
    try:
        model = onnx.load_from_string(data)
    except Exception:
        return True
    return any(
        node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        for node in model.graph.node
    )


def model_has_qdq(ref: str) -> bool:
    """Return whether ``ref`` points at an ONNX file that already has Q/DQ."""
    path = Path(ref)
    if path.is_file() and path.suffix.lower() == ".onnx":
        return onnx_has_qdq(path.read_bytes())
    if path.is_dir():
        candidate = path / "model.onnx"
        if candidate.is_file():
            return onnx_has_qdq(candidate.read_bytes())
    return False


def insert_qdq(model_bytes: bytes, activation_scale: float) -> bytes:
    """Insert explicit Q/DQ around float compute inputs and return the bytes.

    Weights use a per-tensor min/max scale. Activations share
    ``activation_scale`` from the calibration set. The ``onnx`` package does
    the edit. TensorRT 10 then parses this graph instead of attaching the
    deprecated entropy calibrator.
    """
    if (
        not isinstance(activation_scale, (int, float))
        or isinstance(activation_scale, bool)
        or not math.isfinite(float(activation_scale))
        or float(activation_scale) <= 0.0
    ):
        raise OptimizeError(
            "Explicit Q/DQ needs a positive finite activation scale from --calib-data."
        )
    onnx, helper, tensor_proto = _onnx()
    try:
        model = onnx.load_from_string(model_bytes)
    except Exception as exc:
        raise OptimizeError(
            "Explicit Q/DQ needs a readable ONNX graph. "
            f"The onnx package could not load the model: {exc}"
        ) from exc
    opset = _opset(model)
    if opset < 13:
        raise OptimizeError(
            f"Explicit Q/DQ needs ONNX opset 13 or newer (this graph is {opset}). "
            "QuantizeLinear uses an INT8 zero-point from opset 13."
        )
    scale = float(activation_scale)
    wrapped = _wrap_ops(model.graph, helper, tensor_proto, scale, _COMPUTE_OPS)
    if wrapped < 1:
        wrapped = _wrap_ops(model.graph, helper, tensor_proto, scale, None)
    if wrapped < 1 or not any(
        node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        for node in model.graph.node
    ):
        raise OptimizeError(
            "Explicit Q/DQ could not find a float tensor to quantize. "
            "Pass an ONNX model that already contains QuantizeLinear and "
            "DequantizeLinear nodes."
        )
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise OptimizeError(f"Explicit Q/DQ graph failed ONNX check: {exc}") from exc
    return model.SerializeToString()


def _onnx():
    try:
        import onnx
        from onnx import TensorProto, helper
    except Exception as exc:
        raise OptimizeError(
            "TensorRT 10 and newer deprecate implicit INT8 calibration. "
            "Inserting QuantizeLinear and DequantizeLinear nodes needs the onnx "
            "package: pip install 'ohho-os[optimize]'. Or pass an ONNX model that "
            "already contains those nodes."
        ) from exc
    return onnx, helper, TensorProto


def _opset(model) -> int:
    for item in model.opset_import:
        if item.domain in ("", "ai.onnx"):
            return int(item.version or 0)
    return 0


def _float_types(tensor_proto) -> frozenset[int]:
    names = ("FLOAT", "FLOAT16", "DOUBLE", "BFLOAT16")
    found = []
    for name in names:
        value = getattr(tensor_proto, name, None)
        if isinstance(value, int):
            found.append(value)
    return frozenset(found)


def _wrap_ops(graph, helper, tensor_proto, activation_scale: float, ops) -> int:
    initializers = {item.name: item for item in graph.initializer}
    floats = _float_types(tensor_proto)
    insertions: list[tuple[int, list]] = []
    counter = 0
    wrapped = 0
    for index, node in enumerate(graph.node):
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        if ops is not None and node.op_type not in ops:
            continue
        prefix: list = []
        for slot, name in enumerate(list(node.input)):
            if not name:
                continue
            dtype = _elem_type(graph, initializers, name)
            if dtype is not None and dtype not in floats:
                continue
            if dtype == tensor_proto.FLOAT and name in initializers:
                scale = _scale_from_amax(_float_amax(initializers[name]))
            else:
                scale = activation_scale
            dq_name, nodes, counter = _make_qdq(
                graph, helper, tensor_proto, name, scale, counter
            )
            node.input[slot] = dq_name
            prefix.extend(nodes)
            wrapped += 1
        if prefix:
            insertions.append((index, prefix))
    for index, prefix in reversed(insertions):
        for extra in reversed(prefix):
            graph.node.insert(index, extra)
    return wrapped


def _make_qdq(graph, helper, tensor_proto, name: str, scale: float, counter: int):
    counter += 1
    scale_name = f"ohho_qdq_scale_{counter}"
    zero_name = f"ohho_qdq_zero_{counter}"
    quantized = f"ohho_qdq_q_{counter}"
    dequantized = f"ohho_qdq_dq_{counter}"
    graph.initializer.extend(
        [
            helper.make_tensor(scale_name, tensor_proto.FLOAT, [], [float(scale)]),
            helper.make_tensor(zero_name, tensor_proto.INT8, [], [0]),
        ]
    )
    quantize = helper.make_node(
        "QuantizeLinear",
        [name, scale_name, zero_name],
        [quantized],
        name=f"ohho_quantize_{counter}",
    )
    dequantize = helper.make_node(
        "DequantizeLinear",
        [quantized, scale_name, zero_name],
        [dequantized],
        name=f"ohho_dequantize_{counter}",
    )
    return dequantized, [quantize, dequantize], counter


def _elem_type(graph, initializers: dict, name: str) -> int | None:
    tensor = initializers.get(name)
    if tensor is not None:
        return int(tensor.data_type)
    for container in (graph.input, graph.value_info, graph.output):
        for info in container:
            if info.name != name or not info.type.HasField("tensor_type"):
                continue
            return int(info.type.tensor_type.elem_type)
    return None


def _float_amax(tensor) -> float:
    if tensor.float_data:
        values = [float(item) for item in tensor.float_data]
    elif tensor.raw_data:
        raw = bytes(tensor.raw_data)
        if len(raw) % 4 != 0:
            raise OptimizeError(
                f"Initializer {tensor.name!r} is not packed float32, so explicit "
                "Q/DQ cannot scale it."
            )
        decoded = array.array("f")
        decoded.frombytes(raw)
        if sys.byteorder != "little":
            decoded.byteswap()
        values = [float(item) for item in decoded]
    else:
        values = []
    amax = 0.0
    for value in values:
        if not math.isfinite(value):
            raise OptimizeError(
                f"Initializer {tensor.name!r} must be finite float32 to insert Q/DQ."
            )
        magnitude = abs(value)
        if magnitude > amax:
            amax = magnitude
    return amax


def _scale_from_amax(amax: float) -> float:
    if amax <= 0.0 or not math.isfinite(amax):
        return 1.0
    return amax / 127.0
