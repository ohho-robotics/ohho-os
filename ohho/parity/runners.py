"""Run a reference model and a candidate artifact on one batch.

PyTorch and ONNX Runtime are imported only when a run needs them. Both runs
stay on CPU. A TensorRT engine is not executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ohho.optimize.errors import OptimizeError
from ohho.optimize.exporters import describe_export
from ohho.optimize.models import ExportInput, load_model

from ohho.parity.errors import ParityError

_DIRECTORY_FILES = (
    "model.onnx",
    "model.ts",
    "model.pt",
    "model.pth",
    "policy.pt",
    "weights.pt",
)
_ENGINE_SUFFIXES = {".engine", ".plan"}


@dataclass(frozen=True)
class PreparedModel:
    """What to execute for the reference."""

    ref: str
    kind: str
    module: Any
    onnx_path: Path | None
    inputs: tuple[ExportInput, ...]
    outputs: tuple[str, ...]


def resolve_candidate(path: str) -> tuple[Path, str]:
    """Return the file to execute and a format name: onnx, torchscript, or module."""
    candidate = Path(path)
    if not candidate.exists():
        raise ParityError(f"Candidate does not exist: {candidate}")
    if candidate.is_dir():
        for name in _DIRECTORY_FILES:
            found = candidate / name
            if found.is_file():
                return found, _format_of(found)
        raise ParityError(
            f"{candidate} has no model.onnx, model.ts, or model.pt to compare."
        )
    return candidate, _format_of(candidate)


def prepare_reference(ref: str, submodule: str | None) -> PreparedModel:
    """Load the reference and apply the same submodule selection as export."""
    try:
        loaded = load_model(ref, submodule=submodule)
    except OptimizeError as exc:
        raise ParityError(str(exc)) from exc
    if loaded.module is not None:
        try:
            spec = describe_export(loaded)
        except OptimizeError as exc:
            raise ParityError(str(exc)) from exc
        return PreparedModel(
            ref=loaded.ref,
            kind=spec.kind,
            module=spec.module,
            onnx_path=None,
            inputs=spec.inputs,
            outputs=spec.outputs,
        )
    inputs = loaded.inputs
    if not inputs and loaded.onnx_path is not None:
        inputs = specs_from_onnx(loaded.onnx_path)
    outputs = loaded.outputs or ("output",)
    return PreparedModel(
        ref=loaded.ref,
        kind=loaded.kind,
        module=None,
        onnx_path=loaded.onnx_path,
        inputs=inputs,
        outputs=outputs,
    )


def run_prepared(prepared: PreparedModel, batches: dict[str, Any]) -> dict[str, Any]:
    """Run the reference. Returns output name to nested list."""
    if prepared.module is not None:
        return run_torch_module(
            prepared.module, prepared.inputs, batches, prepared.outputs
        )
    if prepared.onnx_path is not None:
        return run_onnx(prepared.onnx_path, batches)
    raise ParityError(f"{prepared.ref} has no module or ONNX file to run.")


def run_candidate(
    path: Path,
    fmt: str,
    batches: dict[str, Any],
    specs: tuple[ExportInput, ...],
    output_names: tuple[str, ...],
) -> dict[str, Any]:
    """Run the candidate artifact."""
    if fmt == "onnx":
        return run_onnx(path, batches, from_names=[spec.name for spec in specs])
    if fmt == "torchscript":
        return run_torchscript(path, specs, batches, output_names)
    if fmt == "module":
        prepared = prepare_reference(str(path), None)
        feed = remap_batches(
            batches,
            [spec.name for spec in specs],
            [spec.name for spec in prepared.inputs],
        )
        return run_prepared(prepared, feed)
    raise ParityError(f"Cannot execute a {fmt} candidate.")


def remap_batches(
    batches: dict[str, Any], from_names: list[str], to_names: list[str]
) -> dict[str, Any]:
    """Align a batch to the candidate's input names, by name then by position."""
    if all(name in batches for name in to_names):
        return {name: batches[name] for name in to_names}
    if len(from_names) != len(to_names):
        raise ParityError(
            "Candidate inputs "
            f"({', '.join(to_names) or '(none)'}) do not match the reference "
            f"({', '.join(from_names) or '(none)'})."
        )
    return {dst: batches[src] for src, dst in zip(from_names, to_names)}


def run_torch_module(module, specs, batches, output_names) -> dict[str, Any]:
    """Evaluate a PyTorch module on CPU."""
    torch = _torch()
    module.eval()
    args = [_tensor(torch, batches, spec.name) for spec in specs]
    with torch.no_grad():
        raw = module(*args)
    return _pack_torch(raw, output_names)


def run_torchscript(path: Path, specs, batches, output_names) -> dict[str, Any]:
    """Evaluate a TorchScript module on CPU."""
    torch = _torch()
    try:
        module = torch.jit.load(str(path), map_location="cpu")
    except Exception as exc:
        raise ParityError(f"Could not load TorchScript {path}: {exc}") from exc
    module.eval()
    args = [_tensor(torch, batches, spec.name) for spec in specs]
    with torch.no_grad():
        raw = module(*args)
    return _pack_torch(raw, output_names)


def run_onnx(
    path: Path,
    batches: dict[str, Any],
    *,
    from_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate an ONNX file with the CPU execution provider."""
    numpy, ort = _onnx_runtime()
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ParityError(f"Could not open ONNX {path}: {exc}") from exc
    active = list(session.get_providers())
    if "CPUExecutionProvider" not in active:
        raise ParityError(
            "ONNX Runtime did not enable CPUExecutionProvider. "
            f"Active providers: {', '.join(active) or '(none)'}."
        )
    names = [str(item.name) for item in session.get_inputs()]
    if from_names is not None:
        batches = remap_batches(batches, from_names, names)
    feed = {}
    for name in names:
        if name not in batches:
            raise ParityError(f"ONNX input {name!r} has no values.")
        feed[name] = numpy.asarray(batches[name], dtype=numpy.float32)
    try:
        values = session.run(None, feed)
    except Exception as exc:
        raise ParityError(f"ONNX Runtime failed on {path.name}: {exc}") from exc
    outputs: dict[str, Any] = {}
    for meta, value in zip(session.get_outputs(), values):
        outputs[str(meta.name)] = numpy.asarray(value).tolist()
    if not outputs:
        raise ParityError(f"{path} produced no outputs.")
    return outputs


def specs_from_onnx(path: Path) -> tuple[ExportInput, ...]:
    """Input names and concrete shapes from an ONNX session. Dynamic batch is 1."""
    _numpy, ort = _onnx_runtime()
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ParityError(f"Could not open ONNX {path}: {exc}") from exc
    specs: list[ExportInput] = []
    for item in session.get_inputs():
        specs.append(
            ExportInput(str(item.name), _concrete_shape(item.shape, item.name))
        )
    if not specs:
        raise ParityError(f"{path} has no inputs.")
    return tuple(specs)


def _concrete_shape(shape, name: str) -> tuple[int, ...]:
    dims: list[int] = []
    for index, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        elif index == 0:
            dims.append(1)
        else:
            raise ParityError(
                f"ONNX input {name!r} has an unknown dimension {dim!r}. "
                "Add model.json with input shapes."
            )
    if not dims:
        raise ParityError(f"ONNX input {name!r} has an empty shape.")
    return tuple(dims)


def _format_of(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _ENGINE_SUFFIXES:
        raise ParityError(
            "TensorRT engines are not executed by the parity gate (GPU follow-up). "
            "Pass the ONNX file exported beside the engine."
        )
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".ts", ".torchscript"}:
        return "torchscript"
    if suffix in {".pt", ".pth"}:
        return "module"
    raise ParityError(
        f"Cannot compare {path.name}. Expected .onnx, .ts, .pt, or a directory "
        "containing one of those."
    )


def _tensor(torch, batches: dict[str, Any], name: str):
    if name not in batches:
        raise ParityError(f"Input {name!r} has no values.")
    return torch.tensor(batches[name], dtype=torch.float32)


def _pack_torch(raw, output_names: tuple[str, ...]) -> dict[str, Any]:
    tensors = _split_tensors(raw)
    names = list(output_names) or ["output"]
    if len(tensors) > 1 and len(names) == 1:
        # ONNX export keeps the first tensor when a policy returns a tuple.
        tensors = tensors[:1]
    if len(tensors) != len(names):
        raise ParityError(
            f"The module returned {len(tensors)} tensors, expected {len(names)} "
            f"({', '.join(names)})."
        )
    return {name: _tolist(tensor) for name, tensor in zip(names, tensors)}


def _split_tensors(raw) -> list[Any]:
    torch = _torch()
    if isinstance(raw, torch.Tensor):
        return [raw]
    if (
        isinstance(raw, (tuple, list))
        and raw
        and all(isinstance(item, torch.Tensor) for item in raw)
    ):
        return list(raw)
    return [raw]


def _tolist(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        data = value.tolist()
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            return [float(data)]
        return data
    raise ParityError("model output is not a tensor.")


def _torch():
    try:
        import torch
    except Exception as exc:
        raise ParityError(
            "Running a PyTorch reference needs the optimize extra: "
            "pip install 'ohho-os[optimize]'."
        ) from exc
    return torch


def _onnx_runtime():
    try:
        import numpy
        import onnxruntime as ort
    except Exception as exc:
        raise ParityError(
            "Comparing an ONNX candidate needs the optimize extra: "
            "pip install 'ohho-os[optimize]'."
        ) from exc
    return numpy, ort
