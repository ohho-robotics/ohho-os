"""Pluggable ONNX and TorchScript exporters.

The generic exporter calls ``torch.onnx.export`` at opset 17 with a dynamic
batch axis. ACT-style and MLP policies match first and use the same export
with their own input and output names. ``register_exporter`` inserts a
matcher ahead of the generic exporter. Submodule selection accepts a dotted
path or the roles ``vision``, ``actor``, ``encoder``, and ``lm``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from ohho.optimize.errors import OptimizeError
from ohho.optimize.models import ExportInput, LoadedModel

ONNX_OPSET = 17

_ROLES = {
    "vision": (
        "model.vision_model",
        "model.vision_tower",
        "model.vision_encoder",
        "vision_encoder",
        "vision_model",
        "vision_tower",
        "backbone",
        "encoder",
    ),
    "actor": ("actor", "model.actor"),
    "encoder": ("encoder", "model.encoder"),
    "lm": (
        "model.vlm_with_expert.vlm.model.text_model",
        "model.vlm_with_expert.vlm.text_model",
        "model.vlm_with_expert.vlm.language_model",
        "vlm_with_expert.vlm.model.text_model",
        "model.text_model",
        "text_model",
        "language_model",
        "model.language_model",
        "lm",
    ),
}


@dataclass(frozen=True)
class ExportSpec:
    """Module and names actually passed to an exporter."""

    module: object
    inputs: tuple[ExportInput, ...]
    outputs: tuple[str, ...]
    kind: str


class _OnnxExporter:
    name = "torch.onnx"

    def matches(self, spec: ExportSpec) -> bool:
        return True

    def export_onnx(
        self, spec: ExportSpec, path: Path, *, opset: int, dynamic_batch: bool
    ) -> None:
        _export_onnx_module(spec, path, opset=opset, dynamic_batch=dynamic_batch)


class _ActExporter:
    name = "act"

    def matches(self, spec: ExportSpec) -> bool:
        return spec.kind == "act" and hasattr(spec.module, "action_head")

    def export_onnx(
        self, spec: ExportSpec, path: Path, *, opset: int, dynamic_batch: bool
    ) -> None:
        _export_onnx_module(spec, path, opset=opset, dynamic_batch=dynamic_batch)


class _MlpExporter:
    name = "rl_mlp"

    def matches(self, spec: ExportSpec) -> bool:
        return spec.kind == "rl_mlp"

    def export_onnx(
        self, spec: ExportSpec, path: Path, *, opset: int, dynamic_batch: bool
    ) -> None:
        _export_onnx_module(spec, path, opset=opset, dynamic_batch=dynamic_batch)


_EXPORTERS: list[object] = [_ActExporter(), _MlpExporter(), _OnnxExporter()]


def register_exporter(exporter: object) -> None:
    """Insert ``exporter`` ahead of the generic ``torch.onnx`` exporter."""
    name = getattr(exporter, "name", None)
    if not isinstance(name, str) or not name:
        raise OptimizeError("An exporter needs a name.")
    for index, existing in enumerate(_EXPORTERS):
        if getattr(existing, "name", None) == "torch.onnx":
            _EXPORTERS.insert(index, exporter)
            return
    _EXPORTERS.append(exporter)


def unregister_exporter(name: str) -> None:
    """Remove an exporter by name. The generic exporter stays if asked."""
    if name == "torch.onnx":
        raise OptimizeError("The generic torch.onnx exporter cannot be removed.")
    _EXPORTERS[:] = [item for item in _EXPORTERS if getattr(item, "name", None) != name]


def exporter_names() -> tuple[str, ...]:
    """Names in match order."""
    return tuple(str(getattr(item, "name", "")) for item in _EXPORTERS)


def select_submodule(module, selector: str):
    """Return a nested module by dotted path or by the vision/actor/encoder/lm role."""
    import torch.nn as nn

    candidates = _ROLES.get(selector, (selector,))
    for dotted in candidates:
        current = module
        found = True
        for part in dotted.split("."):
            if not hasattr(current, part):
                found = False
                break
            current = getattr(current, part)
        if found and isinstance(current, nn.Module):
            return current
    names = [name for name, _child in module.named_modules() if name][:12]
    shown = ", ".join(names) if names else "(none)"
    tried = ", ".join(candidates)
    raise OptimizeError(
        f"No submodule {selector!r}. Tried {tried}. Named modules include: {shown}."
    )


def describe_export(loaded: LoadedModel) -> ExportSpec:
    """Apply submodule selection and decide names for the exporter."""
    if loaded.module is None:
        raise OptimizeError(f"{loaded.ref} has no PyTorch module to export.")
    module = loaded.module
    kind = loaded.kind
    inputs = loaded.inputs
    outputs = loaded.outputs
    if loaded.submodule:
        module = select_submodule(module, loaded.submodule)
        inputs, outputs = _retarget(module, inputs, loaded.submodule, kind)
        if not _still_act(module) and not _still_mlp(module):
            kind = "module"
    if not inputs:
        raise OptimizeError(f"{loaded.ref} has no input shapes.")
    if not outputs:
        outputs = ("output",)
    return ExportSpec(module=module, inputs=inputs, outputs=outputs, kind=kind)


def export_onnx(
    loaded: LoadedModel,
    path: str | Path,
    *,
    opset: int = ONNX_OPSET,
    dynamic_batch: bool = True,
) -> str:
    """Export with the first matching exporter. Returns that exporter's name."""
    spec = describe_export(loaded)
    destination = Path(path)
    for exporter in _EXPORTERS:
        if exporter.matches(spec):
            exporter.export_onnx(
                spec, destination, opset=opset, dynamic_batch=dynamic_batch
            )
            return str(exporter.name)
    raise OptimizeError(f"No ONNX exporter matched {loaded.ref}.")


def export_torchscript(loaded: LoadedModel, path: str | Path) -> Path:
    """Trace the module, then script it if tracing fails. Return the ``.ts`` path."""
    import torch

    spec = describe_export(loaded)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    module = spec.module
    if isinstance(module, torch.jit.ScriptModule):
        module.save(str(destination))
        return destination
    wrapped = _wrap(module)
    wrapped.eval()
    sample = _sample(spec)
    args = sample[0] if len(sample) == 1 else tuple(sample)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        try:
            traced = torch.jit.trace(wrapped, args)
        except Exception as trace_error:
            try:
                traced = torch.jit.script(wrapped)
            except Exception as script_error:
                raise OptimizeError(
                    f"TorchScript export failed: {trace_error}; script failed: {script_error}"
                ) from script_error
    traced.save(str(destination))
    return destination


def _export_onnx_module(
    spec: ExportSpec, path: Path, *, opset: int, dynamic_batch: bool
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = _wrap(spec.module)
    wrapped.eval()
    sample = _sample(spec)
    args = sample[0] if len(sample) == 1 else tuple(sample)
    names_in = [item.name for item in spec.inputs]
    names_out = list(spec.outputs)
    kwargs = {
        "input_names": names_in,
        "output_names": names_out,
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamo": False,
    }
    if dynamic_batch:
        kwargs["dynamic_axes"] = {
            name: {0: "batch"} for name in (*names_in, *names_out)
        }
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            torch.onnx.export(wrapped, args, str(path), **kwargs)
        except TypeError:
            kwargs.pop("dynamo", None)
            torch.onnx.export(wrapped, args, str(path), **kwargs)
    if not path.is_file():
        raise OptimizeError(f"ONNX export did not write {path}.")


def _sample(spec: ExportSpec):
    import torch

    tensors = []
    for item in spec.inputs:
        tensors.append(torch.zeros(*item.shape, dtype=torch.float32))
    return tensors


def _wrap(module):
    import torch.nn as nn

    cls = getattr(_wrap, "cls", None)
    if cls is None:

        class InferenceModule(nn.Module):
            """Forward that returns the first tensor when a policy returns a tuple."""

            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, *args):
                out = self.inner(*args)
                if isinstance(out, (tuple, list)):
                    if not out:
                        raise OptimizeError("The module returned an empty tuple.")
                    return out[0]
                return out

        cls = InferenceModule
        _wrap.cls = cls
    return cls(module)


def _retarget(module, inputs, selector: str, kind: str):
    import torch.nn as nn

    linear = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            linear = child
            break
    if (
        inputs
        and linear is not None
        and int(inputs[0].shape[-1]) == int(linear.in_features)
    ):
        new_inputs = inputs
    elif linear is not None:
        name = inputs[0].name if inputs else "input"
        new_inputs = (ExportInput(name, (1, int(linear.in_features))),)
    else:
        new_inputs = inputs
    if selector in {"encoder", "vision"} or selector.endswith(".encoder"):
        output = "features"
    elif kind == "rl_mlp" or selector in {"actor", "model.actor"}:
        output = "action_mean"
    else:
        output = "output"
    return new_inputs, (output,)


def _still_act(module) -> bool:
    return hasattr(module, "action_head") and hasattr(module, "encoder")


def _still_mlp(module) -> bool:
    import torch.nn as nn

    if isinstance(module, nn.Sequential):
        return any(isinstance(child, nn.Linear) for child in module.modules())
    actor = getattr(module, "actor", None)
    return isinstance(actor, nn.Sequential)
