"""Resolve ``--model`` to a module, an ONNX file, or a registry id.

Built-in ids ``reference-act`` and ``reference-mlp`` are tiny untrained
fixtures for CPU checks. They do not download weights. A path may be a
``.pt`` module, an actor state dict, TorchScript, ONNX, or a directory
with ``model.json`` and a weight file. PyTorch is imported only when a
checkpoint has to be read.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ohho.optimize.errors import OptimizeCapabilityError, OptimizeError

REFERENCE_ACT = {"state_dim": 8, "action_dim": 4, "chunk_size": 3, "hidden": 16}
REFERENCE_MLP = {"obs_dim": 6, "action_dim": 3, "hidden": (16, 8)}
_ID_OK = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_WEIGHT_NAMES = (
    "model.pt",
    "model.pth",
    "weights.pt",
    "policy.pt",
    "model.ts",
    "model.onnx",
)
_SIDECAR_KEYS = frozenset(
    {
        "kind",
        "submodule",
        "inputs",
        "outputs",
        "state_dim",
        "action_dim",
        "obs_dim",
        "chunk_size",
        "hidden",
    }
)
_ACTOR_WEIGHT = re.compile(r"^(\d+)\.weight$")


@dataclass(frozen=True)
class ExportInput:
    """One ONNX input. ``shape`` includes the batch axis."""

    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ModelRecord:
    """A registry id. ``factory`` names a built-in constructor."""

    id: str
    kind: str
    summary: str
    factory: str | None = None
    path: str | None = None
    submodule: str | None = None


@dataclass(frozen=True)
class LoadedModel:
    """A model ready to export, quantize, or copy."""

    ref: str
    kind: str
    source: str
    module: Any
    path: Path | None
    onnx_path: Path | None
    submodule: str | None
    inputs: tuple[ExportInput, ...]
    outputs: tuple[str, ...]


_REGISTRY: dict[str, ModelRecord] = {}


def register_model(record: ModelRecord) -> None:
    """Add or replace a registry id. Ids are lowercase slugs."""
    if not isinstance(record.id, str) or _ID_OK.fullmatch(record.id) is None:
        raise OptimizeError(
            f"Model id {record.id!r} must be a lowercase slug such as reference-act."
        )
    _REGISTRY[record.id] = record


def unregister_model(model_id: str) -> None:
    """Drop a registry id. Built-ins can be registered again."""
    _REGISTRY.pop(model_id, None)


def list_models() -> list[str]:
    """Registry ids in sorted order."""
    return sorted(_REGISTRY)


def get_model(model_id: str) -> ModelRecord:
    """Return one registry record."""
    try:
        return _REGISTRY[model_id]
    except KeyError:
        known = ", ".join(list_models()) or "(none)"
        raise OptimizeError(
            f"Unknown model {model_id!r}. Give a file path or a registry id. "
            f"Registry: {known}."
        ) from None


def load_model(ref: str, *, submodule: str | None = None) -> LoadedModel:
    """Load a path or a registry id."""
    text = ref.strip()
    if not text:
        raise OptimizeError("--model is required.")
    path = Path(text)
    if path.exists():
        return _load_path(path, submodule)
    record = get_model(text)
    chosen = submodule if submodule is not None else record.submodule
    return _load_record(record, chosen)


def build_reference_act():
    """Tiny ACT-style chunk policy. Weights are seeded and the module is eval."""
    torch, nn = _torch()
    torch.manual_seed(0)
    model = _act_type(torch, nn)()
    model.eval()
    return model


def build_reference_mlp():
    """Tiny MLP actor. Weights are seeded and the module is eval."""
    torch, nn = _torch()
    torch.manual_seed(0)
    model = _mlp_type(torch, nn)()
    model.eval()
    return model


def _torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise OptimizeCapabilityError(
            "Loading and exporting a PyTorch model needs the optimize extra: "
            "pip install 'ohho-os[optimize]'."
        ) from exc
    return torch, nn


def _act_type(torch, nn):
    cached = getattr(_act_type, "cls", None)
    if cached is not None:
        return cached

    class ActReference(nn.Module):
        """Inference-only ACT-style policy: state in, action chunk out.

        Encoder, action queries, decoder, and action head follow an ACT
        policy. There is no sampling, so an ONNX Runtime run can match.
        """

        def __init__(self) -> None:
            super().__init__()
            state_dim = int(REFERENCE_ACT["state_dim"])
            hidden = int(REFERENCE_ACT["hidden"])
            chunk = int(REFERENCE_ACT["chunk_size"])
            action_dim = int(REFERENCE_ACT["action_dim"])
            self.encoder = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
            self.action_queries = nn.Parameter(torch.zeros(chunk, hidden))
            self.decoder = nn.Linear(hidden, hidden)
            self.action_head = nn.Linear(hidden, action_dim)

        def forward(self, state):
            encoded = self.encoder(state)
            queries = self.action_queries.unsqueeze(0) + encoded.unsqueeze(1)
            decoded = torch.relu(self.decoder(queries))
            return self.action_head(decoded)

    _act_type.cls = ActReference
    return ActReference


def _mlp_type(torch, nn):
    del torch
    cached = getattr(_mlp_type, "cls", None)
    if cached is not None:
        return cached

    class MlpReference(nn.Module):
        """MLP actor: observation in, action mean out."""

        def __init__(self) -> None:
            super().__init__()
            obs_dim = int(REFERENCE_MLP["obs_dim"])
            action_dim = int(REFERENCE_MLP["action_dim"])
            hidden = tuple(int(item) for item in REFERENCE_MLP["hidden"])
            layers: list[nn.Module] = []
            width = obs_dim
            for size in hidden:
                layers.append(nn.Linear(width, size))
                layers.append(nn.ELU())
                width = size
            layers.append(nn.Linear(width, action_dim))
            self.actor = nn.Sequential(*layers)

        def forward(self, obs):
            return self.actor(obs)

    _mlp_type.cls = MlpReference
    return MlpReference


def _load_record(record: ModelRecord, submodule: str | None) -> LoadedModel:
    if record.factory == "reference-act":
        return LoadedModel(
            ref=record.id,
            kind="act",
            source="registry",
            module=build_reference_act(),
            path=None,
            onnx_path=None,
            submodule=submodule,
            inputs=(ExportInput("state", (1, int(REFERENCE_ACT["state_dim"]))),),
            outputs=("actions",),
        )
    if record.factory == "reference-mlp":
        obs = int(REFERENCE_MLP["obs_dim"])
        return LoadedModel(
            ref=record.id,
            kind="rl_mlp",
            source="registry",
            module=build_reference_mlp(),
            path=None,
            onnx_path=None,
            submodule=submodule,
            inputs=(ExportInput("obs", (1, obs)),),
            outputs=("action_mean",),
        )
    if record.path:
        return _load_path(Path(record.path), submodule)
    raise OptimizeError(f"Model {record.id!r} has no factory and no path.")


def _load_path(path: Path, submodule: str | None) -> LoadedModel:
    if path.is_dir():
        return _load_dir(path, submodule)
    sidecar = _sibling_sidecar(path)
    return _load_file(path, submodule, sidecar)


def _load_dir(path: Path, submodule: str | None) -> LoadedModel:
    sidecar = (
        _read_sidecar(path / "model.json") if (path / "model.json").is_file() else None
    )
    for name in _WEIGHT_NAMES:
        candidate = path / name
        if candidate.is_file():
            return _load_file(candidate, submodule, sidecar, ref=str(path))
    if (path / "config.json").is_file():
        chosen = submodule
        if sidecar and sidecar.get("submodule") and chosen is None:
            chosen = str(sidecar["submodule"])
        kind = str(sidecar.get("kind")) if sidecar and sidecar.get("kind") else "llm"
        return LoadedModel(
            ref=str(path),
            kind=kind,
            source="file",
            module=None,
            path=path,
            onnx_path=None,
            submodule=chosen,
            inputs=_inputs_from_sidecar(sidecar),
            outputs=_outputs_from_sidecar(sidecar, kind),
        )
    raise OptimizeError(
        f"{path} has no model.pt, model.onnx, model.ts, or config.json."
    )


def _load_file(
    path: Path,
    submodule: str | None,
    sidecar: dict[str, Any] | None,
    *,
    ref: str | None = None,
) -> LoadedModel:
    suffix = path.suffix.lower()
    label = ref or str(path)
    chosen = submodule
    if chosen is None and sidecar and sidecar.get("submodule"):
        chosen = str(sidecar["submodule"])
    if suffix == ".onnx":
        kind = "onnx"
        return LoadedModel(
            ref=label,
            kind=kind,
            source="file",
            module=None,
            path=path,
            onnx_path=path,
            submodule=chosen,
            inputs=_inputs_from_sidecar(sidecar),
            outputs=_outputs_from_sidecar(sidecar, kind),
        )
    if suffix in {".ts", ".torchscript"} or (
        sidecar and sidecar.get("kind") == "torchscript"
    ):
        module = _load_torchscript(path)
        return _finish_module(label, path, module, "torchscript", chosen, sidecar)
    module = _load_checkpoint(path, sidecar)
    kind = _kind_of(module, sidecar)
    return _finish_module(label, path, module, kind, chosen, sidecar)


def _finish_module(ref, path, module, kind, submodule, sidecar) -> LoadedModel:
    module.eval()
    inputs = _inputs_from_sidecar(sidecar)
    if not inputs:
        inputs = _infer_inputs(module, kind)
    outputs = _outputs_from_sidecar(sidecar, kind)
    if not outputs:
        outputs = _default_outputs(kind)
    return LoadedModel(
        ref=ref,
        kind=kind,
        source="file",
        module=module,
        path=path,
        onnx_path=None,
        submodule=submodule,
        inputs=inputs,
        outputs=outputs,
    )


def _sibling_sidecar(path: Path) -> dict[str, Any] | None:
    candidate = path.with_name("model.json")
    if candidate.is_file() and candidate != path:
        return _read_sidecar(candidate)
    return None


def _read_sidecar(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OptimizeError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise OptimizeError(f"{path} must be an object.")
    extra = set(data) - _SIDECAR_KEYS
    if extra:
        raise OptimizeError(
            f"{path.name} has unknown fields: {', '.join(sorted(extra))}."
        )
    return data


def _inputs_from_sidecar(sidecar: dict[str, Any] | None) -> tuple[ExportInput, ...]:
    if not sidecar or "inputs" not in sidecar:
        return ()
    raw = sidecar["inputs"]
    if not isinstance(raw, list) or not raw:
        raise OptimizeError("model.json inputs must be a non-empty list.")
    inputs: list[ExportInput] = []
    for item in raw:
        if not isinstance(item, dict):
            raise OptimizeError("model.json inputs must be objects.")
        name = item.get("name")
        shape = item.get("shape")
        if not isinstance(name, str) or not name:
            raise OptimizeError("model.json input name must be a string.")
        if (
            not isinstance(shape, list)
            or not shape
            or not all(type(dim) is int and dim >= 1 for dim in shape)
        ):
            raise OptimizeError(
                f"model.json input {name!r} shape must be positive integers."
            )
        inputs.append(ExportInput(name, tuple(shape)))
    return tuple(inputs)


def _outputs_from_sidecar(sidecar: dict[str, Any] | None, kind: str) -> tuple[str, ...]:
    if not sidecar or "outputs" not in sidecar:
        return ()
    raw = sidecar["outputs"]
    if not isinstance(raw, list) or not all(
        isinstance(item, str) and item for item in raw
    ):
        raise OptimizeError("model.json outputs must be a list of strings.")
    if not raw:
        return _default_outputs(kind)
    return tuple(raw)


def _default_outputs(kind: str) -> tuple[str, ...]:
    if kind == "act":
        return ("actions",)
    if kind == "rl_mlp":
        return ("action_mean",)
    return ("output",)


def _infer_inputs(module, kind: str) -> tuple[ExportInput, ...]:
    _torch_mod, nn = _torch()
    if kind == "act":
        encoder = getattr(module, "encoder", None)
        linear = _first_linear(encoder if encoder is not None else module, nn)
        if linear is not None:
            return (ExportInput("state", (1, int(linear.in_features))),)
    if kind == "rl_mlp":
        actor = getattr(module, "actor", module)
        linear = _first_linear(actor, nn)
        if linear is not None:
            return (ExportInput("obs", (1, int(linear.in_features))),)
    linear = _first_linear(module, nn)
    if linear is not None:
        return (ExportInput("input", (1, int(linear.in_features))),)
    raise OptimizeError(
        "Could not infer an input shape. Add model.json with an inputs list."
    )


def _first_linear(module, nn):
    if module is None:
        return None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    return None


def _kind_of(module, sidecar: dict[str, Any] | None) -> str:
    if sidecar and isinstance(sidecar.get("kind"), str):
        return str(sidecar["kind"])
    if _looks_like_act(module):
        return "act"
    if _looks_like_mlp(module):
        return "rl_mlp"
    return "module"


def _looks_like_act(module) -> bool:
    return hasattr(module, "action_head") and hasattr(module, "encoder")


def _looks_like_mlp(module) -> bool:
    _torch_mod, nn = _torch()
    actor = getattr(module, "actor", None)
    if isinstance(actor, nn.Sequential) and _linear_stack(actor, nn):
        return True
    return isinstance(module, nn.Sequential) and _linear_stack(module, nn)


def _linear_stack(module, nn) -> bool:
    saw_linear = False
    for child in module.children():
        if isinstance(child, nn.Linear):
            saw_linear = True
            continue
        if child.__class__.__name__ in {"ReLU", "ELU", "GELU", "Tanh", "SiLU"}:
            continue
        return False
    return saw_linear


def _load_torchscript(path: Path):
    torch, _nn = _torch()
    try:
        module = torch.jit.load(str(path), map_location="cpu")
    except Exception as exc:
        raise OptimizeError(f"Could not load TorchScript {path}: {exc}") from exc
    return module


def _load_checkpoint(path: Path, sidecar: dict[str, Any] | None):
    torch, _nn = _torch()
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    return _module_from_object(obj, sidecar)


def _module_from_object(obj, sidecar: dict[str, Any] | None):
    torch, nn = _torch()
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict):
        for key in ("module", "model"):
            if isinstance(obj.get(key), nn.Module):
                return obj[key]
        state = obj.get("model_state_dict", obj.get("state_dict", obj))
        if isinstance(state, dict) and _actor_keys(state):
            return mlp_from_actor_state(state)
        if sidecar and sidecar.get("kind") == "rl_mlp" and isinstance(state, dict):
            return mlp_from_actor_state(state)
    raise OptimizeError(
        "Checkpoint is not a module or an actor state dict. "
        "Save an nn.Module, or a state dict with actor.N.weight keys."
    )


def _actor_keys(state: dict) -> bool:
    return any(isinstance(key, str) and key.startswith("actor.") for key in state)


def mlp_from_actor_state(state: dict):
    """Rebuild an MLP actor from ``actor.<index>.weight`` tensors.

    Activation slots (the odd indexes in a Linear/ELU stack) have no
    tensors. ELU is inserted between each Linear except the last one.
    """
    torch, nn = _torch()
    actor = {
        key[len("actor.") :]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("actor.")
    }
    if not actor:
        actor = {
            key: value
            for key, value in state.items()
            if isinstance(key, str) and _ACTOR_WEIGHT.match(key)
        }
    weights = []
    for key, value in actor.items():
        match = _ACTOR_WEIGHT.match(key)
        if match is None:
            continue
        weights.append((int(match.group(1)), value))
    if not weights:
        raise OptimizeError(
            "No actor Linear weights found. Expected keys like actor.0.weight."
        )
    weights.sort(key=lambda item: item[0])
    layers: list[nn.Module] = []
    for index, (_slot, weight) in enumerate(weights):
        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        layers.append(nn.Linear(in_features, out_features))
        if index != len(weights) - 1:
            layers.append(nn.ELU())
    module = nn.Sequential(*layers)
    stripped = {key: value for key, value in actor.items() if key[:1].isdigit()}
    module.load_state_dict(stripped)
    module.eval()
    return module


def _install_builtins() -> None:
    register_model(
        ModelRecord(
            id="reference-act",
            kind="act",
            summary="Tiny ACT-style chunk policy for CPU export checks.",
            factory="reference-act",
        )
    )
    register_model(
        ModelRecord(
            id="reference-mlp",
            kind="rl_mlp",
            summary="Tiny MLP actor for CPU export checks.",
            factory="reference-mlp",
        )
    )


_install_builtins()
