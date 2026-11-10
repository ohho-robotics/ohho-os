"""``manifest.json`` for one optimize run, and its published JSON Schema."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path
from typing import Any

from ohho.optimize.errors import OptimizeError

SCHEMA_ID = "ohho.optimize_manifest.v1"
SCHEMA_URL = "https://ohho-robotics.com/schemas/optimize-manifest.v1.json"
_REQUIRED = (
    "schema",
    "ohho_version",
    "created",
    "model",
    "target",
    "precision",
    "execution_provider",
    "flags",
    "versions",
    "inputs",
    "outputs",
    "session_providers",
)
_FLAG_KEYS = (
    "onnx",
    "tensorrt",
    "quant",
    "ort_ep",
    "workspace_gb",
    "workspace_bytes",
    "max_batch",
    "dynamic_batch",
    "calib_batch",
    "opset",
    "submodule",
)
_MODEL_KEYS = ("ref", "kind", "submodule", "source")
_TARGET_KEYS = ("id", "arch", "accelerator")
_VERSION_KEYS = ("torch", "onnx", "onnxruntime", "tensorrt")
_INPUT_KEYS = (
    "model",
    "target",
    "precision",
    "onnx",
    "tensorrt",
    "quant",
    "ort_ep",
    "workspace_gb",
    "max_batch",
    "dynamic_batch",
    "calib_data",
    "calib_cache",
    "calib_batch",
    "submodule",
)
_OUTPUT_KEYS = ("role", "path", "sha256", "bytes")
_ROLES = frozenset(
    {"onnx", "torchscript", "engine", "awq", "gptq", "bitsandbytes", "fp8"}
)
_DISTS = {
    "torch": "torch",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "tensorrt": "tensorrt",
}


def schema_text() -> str:
    """The JSON Schema document shipped with the package."""
    path = files("ohho.optimize").joinpath("optimize_manifest.schema.json")
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def package_versions() -> dict[str, str | None]:
    """Installed torch, ONNX, ONNX Runtime, and TensorRT versions, or null."""
    found: dict[str, str | None] = {}
    for key, dist in _DISTS.items():
        try:
            found[key] = version(dist)
        except PackageNotFoundError:
            found[key] = None
    return found


def file_record(path: Path, role: str, out_dir: Path) -> dict[str, Any]:
    """sha256 and size of one output, with a path relative to ``out_dir``.

    A directory is hashed as the sorted relative paths and file bytes joined
    together. ``bytes`` is the sum of those file sizes.
    """
    payload, size = _payload(path)
    try:
        rel = path.resolve().relative_to(out_dir.resolve()).as_posix()
    except ValueError:
        rel = path.name
    return {
        "role": role,
        "path": rel,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": size,
    }


def _payload(path: Path) -> tuple[bytes, int]:
    if path.is_file():
        data = path.read_bytes()
        return data, len(data)
    if not path.is_dir():
        raise OptimizeError(f"Output does not exist: {path}")
    chunks: list[bytes] = []
    size = 0
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        data = item.read_bytes()
        size += len(data)
        rel = item.relative_to(path).as_posix().encode("utf-8")
        chunks.append(rel + b"\0" + data)
    return b"".join(chunks), size


def build_manifest(
    *,
    model: dict[str, Any],
    target: dict[str, Any],
    precision: str,
    execution_provider: str,
    flags: dict[str, Any],
    inputs: dict[str, Any],
    outputs: list[dict[str, Any]],
    session_providers: list[str],
    versions: dict[str, str | None] | None = None,
    created: str | None = None,
) -> dict[str, Any]:
    """Assemble a manifest document. ``created`` defaults to the current UTC time."""
    from ohho import __version__

    document = {
        "schema": SCHEMA_ID,
        "ohho_version": __version__,
        "created": created
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": model,
        "target": target,
        "precision": precision,
        "execution_provider": execution_provider,
        "flags": flags,
        "versions": versions if versions is not None else package_versions(),
        "inputs": inputs,
        "outputs": outputs,
        "session_providers": session_providers,
    }
    problems = manifest_problems(document)
    if problems:
        raise OptimizeError("manifest is invalid: " + "; ".join(problems))
    return document


def write_manifest(out_dir: str | Path, document: dict[str, Any]) -> Path:
    """Validate and write ``manifest.json``."""
    problems = manifest_problems(document)
    if problems:
        raise OptimizeError("manifest is invalid: " + "; ".join(problems))
    path = Path(out_dir) / "manifest.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def manifest_problems(document: object) -> list[str]:
    """Return schema problems. An empty list means the document matches."""
    if not isinstance(document, dict):
        return ["manifest must be an object"]
    problems: list[str] = []
    missing = [key for key in _REQUIRED if key not in document]
    if missing:
        problems.append("missing " + ", ".join(missing))
    extra = [key for key in document if key not in _REQUIRED]
    if extra:
        problems.append("unknown " + ", ".join(sorted(extra)))
    if document.get("schema") != SCHEMA_ID:
        problems.append(f"schema must be {SCHEMA_ID}")
    if not isinstance(document.get("ohho_version"), str):
        problems.append("ohho_version must be a string")
    if not isinstance(document.get("created"), str) or not document.get("created"):
        problems.append("created must be a string")
    _object_keys(document.get("model"), _MODEL_KEYS, "model", problems)
    _object_keys(document.get("target"), _TARGET_KEYS, "target", problems)
    if not isinstance(document.get("precision"), str):
        problems.append("precision must be a string")
    if not isinstance(document.get("execution_provider"), str):
        problems.append("execution_provider must be a string")
    _object_keys(document.get("flags"), _FLAG_KEYS, "flags", problems)
    _object_keys(document.get("versions"), _VERSION_KEYS, "versions", problems)
    _object_keys(document.get("inputs"), _INPUT_KEYS, "inputs", problems)
    outputs = document.get("outputs")
    if not isinstance(outputs, list):
        problems.append("outputs must be a list")
    else:
        for index, item in enumerate(outputs):
            _object_keys(item, _OUTPUT_KEYS, f"outputs[{index}]", problems)
            if isinstance(item, dict):
                if item.get("role") not in _ROLES:
                    problems.append(f"outputs[{index}].role is not a known role")
                digest = item.get("sha256")
                if not isinstance(digest, str) or len(digest) != 64:
                    problems.append(
                        f"outputs[{index}].sha256 must be 64 hex characters"
                    )
                elif any(ch not in "0123456789abcdef" for ch in digest):
                    problems.append(f"outputs[{index}].sha256 must be lowercase hex")
                if type(item.get("bytes")) is not int or item.get("bytes") < 0:
                    problems.append(
                        f"outputs[{index}].bytes must be a non-negative integer"
                    )
    providers = document.get("session_providers")
    if not isinstance(providers, list) or not all(
        isinstance(item, str) for item in providers
    ):
        problems.append("session_providers must be a list of strings")
    flags = document.get("flags")
    if isinstance(flags, dict):
        if (
            type(flags.get("onnx")) is not bool
            or type(flags.get("tensorrt")) is not bool
        ):
            problems.append("flags.onnx and flags.tensorrt must be booleans")
        if type(flags.get("dynamic_batch")) is not bool:
            problems.append("flags.dynamic_batch must be a boolean")
        if type(flags.get("max_batch")) is not int or flags.get("max_batch") < 1:
            problems.append("flags.max_batch must be an integer >= 1")
        if (
            type(flags.get("workspace_bytes")) is not int
            or flags.get("workspace_bytes") < 0
        ):
            problems.append("flags.workspace_bytes must be a non-negative integer")
    return problems


def _object_keys(
    value: object, keys: tuple[str, ...], label: str, problems: list[str]
) -> None:
    if not isinstance(value, dict):
        problems.append(f"{label} must be an object")
        return
    missing = [key for key in keys if key not in value]
    extra = [key for key in value if key not in keys]
    if missing:
        problems.append(f"{label} missing " + ", ".join(missing))
    if extra:
        problems.append(f"{label} unknown " + ", ".join(sorted(extra)))
