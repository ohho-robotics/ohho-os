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
    "model_sha256",
    "calibration_cache_sha256",
    "host",
    "build_duration_s",
    "cache",
    "parity",
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
    "calib_max_images",
    "quantization",
    "opset",
    "submodule",
    "allow_parity_fail",
)
_MODEL_KEYS = ("ref", "kind", "submodule", "source")
_TARGET_KEYS = ("id", "arch", "accelerator")
_VERSION_KEYS = (
    "torch",
    "onnx",
    "onnxruntime",
    "tensorrt",
    "cuda",
    "driver",
    "jetpack",
)
_HOST_KEYS = ("gpu_name", "gpu_architecture")
_CACHE_KEYS = ("key", "hit", "source")
_CACHE_SOURCES = frozenset({"miss", "local", "registry"})
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
    "calib_max_images",
    "submodule",
    "allow_parity_fail",
)
_OUTPUT_KEYS = ("role", "path", "sha256", "bytes", "backend", "backend_version")
_ROLES = frozenset(
    {
        "onnx",
        "torchscript",
        "engine",
        "awq",
        "gptq",
        "bitsandbytes",
        "fp8",
        "ort-int8",
    }
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
    """Installed torch, ONNX, ONNX Runtime, and TensorRT versions, or null.

    CUDA, the driver, and JetPack are probed separately and stored on the
    same manifest ``versions`` object.
    """
    found: dict[str, str | None] = {}
    for key, dist in _DISTS.items():
        try:
            found[key] = version(dist)
        except PackageNotFoundError:
            found[key] = None
    return found


def file_record(
    path: Path,
    role: str,
    out_dir: Path,
    *,
    backend: str | None = None,
    backend_version: str | None = None,
) -> dict[str, Any]:
    """sha256 and size of one output, with a path relative to ``out_dir``.

    A directory is hashed as the sorted relative paths and file bytes joined
    together. ``bytes`` is the sum of those file sizes. ``backend`` and
    ``backend_version`` name the tool that wrote the file.
    """
    payload, size = _payload(path)
    try:
        rel = path.resolve().relative_to(out_dir.resolve()).as_posix()
    except ValueError:
        rel = path.name
    default_name, default_version = _backend_identity(role)
    return {
        "role": role,
        "path": rel,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": size,
        "backend": backend or default_name,
        "backend_version": backend_version or default_version,
    }


def _backend_identity(role: str) -> tuple[str, str]:
    if role == "onnx":
        return "onnx", _dist_version("onnx")
    if role == "torchscript":
        return "torch", _dist_version("torch")
    if role == "engine":
        return "tensorrt", _dist_version("tensorrt")
    if role == "ort-int8":
        return "ort-int8", _dist_version("onnxruntime")
    if role == "fp8":
        return "fp8", _dist_version("tensorrt")
    return role, "unknown"


def _dist_version(dist: str) -> str:
    try:
        return version(dist)
    except PackageNotFoundError:
        return "unknown"


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
    parity: dict[str, Any],
    model_sha256: str = "0" * 64,
    calibration_cache_sha256: str | None = None,
    host: dict[str, Any] | None = None,
    build_duration_s: float = 0.0,
    cache: dict[str, Any] | None = None,
    versions: dict[str, str | None] | None = None,
    created: str | None = None,
) -> dict[str, Any]:
    """Assemble a manifest document. ``created`` defaults to the current UTC time."""
    from ohho import __version__

    if versions is None:
        from ohho.optimize.toolchain import detect_toolchain

        versions = detect_toolchain().as_dict()
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
        "versions": versions,
        "inputs": inputs,
        "outputs": outputs,
        "session_providers": session_providers,
        "model_sha256": model_sha256,
        "calibration_cache_sha256": calibration_cache_sha256,
        "host": host if host is not None else {"gpu_name": None, "gpu_architecture": None},
        "build_duration_s": build_duration_s,
        "cache": cache if cache is not None else {"key": "0" * 64, "hit": False, "source": "miss"},
        "parity": parity,
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


def manifest_beside(artifact: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load the ``manifest.json`` next to a build directory or an artifact file."""
    path = Path(artifact)
    if path.is_dir():
        manifest = path / "manifest.json"
    elif path.name == "manifest.json":
        manifest = path
    else:
        manifest = path.parent / "manifest.json"
    if not manifest.is_file():
        raise OptimizeError(
            f"No manifest.json beside {path}. Pass a build directory, an "
            "artifact in that directory, or manifest.json."
        )
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OptimizeError(f"{manifest} is not valid JSON: {exc}") from exc
    problems = manifest_problems(document)
    if problems:
        raise OptimizeError("manifest is invalid: " + "; ".join(problems))
    if not isinstance(document, dict):
        raise OptimizeError(f"{manifest} is not an object.")
    return manifest, document


def explain_artifact(artifact: str | Path) -> dict[str, Any]:
    """Return the manifest stored beside ``artifact``."""
    _path, document = manifest_beside(artifact)
    return document


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
                backend_name = item.get("backend")
                if not isinstance(backend_name, str) or not backend_name:
                    problems.append(f"outputs[{index}].backend must be a string")
                backend_version = item.get("backend_version")
                if not isinstance(backend_version, str) or not backend_version:
                    problems.append(
                        f"outputs[{index}].backend_version must be a string"
                    )
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
    inputs = document.get("inputs")
    if (
        isinstance(inputs, dict)
        and "allow_parity_fail" in inputs
        and type(inputs.get("allow_parity_fail")) is not bool
    ):
        problems.append("inputs.allow_parity_fail must be a boolean")
    parity = document.get("parity")
    if "parity" in document:
        from ohho.parity.report import report_problems

        for problem in report_problems(parity):
            problems.append("parity: " + problem)
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
        if (
            "allow_parity_fail" in flags
            and type(flags.get("allow_parity_fail")) is not bool
        ):
            problems.append("flags.allow_parity_fail must be a boolean")
        if type(flags.get("max_batch")) is not int or flags.get("max_batch") < 1:
            problems.append("flags.max_batch must be an integer >= 1")
        if (
            type(flags.get("workspace_bytes")) is not int
            or flags.get("workspace_bytes") < 0
        ):
            problems.append("flags.workspace_bytes must be a non-negative integer")
    versions = document.get("versions")
    if isinstance(versions, dict):
        for key in _VERSION_KEYS:
            value = versions.get(key)
            if value is not None and not isinstance(value, str):
                problems.append(f"versions.{key} must be a string or null")
    _sha256_field(document.get("model_sha256"), "model_sha256", problems)
    if document.get("calibration_cache_sha256") is not None:
        _sha256_field(
            document.get("calibration_cache_sha256"),
            "calibration_cache_sha256",
            problems,
        )
    _object_keys(document.get("host"), _HOST_KEYS, "host", problems)
    host = document.get("host")
    if isinstance(host, dict):
        for key in _HOST_KEYS:
            value = host.get(key)
            if value is not None and not isinstance(value, str):
                problems.append(f"host.{key} must be a string or null")
    duration = document.get("build_duration_s")
    if isinstance(duration, bool) or type(duration) not in (int, float) or duration < 0:
        problems.append("build_duration_s must be a non-negative number")
    _object_keys(document.get("cache"), _CACHE_KEYS, "cache", problems)
    cache = document.get("cache")
    if isinstance(cache, dict):
        _sha256_field(cache.get("key"), "cache.key", problems)
        hit = cache.get("hit")
        source = cache.get("source")
        if type(hit) is not bool:
            problems.append("cache.hit must be a boolean")
        elif hit and source not in {"local", "registry"}:
            problems.append(
                "cache.source must be local or registry when cache.hit is true"
            )
        elif hit is False and source != "miss":
            problems.append("cache.source must be miss when cache.hit is false")
        elif source not in _CACHE_SOURCES:
            problems.append("cache.source must be miss, local, or registry")
    return problems


def _sha256_field(value: object, label: str, problems: list[str]) -> None:
    if not isinstance(value, str) or len(value) != 64:
        problems.append(f"{label} must be 64 hex characters")
        return
    if any(character not in "0123456789abcdef" for character in value):
        problems.append(f"{label} must be lowercase hex")


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
