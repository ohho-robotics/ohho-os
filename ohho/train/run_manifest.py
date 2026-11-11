"""``ohho.run/v1`` manifest for a training run.

``ohho train`` writes ``run_manifest.json`` when a family emits ``train_start``
and finalises that same file when the run succeeds, fails, or is interrupted.
The file sits in the run directory with the other artifacts.

Database storage is :meth:`RunStore.save` only. This module does not open a
connection, read credentials, or contact the network. Remote base-weight URIs
are recorded and are not downloaded. ``cloned_from`` is omitted unless
``ohho run clone`` produced the run.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .precision import (
    PRECISIONS,
    backend_for,
    blank_precision_metrics,
    normalize_peak,
    normalize_throughput,
)
from .schema_validate import validate_instance

SCHEMA_ID = "ohho.run/v1"
SCHEMA_FILENAME = "ohho.run.v1.json"
MANIFEST_NAME = "run_manifest.json"
_SCHEMA: dict[str, Any] | None = None
_COMMIT = re.compile(r"[0-9a-f]{40}")
_STORE: Any = None

# Lines ``ohho run show`` prints, and the summary the console run page renders.
CONSOLE_FIELDS = (
    "status",
    "started_at",
    "finished_at",
    "family",
    "device",
    "precision",
    "loss_scaling",
    "precision_backend",
    "seeds.recipe",
    "seeds.python",
    "seeds.numpy",
    "seeds.torch",
    "seeds.pythonhashseed",
    "seeds.extra",
    "git.commit",
    "git.dirty",
    "dataset.version_id",
    "dataset.hash",
    "dataset.path",
    "base_weights.uri",
    "base_weights.revision",
    "base_weights.hash",
    "base_weights.loaded",
    "container_image_digest",
    "library_versions",
    "gpu",
    "final_loss",
    "precision_metrics",
    "qat",
    "error",
    "resolved_config",
)


class RunManifestError(ValueError):
    """A manifest document is not valid ``ohho.run/v1``."""

    def __init__(self, message: str, errors: Sequence[str] | None = None) -> None:
        self.errors = list(errors or [])
        text = message
        if self.errors:
            text = message + ": " + "; ".join(self.errors)
        super().__init__(text)


class RunStore(Protocol):
    """Local sink for a manifest document.

    ``save`` is called with a JSON object after the running file is written
    and again after the final file is written. ohho-os does not open a
    database, read credentials, or make network calls. An operator-supplied
    sink may store the document on its own.
    """

    def save(self, manifest: Mapping[str, Any]) -> None:
        """Persist one manifest document."""
        ...


@dataclass(frozen=True)
class SeedRecord:
    """Seeds that apply to this run.

    ``recipe`` is ``hyperparameters.seed``. ``python``, ``numpy``, and
    ``torch`` are that same integer when the library was seeded, and null
    when it was not. ``extra`` holds other ``*_seed`` hyperparameters.
    """

    recipe: int | None
    python: int | None
    numpy: int | None
    torch: int | None
    pythonhashseed: str | None
    extra: dict[str, int | str | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "python": self.python,
            "numpy": self.numpy,
            "torch": self.torch,
            "pythonhashseed": self.pythonhashseed,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class GitState:
    """Commit and dirty flag for the checkout that contains the recipe."""

    commit: str | None
    dirty: bool
    root: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"commit": self.commit, "dirty": self.dirty, "root": self.root}


@dataclass(frozen=True)
class DatasetRecord:
    """Dataset version id and the hash of a local tree."""

    version_id: str | None
    path: str | None
    hash: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"version_id": self.version_id, "path": self.path, "hash": self.hash}


@dataclass(frozen=True)
class BaseWeightsRecord:
    """Base-checkpoint identity. ``hash`` is set for a local file only."""

    uri: str | None
    revision: str | None
    hash: str | None
    loaded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "revision": self.revision,
            "hash": self.hash,
            "loaded": self.loaded,
        }


@dataclass(frozen=True)
class GpuRecord:
    """CUDA device name, driver version, and CUDA version."""

    name: str | None
    driver: str | None
    cuda: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "driver": self.driver, "cuda": self.cuda}


@dataclass(frozen=True)
class ClonedFrom:
    """Source run a cloned config was derived from."""

    run: str
    manifest: str

    def to_dict(self) -> dict[str, str]:
        return {"run": self.run, "manifest": self.manifest}


@dataclass(frozen=True)
class RunManifest:
    """One ``ohho.run/v1`` document.

    ``cloned_from`` is set only for a run written by ``ohho run clone``.
    Training manifests omit it.
    """

    schema: str
    status: str
    started_at: str
    finished_at: str | None
    family: str
    device: str
    precision: str
    loss_scaling: bool
    precision_backend: str
    resolved_config: dict[str, Any]
    seeds: SeedRecord
    git: GitState
    dataset: DatasetRecord
    base_weights: BaseWeightsRecord
    container_image_digest: str | None
    library_versions: dict[str, str]
    gpu: GpuRecord | None
    final_loss: int | float | None
    error: str | None
    recipe_path: str
    overrides: list[str]
    checkpoint_dir: str | None
    metrics_path: str | None
    declared_hooks: dict[str, list[str]]
    precision_metrics: dict[str, Any]
    qat: dict[str, Any] | None
    lineage: dict[str, Any] | None = None
    cloned_from: ClonedFrom | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON object, with keys in console-summary order."""
        gpu = None if self.gpu is None else self.gpu.to_dict()
        document = {
            "schema": self.schema,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "family": self.family,
            "device": self.device,
            "precision": self.precision,
            "loss_scaling": self.loss_scaling,
            "precision_backend": self.precision_backend,
            "resolved_config": self.resolved_config,
            "seeds": self.seeds.to_dict(),
            "git": self.git.to_dict(),
            "dataset": self.dataset.to_dict(),
            "base_weights": self.base_weights.to_dict(),
            "container_image_digest": self.container_image_digest,
            "library_versions": dict(self.library_versions),
            "gpu": gpu,
            "final_loss": self.final_loss,
            "precision_metrics": _copy_precision_metrics(self.precision_metrics),
            "error": self.error,
            "recipe_path": self.recipe_path,
            "overrides": list(self.overrides),
            "checkpoint_dir": self.checkpoint_dir,
            "metrics_path": self.metrics_path,
            "declared_hooks": {
                "compression_hooks": list(
                    self.declared_hooks.get("compression_hooks", [])
                ),
                "eval_hooks": list(self.declared_hooks.get("eval_hooks", [])),
            },
            "qat": _copy_qat(self.qat),
        }
        if self.lineage is not None:
            document["lineage"] = json.loads(json.dumps(self.lineage))
        if self.cloned_from is not None:
            document["cloned_from"] = self.cloned_from.to_dict()
        return document

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> RunManifest:
        """Validate ``document`` and return it as a :class:`RunManifest`."""
        errors = validate_run_manifest(document)
        if errors:
            raise RunManifestError("invalid ohho.run/v1 manifest", errors)
        git = document["git"]
        dataset = document["dataset"]
        weights = document["base_weights"]
        seeds = document["seeds"]
        gpu = document["gpu"]
        hooks = document["declared_hooks"]
        return cls(
            schema=str(document["schema"]),
            status=str(document["status"]),
            started_at=str(document["started_at"]),
            finished_at=_optional_str(document["finished_at"]),
            family=str(document["family"]),
            device=str(document["device"]),
            precision=str(document["precision"]),
            loss_scaling=bool(document["loss_scaling"]),
            precision_backend=str(document["precision_backend"]),
            resolved_config=dict(document["resolved_config"]),
            seeds=SeedRecord(
                recipe=seeds["recipe"],
                python=seeds["python"],
                numpy=seeds["numpy"],
                torch=seeds["torch"],
                pythonhashseed=_optional_str(seeds["pythonhashseed"]),
                extra=dict(seeds["extra"]),
            ),
            git=GitState(
                commit=_optional_str(git["commit"]),
                dirty=bool(git["dirty"]),
                root=_optional_str(git["root"]),
            ),
            dataset=DatasetRecord(
                version_id=_optional_str(dataset["version_id"]),
                path=_optional_str(dataset["path"]),
                hash=_optional_str(dataset["hash"]),
            ),
            base_weights=BaseWeightsRecord(
                uri=_optional_str(weights["uri"]),
                revision=_optional_str(weights["revision"]),
                hash=_optional_str(weights["hash"]),
                loaded=bool(weights["loaded"]),
            ),
            container_image_digest=_optional_str(document["container_image_digest"]),
            library_versions={
                str(key): str(value)
                for key, value in document["library_versions"].items()
            },
            gpu=None
            if gpu is None
            else GpuRecord(
                name=_optional_str(gpu["name"]),
                driver=_optional_str(gpu["driver"]),
                cuda=_optional_str(gpu["cuda"]),
            ),
            final_loss=document["final_loss"],
            error=_optional_str(document["error"]),
            recipe_path=str(document["recipe_path"]),
            overrides=[str(item) for item in document["overrides"]],
            checkpoint_dir=_optional_str(document["checkpoint_dir"]),
            metrics_path=_optional_str(document["metrics_path"]),
            declared_hooks={
                "compression_hooks": [str(item) for item in hooks["compression_hooks"]],
                "eval_hooks": [str(item) for item in hooks["eval_hooks"]],
            },
            precision_metrics=_copy_precision_metrics(document["precision_metrics"]),
            qat=_copy_qat(document.get("qat")),
            lineage=_copy_lineage(document.get("lineage")),
            cloned_from=_cloned_from(document.get("cloned_from")),
        )


@dataclass(frozen=True)
class RunCapture:
    """Environment snapshot taken when a run starts. Reused when it ends."""

    started_at: str
    recipe_path: str
    overrides: list[str]
    family: str
    device: str
    precision: str
    loss_scaling: bool
    precision_backend: str
    resolved_config: dict[str, Any]
    seeds: SeedRecord
    git: GitState
    dataset: DatasetRecord
    base_uri: str | None
    base_revision: str | None
    base_hash: str | None
    library_versions: dict[str, str]
    gpu: GpuRecord | None
    container_image_digest: str | None
    declared_hooks: dict[str, list[str]]
    qat: dict[str, Any] | None
    lineage: dict[str, Any] | None = None


def schema_path() -> Path:
    """Path of the shipped ``ohho.run/v1`` JSON Schema."""
    return Path(__file__).resolve().parent / "schema" / SCHEMA_FILENAME


def schema_document() -> dict[str, Any]:
    """Return the parsed schema document."""
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(schema_path().read_text(encoding="utf-8"))
    return _SCHEMA


def validate_run_manifest(document: Any) -> list[str]:
    """Return schema and lifecycle errors. An empty list means the document is valid."""
    errors = validate_instance(document, schema_document())
    if not isinstance(document, dict):
        return errors
    status = document.get("status")
    finished = document.get("finished_at")
    error = document.get("error")
    started = document.get("started_at")
    if status == "running":
        if finished is not None:
            errors.append("$.finished_at: must be null while the run is running")
        if error is not None:
            errors.append("$.error: must be null while the run is running")
    elif status in {"succeeded", "failed", "interrupted"}:
        if not isinstance(finished, str):
            errors.append("$.finished_at: is required once the run has ended")
        elif isinstance(started, str) and finished < started:
            errors.append("$.finished_at: is earlier than started_at")
        if status == "succeeded" and error is not None:
            errors.append("$.error: must be null when the run succeeded")
        if status in {"failed", "interrupted"} and (
            not isinstance(error, str) or not error.strip()
        ):
            errors.append("$.error: is required when the run failed or was interrupted")
    errors.extend(_precision_errors(document))
    return errors


_LINEAGE_CONTRIBUTORS: list[
    Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
] = []


def register_lineage_contributor(
    contributor: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
) -> None:
    """Register a function that adds keys to a run's optional ``lineage``.

    Each compression method registers its own contributor and returns only
    the keys it owns, or ``None`` when the recipe does not use that method.
    Contributors run in registration order. The manifest omits ``lineage``
    when every contributor returns nothing.
    """
    if contributor not in _LINEAGE_CONTRIBUTORS:
        _LINEAGE_CONTRIBUTORS.append(contributor)


def _collect_lineage(recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for contributor in _LINEAGE_CONTRIBUTORS:
        part = contributor(recipe)
        if not isinstance(part, Mapping) or not part:
            continue
        for key, value in part.items():
            merged[str(key)] = value
    if not merged:
        return None
    return json.loads(json.dumps(merged))


def _copy_lineage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    return json.loads(json.dumps(value))


def _lineage_line(document: Mapping[str, Any]) -> str | None:
    lineage = document.get("lineage")
    if not isinstance(lineage, Mapping):
        return None
    teacher = lineage.get("teacher")
    if not isinstance(teacher, Mapping) or "version" not in teacher:
        return None
    return f"lineage.teacher.version: {_format_value(teacher.get('version'))}"


def set_run_store(store: RunStore | Callable[[Mapping[str, Any]], None] | None) -> None:
    """Install the process-local store hook. ``None`` clears it.

    The default is no hook. Training still writes ``run_manifest.json``.
    A callable of one argument is accepted in place of a :class:`RunStore`.
    Hook exceptions are written to stderr and do not change the process
    exit code.
    """
    global _STORE
    _STORE = store


def get_run_store() -> RunStore | Callable[[Mapping[str, Any]], None] | None:
    """Return the installed store hook, or ``None``."""
    return _STORE


def utc_now() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_state(start: str | Path) -> dict[str, Any]:
    """Commit and dirty flag for the checkout that contains ``start``.

    Walks parents of ``start`` for a ``.git`` entry. A missing checkout yields
    a null commit and ``dirty`` false. When a checkout exists and ``git status``
    cannot be read, ``dirty`` is true so an unreadable tree is not reported
    as clean. ``commit`` is the full hash, or null when there is no commit yet.
    """
    root = _git_root(Path(start))
    if root is None:
        return {"commit": None, "dirty": False, "root": None}
    commit_text = _git_output(root, ["rev-parse", "HEAD"])
    status = _git_output(root, ["status", "--porcelain", "--untracked-files=normal"])
    commit: str | None = None
    if commit_text is not None and _COMMIT.fullmatch(commit_text.strip()):
        commit = commit_text.strip()
    if status is None:
        dirty = True
    else:
        dirty = bool(status.strip())
    return {"commit": commit, "dirty": dirty, "root": str(root)}


def hash_tree(root: Path) -> str:
    """sha256 hex of files under ``root``, in sorted relative-path order.

    Directory symlinks are not followed. A symlinked file contributes its
    link text, not the target bytes.
    """
    digest = hashlib.sha256()
    base = root.resolve()
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            files.append(Path(dirpath) / name)
    for file in files:
        rel = file.relative_to(base).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        if file.is_symlink():
            digest.update(b"link\0")
            target = os.readlink(file)
            digest.update(target.encode("utf-8", errors="surrogateescape"))
        else:
            digest.update(b"file\0")
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def hash_base_weights(uri: str | None) -> str | None:
    """sha256 hex of a local weights file or directory.

    A remote URI (``https://``, ``hf://``, ``local:``, and any other scheme)
    returns null. This function does not download anything.
    """
    path = _local_weight_path(uri)
    if path is None or not path.exists():
        return None
    try:
        if path.is_file():
            return _sha256_file(path)
        if path.is_dir():
            return hash_tree(path)
    except OSError:
        return None
    return None


def in_container() -> bool:
    """True when the process looks like it is inside a container."""
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    try:
        text = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    markers = ("docker", "kubepods", "containerd", "libpod")
    return any(marker in text for marker in markers)


def container_image_digest() -> str | None:
    """Image digest inside a container, otherwise null.

    Inside a container the digest is the first non-empty value of
    ``OHHO_IMAGE_DIGEST`` or the first line of ``/etc/ohho/image-digest``
    or ``/.ohho-image-digest``. Outside a container the result is null
    even when that variable is set.
    """
    if not in_container():
        return None
    value = os.environ.get("OHHO_IMAGE_DIGEST", "").strip()
    if value:
        return value
    for candidate in (Path("/etc/ohho/image-digest"), Path("/.ohho-image-digest")):
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text.splitlines()[0].strip()
    return None


def library_versions() -> dict[str, str]:
    """Python, ohho-os, and installed torch, numpy, and lerobot versions.

    Missing optional libraries are omitted. Versions come from package
    metadata, so importing torch is not required for this map.
    """
    versions = {"python": sys.version.split()[0], "ohho": _ohho_version()}
    for dist in ("torch", "numpy", "lerobot"):
        found = _dist_version(dist)
        if found:
            versions[dist] = found
    return versions


def gpu_snapshot() -> dict[str, str | None] | None:
    """CUDA device, driver, and CUDA version, or null when none is reported."""
    torch_info = _cuda_from_torch() or {}
    smi_info = _cuda_from_nvidia_smi() or {}
    name = _text(torch_info.get("name")) or _text(smi_info.get("name"))
    driver = _text(smi_info.get("driver")) or _text(torch_info.get("driver"))
    cuda = _text(torch_info.get("cuda")) or _text(smi_info.get("cuda"))
    if name is None and driver is None and cuda is None:
        return None
    return {"name": name, "driver": driver, "cuda": cuda}


def capture_run(
    *,
    recipe: Mapping[str, Any],
    recipe_path: str,
    overrides: Sequence[str],
    device: str,
    started_at: str | None = None,
) -> RunCapture:
    """Record the environment and apply ``hyperparameters.seed`` when it is set."""
    recipe_file = Path(recipe_path).expanduser().resolve()
    used = device.strip() if isinstance(device, str) else ""
    if not used:
        raw = recipe.get("device")
        used = raw if isinstance(raw, str) and raw else "cpu"
    resolved = json.loads(json.dumps(dict(recipe)))
    resolved["device"] = used
    family = recipe.get("family")
    family_id = family if isinstance(family, str) else ""
    precision = _capture_precision(recipe)
    loss_scaling, precision_backend = backend_for(precision)
    git = git_state(recipe_file.parent)
    gpu = gpu_snapshot()
    weights_uri, weights_revision, weights_hash = _base_identity(recipe)
    return RunCapture(
        started_at=started_at or utc_now(),
        recipe_path=str(recipe_file),
        overrides=[str(item) for item in overrides],
        family=family_id,
        device=used,
        precision=precision,
        loss_scaling=loss_scaling,
        precision_backend=precision_backend,
        resolved_config=resolved,
        seeds=collect_seeds(recipe),
        git=GitState(commit=git["commit"], dirty=bool(git["dirty"]), root=git["root"]),
        dataset=dataset_record(recipe),
        base_uri=weights_uri,
        base_revision=weights_revision,
        base_hash=weights_hash,
        library_versions=library_versions(),
        gpu=None if gpu is None else GpuRecord(**gpu),
        container_image_digest=container_image_digest(),
        declared_hooks=declared_hooks(recipe),
        qat=_qat_from_recipe(recipe),
        lineage=_collect_lineage(recipe),
    )


def manifest_from_capture(
    capture: RunCapture,
    *,
    status: str,
    finished_at: str | None,
    error: str | None,
    checkpoint_dir: str | None,
    metrics_path: str | None,
    final_loss: Any,
    base_weights_loaded: bool,
    device: str | None = None,
    throughput_steps_per_s: Any = None,
    peak_memory_bytes: Any = None,
) -> RunManifest:
    """Build a manifest from a start snapshot plus the outcome fields."""
    used = device.strip() if isinstance(device, str) and device.strip() else ""
    if not used:
        used = capture.device
    resolved = json.loads(json.dumps(capture.resolved_config))
    resolved["device"] = used
    current = git_state(Path(capture.recipe_path).parent)
    git = GitState(
        commit=capture.git.commit,
        dirty=bool(capture.git.dirty or current["dirty"]),
        root=capture.git.root,
    )
    if status == "running":
        finished_at = None
        error = None
    elif status == "succeeded":
        error = None
    elif not isinstance(error, str) or not error.strip():
        error = status
    return RunManifest(
        schema=SCHEMA_ID,
        status=status,
        started_at=capture.started_at,
        finished_at=finished_at,
        family=capture.family,
        device=used,
        precision=capture.precision,
        loss_scaling=capture.loss_scaling,
        precision_backend=capture.precision_backend,
        resolved_config=resolved,
        seeds=capture.seeds,
        git=git,
        dataset=capture.dataset,
        base_weights=BaseWeightsRecord(
            uri=capture.base_uri,
            revision=capture.base_revision,
            hash=capture.base_hash,
            loaded=bool(base_weights_loaded),
        ),
        container_image_digest=capture.container_image_digest,
        library_versions=dict(capture.library_versions),
        gpu=capture.gpu,
        final_loss=_json_number(final_loss),
        error=error,
        recipe_path=capture.recipe_path,
        overrides=list(capture.overrides),
        checkpoint_dir=checkpoint_dir,
        metrics_path=metrics_path,
        declared_hooks={
            "compression_hooks": list(capture.declared_hooks["compression_hooks"]),
            "eval_hooks": list(capture.declared_hooks["eval_hooks"]),
        },
        precision_metrics=blank_precision_metrics(
            capture.precision,
            normalize_throughput(throughput_steps_per_s),
            normalize_peak(peak_memory_bytes),
        ),
        qat=_copy_qat(capture.qat),
        lineage=_copy_lineage(capture.lineage),
    )


def write_run_manifest(path: Path, manifest: RunManifest) -> Path:
    """Validate, write ``path`` atomically, then call the store hook."""
    document = manifest.to_dict()
    errors = validate_run_manifest(document)
    if errors:
        raise RunManifestError(
            "refusing to write an invalid ohho.run/v1 manifest", errors
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2) + "\n"
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    _publish(document)
    return path


def load_run_manifest(path: str | Path) -> RunManifest:
    """Read and validate a manifest file."""
    file = Path(path)
    try:
        document = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunManifestError(f"could not read run manifest: {exc}") from exc
    return RunManifest.from_dict(document)


def show_run(run_dir: str | Path, *, as_json: bool = False) -> int:
    """Print the manifest in ``run_dir``. Return 0 on success and 1 otherwise."""
    path = resolve_manifest_path(run_dir)
    if path is None:
        print(f"no run manifest in {Path(run_dir).expanduser()}", file=sys.stderr)
        return 1
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read run manifest: {exc}", file=sys.stderr)
        return 1
    errors = validate_run_manifest(document)
    if errors:
        print("run manifest is not valid ohho.run/v1", file=sys.stderr)
        for line in errors:
            print(line, file=sys.stderr)
        return 1
    if as_json:
        sys.stdout.write(json.dumps(document, indent=2) + "\n")
        return 0
    sys.stdout.write(format_run_text(document))
    return 0


def resolve_manifest_path(run_dir: str | Path) -> Path | None:
    """``run_manifest.json`` inside a directory, or that file itself."""
    path = Path(run_dir).expanduser()
    if path.is_file() and path.name == MANIFEST_NAME:
        return path
    candidate = path / MANIFEST_NAME
    if candidate.is_file():
        return candidate
    return None


def format_run_text(document: Mapping[str, Any]) -> str:
    """Human summary. The same fields are the console run-page summary."""
    seeds = document.get("seeds") if isinstance(document.get("seeds"), dict) else {}
    git = document.get("git") if isinstance(document.get("git"), dict) else {}
    dataset = (
        document.get("dataset") if isinstance(document.get("dataset"), dict) else {}
    )
    weights = (
        document.get("base_weights")
        if isinstance(document.get("base_weights"), dict)
        else {}
    )
    versions = document.get("library_versions")
    if not isinstance(versions, dict):
        versions = {}
    lines = [
        SCHEMA_ID,
        f"status: {_format_value(document.get('status'))}",
        f"started_at: {_format_value(document.get('started_at'))}",
        f"finished_at: {_format_value(document.get('finished_at'))}",
        f"family: {_format_value(document.get('family'))}",
        f"device: {_format_value(document.get('device'))}",
        f"precision: {_format_value(document.get('precision'))}",
        f"loss_scaling: {_format_value(document.get('loss_scaling'))}",
        f"precision_backend: {_format_value(document.get('precision_backend'))}",
    ]
    for key in ("recipe", "python", "numpy", "torch", "pythonhashseed", "extra"):
        lines.append(f"seeds.{key}: {_format_value(seeds.get(key))}")
    lines.extend(
        [
            f"git.commit: {_format_value(git.get('commit'))}",
            f"git.dirty: {_format_value(git.get('dirty'))}",
            f"dataset.version_id: {_format_value(dataset.get('version_id'))}",
            f"dataset.hash: {_format_value(dataset.get('hash'))}",
            f"dataset.path: {_format_value(dataset.get('path'))}",
            f"base_weights.uri: {_format_value(weights.get('uri'))}",
            f"base_weights.revision: {_format_value(weights.get('revision'))}",
            f"base_weights.hash: {_format_value(weights.get('hash'))}",
            f"base_weights.loaded: {_format_value(weights.get('loaded'))}",
        ]
    )
    lineage_line = _lineage_line(document)
    if lineage_line is not None:
        lines.append(lineage_line)
    lines.extend(
        [
            "container_image_digest: "
            + _format_value(document.get("container_image_digest")),
            "library_versions: "
            + " ".join(f"{key}={versions[key]}" for key in sorted(versions)),
            "gpu: " + _format_gpu(document.get("gpu")),
            f"final_loss: {_format_value(document.get('final_loss'))}",
            "precision_metrics: "
            + _format_precision_metrics(document.get("precision_metrics")),
            f"qat: {_format_value(document.get('qat'))}",
            f"error: {_format_value(document.get('error'))}",
            "resolved_config: "
            + json.dumps(
                document.get("resolved_config"),
                sort_keys=True,
                separators=(", ", ": "),
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def collect_seeds(recipe: Mapping[str, Any]) -> SeedRecord:
    """Read seeds from the recipe and apply the recipe seed to installed libraries."""
    hyperparameters = recipe.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        hyperparameters = {}
    recipe_seed = hyperparameters.get("seed")
    if isinstance(recipe_seed, bool) or not isinstance(recipe_seed, int):
        recipe_seed = None
    extra: dict[str, int | str | None] = {}
    for key, value in hyperparameters.items():
        if not isinstance(key, str) or key == "seed" or not key.endswith("_seed"):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int) or isinstance(value, str) or value is None:
            extra[key] = value
    python_seed = None
    numpy_seed = None
    torch_seed = None
    if recipe_seed is not None:
        random.seed(recipe_seed)
        python_seed = recipe_seed
        numpy_seed = _seed_numpy(recipe_seed)
        torch_seed = _seed_torch(recipe_seed)
    hashseed = os.environ.get("PYTHONHASHSEED", "").strip()
    return SeedRecord(
        recipe=recipe_seed,
        python=python_seed,
        numpy=numpy_seed,
        torch=torch_seed,
        pythonhashseed=hashseed or None,
        extra=extra,
    )


def dataset_record(recipe: Mapping[str, Any]) -> DatasetRecord:
    """Version id from the recipe, plus a hash when a local path exists."""
    dataset = recipe.get("dataset")
    if not isinstance(dataset, dict):
        dataset = {}
    version_id = _nonempty_str(dataset.get("version_id"))
    raw_path = dataset.get("path")
    path_text = None
    digest = None
    if isinstance(raw_path, str) and raw_path.strip():
        path = Path(raw_path).expanduser()
        try:
            path_text = str(path.resolve())
        except OSError:
            path_text = str(path)
        if path.exists():
            try:
                if path.is_file():
                    digest = _sha256_file(path)
                elif path.is_dir():
                    digest = hash_tree(path)
            except OSError:
                digest = None
    return DatasetRecord(version_id=version_id, path=path_text, hash=digest)


def declared_hooks(recipe: Mapping[str, Any]) -> dict[str, list[str]]:
    """Hook names recorded on the manifest. The runner does not call them."""
    return {
        "compression_hooks": _hook_names(recipe.get("compression_hooks")),
        "eval_hooks": _hook_names(recipe.get("eval_hooks")),
    }


def _publish(document: Mapping[str, Any]) -> None:
    store = _STORE
    if store is None:
        return
    try:
        payload = json.loads(json.dumps(document))
        save = getattr(store, "save", None)
        if callable(save):
            save(payload)
        else:
            store(payload)
    except Exception as exc:
        print(
            f"run store hook failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _hook_names(hooks: Any) -> list[str]:
    if not isinstance(hooks, list):
        return []
    names: list[str] = []
    for hook in hooks:
        if isinstance(hook, dict) and isinstance(hook.get("name"), str):
            names.append(hook["name"])
    return names


def _base_identity(
    recipe: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None]:
    weights = recipe.get("base_weights")
    if not isinstance(weights, dict):
        return None, None, None
    uri = _nonempty_str(weights.get("uri"))
    revision = _nonempty_str(weights.get("revision"))
    return uri, revision, hash_base_weights(uri)


def _local_weight_path(uri: str | None) -> Path | None:
    if not isinstance(uri, str) or not uri.strip():
        return None
    text = uri.strip()
    if text.startswith("file://"):
        return _file_uri_path(text)
    if "://" in text or text.startswith("local:"):
        return None
    return Path(text).expanduser()


def _file_uri_path(uri: str) -> Path | None:
    raw = uri[len("file://") :]
    if raw.startswith("//"):
        return None
    if len(raw) > 3 and raw[0] == "/" and raw[2] == ":":
        raw = raw[1:]
    if not raw:
        return None
    return Path(raw)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_root(start: Path) -> Path | None:
    try:
        current = start.resolve()
    except OSError:
        return None
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_output(root: Path, args: list[str]) -> str | None:
    command = [
        "git",
        "-c",
        "safe.directory=*",
        "-c",
        "core.autocrlf=false",
        "-c",
        "core.pager=cat",
        *args,
    ]
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    try:
        proc = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=env,
            stdin=subprocess.DEVNULL,
            **_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _cuda_from_torch() -> dict[str, str | None] | None:
    try:
        import torch
    except ImportError:
        return None
    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        return None
    if not available:
        return None
    name = None
    try:
        name = str(torch.cuda.get_device_name(0)).strip() or None
    except Exception:
        name = None
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    cuda_text = cuda.strip() if isinstance(cuda, str) and cuda.strip() else None
    return {"name": name, "driver": None, "cuda": cuda_text}


def _cuda_from_nvidia_smi() -> dict[str, str | None] | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
            stdin=subprocess.DEVNULL,
            **_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 2:
        return None
    driver = parts[-1] or None
    name = ",".join(parts[:-1]).strip() or None
    return {"name": name, "driver": driver, "cuda": None}


def _seed_numpy(seed: int) -> int | None:
    try:
        import numpy as np
    except ImportError:
        return None
    np.random.seed(seed)
    return seed


def _seed_torch(seed: int) -> int | None:
    try:
        import torch
    except ImportError:
        return None
    torch.manual_seed(seed)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and getattr(cuda, "is_available", lambda: False)():
        try:
            cuda.manual_seed_all(seed)
        except Exception:
            pass
    return seed


def _ohho_version() -> str:
    from ohho import __version__

    return __version__


def _dist_version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return None
    return value


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(", ", ": "))


def _format_gpu(value: Any) -> str:
    if not isinstance(value, dict):
        return "null"
    return " ".join(
        f"{key}={_format_value(value.get(key))}" for key in ("name", "driver", "cuda")
    )


def _cloned_from(value: Any) -> ClonedFrom | None:
    if value is None:
        return None
    return ClonedFrom(run=str(value["run"]), manifest=str(value["manifest"]))


def _format_precision_metrics(value: Any) -> str:
    if not isinstance(value, dict):
        return "null"
    parts: list[str] = []
    for name in PRECISIONS:
        slot = value.get(name)
        if not isinstance(slot, dict):
            parts.append(f"{name}=null")
            continue
        throughput = _format_value(slot.get("throughput_steps_per_s"))
        peak = _format_value(slot.get("peak_memory_bytes"))
        parts.append(
            f"{name} throughput_steps_per_s={throughput} peak_memory_bytes={peak}"
        )
    return " ".join(parts)


def _qat_from_recipe(recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    from ohho.compression.qat import settings_from_recipe

    return settings_from_recipe(recipe)


def _copy_qat(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return json.loads(json.dumps(dict(value)))


def _capture_precision(recipe: Mapping[str, Any]) -> str:
    precision = recipe.get("precision", "fp32")
    if isinstance(precision, str) and precision in PRECISIONS:
        return precision
    return "fp32"


def _copy_precision_metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for name in PRECISIONS:
        slot = raw.get(name) if isinstance(raw, Mapping) else None
        if not isinstance(slot, dict):
            copied[name] = None
            continue
        copied[name] = {
            "throughput_steps_per_s": slot.get("throughput_steps_per_s"),
            "peak_memory_bytes": slot.get("peak_memory_bytes"),
        }
    return copied


def _precision_errors(document: Mapping[str, Any]) -> list[str]:
    precision = document.get("precision")
    if precision not in PRECISIONS:
        return []
    errors: list[str] = []
    loss_scaling, backend = backend_for(str(precision))
    if document.get("loss_scaling") is not loss_scaling:
        errors.append(
            "$.loss_scaling: must be true for fp16 and false for fp32, bf16, and fp8"
        )
    if document.get("precision_backend") != backend:
        errors.append("$.precision_backend: does not match precision")
    metrics = document.get("precision_metrics")
    if not isinstance(metrics, dict):
        return errors
    for name in PRECISIONS:
        value = metrics.get(name)
        if name == precision:
            if not isinstance(value, dict):
                errors.append(
                    f"$.precision_metrics.{name}: record throughput and peak memory "
                    "for the precision this run used"
                )
        elif value is not None:
            errors.append(
                f"$.precision_metrics.{name}: must be null for a {precision} run"
            )
    return errors


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _nonempty_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _subprocess_kwargs() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}
