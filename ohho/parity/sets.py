"""Versioned parity sets: golden inputs and FP32 reference outputs.

A set is a directory, or a ``.tar.gz`` archive of that directory. It holds
``manifest.json`` (``ohho.parity_set/v1``) and ``samples.json``. The manifest
``sha256`` is the SHA-256 of the ``samples.json`` bytes. Those bytes are the
inputs and the FP32 reference outputs. Media files copied from a camera log
travel with the directory and are not part of the hash.

``synthetic`` and ``ephemeral`` are reserved ids. A scored run that did not
load a set uses one of those ids in the parity report. Capture will not write
them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ohho.parity.errors import ParityError

SET_SCHEMA_ID = "ohho.parity_set/v1"
SET_SCHEMA_URL = "https://ohho-robotics.com/schemas/parity-set.v1.json"
_SAMPLES_NAME = "samples.json"
_MANIFEST_NAME = "manifest.json"
_SOURCES = frozenset({"dataset", "recorded", "camera"})
_RESERVED_IDS = frozenset({"synthetic", "ephemeral"})
_SLUG = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_HEX = frozenset("0123456789abcdef")
_MANIFEST_KEYS = (
    "schema",
    "id",
    "version",
    "model",
    "submodule",
    "source",
    "path",
    "synthetic",
    "seed",
    "n_requested",
    "n",
    "indices",
    "dtype",
    "sha256",
    "samples",
    "media",
)


@dataclass(frozen=True)
class ParitySet:
    """One captured set. ``location`` is the directory or archive that was loaded."""

    set_id: str
    version: int
    model: str
    submodule: str | None
    source: str
    path: str | None
    synthetic: bool
    seed: int
    n_requested: int
    n: int
    indices: tuple[int, ...] | None
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    sha256: str
    media: tuple[str | None, ...]
    location: str

    def report_record(self) -> dict[str, Any]:
        """The ``parity_set`` object stored on an ``ohho.parity/v1`` report."""
        source = "synthetic" if self.synthetic else self.source
        return {
            "id": self.set_id,
            "version": self.version,
            "sha256": self.sha256,
            "synthetic": self.synthetic,
            "source": source,
        }

    def manifest_document(self) -> dict[str, Any]:
        """The on-disk manifest. Samples stay in ``samples.json``."""
        return {
            "schema": SET_SCHEMA_ID,
            "id": self.set_id,
            "version": self.version,
            "model": self.model,
            "submodule": self.submodule,
            "source": self.source,
            "path": self.path,
            "synthetic": self.synthetic,
            "seed": self.seed,
            "n_requested": self.n_requested,
            "n": self.n,
            "indices": None if self.indices is None else list(self.indices),
            "dtype": "float32",
            "sha256": self.sha256,
            "samples": _SAMPLES_NAME,
            "media": list(self.media),
        }


def set_schema_text() -> str:
    """The JSON Schema document shipped with the package."""
    from importlib.resources import files

    path = files("ohho.parity").joinpath("parity_set.schema.json")
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def unscored_parity_set() -> dict[str, Any]:
    """Synthetic record used when no samples were scored.

    ``sha256`` is null. A scored fallback uses :func:`generated_parity_set`.
    """
    return {
        "id": "synthetic",
        "version": 1,
        "sha256": None,
        "synthetic": True,
        "source": "synthetic",
    }


def generated_parity_set(
    source: str, batches: dict[str, Any], outputs: dict[str, Any]
) -> dict[str, Any]:
    """Identity of samples that were scored without loading a captured set.

    Random inputs are the synthetic fallback (``id`` ``synthetic``). A dataset,
    sim rollout, or camera log passed with ``--inputs`` and no ``--parity-set``
    is recorded as ``ephemeral`` and is not marked synthetic.
    """
    digest = hashlib.sha256(canonical_samples(batches, outputs)).hexdigest()
    if source == "random":
        return {
            "id": "synthetic",
            "version": 1,
            "sha256": digest,
            "synthetic": True,
            "source": "synthetic",
        }
    if source not in _SOURCES:
        raise ParityError(f"Cannot record a parity set from {source!r}.")
    return {
        "id": "ephemeral",
        "version": 1,
        "sha256": digest,
        "synthetic": False,
        "source": source,
    }


def canonical_samples(inputs: dict[str, Any], outputs: dict[str, Any]) -> bytes:
    """UTF-8 bytes of ``samples.json``, including the trailing newline.

    Keys are sorted. The SHA-256 of these bytes is the set id's content hash.
    """
    count = _check_named_tensors(inputs, outputs)
    if count < 1:
        raise ParityError("A parity set needs at least one sample.")
    try:
        text = json.dumps(
            {"inputs": inputs, "outputs": outputs},
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ParityError(f"Parity set samples are not finite numbers: {exc}") from exc
    return (text + "\n").encode("utf-8")


def write_parity_set(
    out: str | Path,
    *,
    set_id: str,
    version: int,
    model: str,
    submodule: str | None,
    source: str,
    source_path: str | None,
    seed: int,
    n_requested: int,
    indices: list[int] | None,
    batches: dict[str, Any],
    outputs: dict[str, Any],
    media_files: tuple[str | None, ...] | None = None,
) -> ParitySet:
    """Write a directory, or a ``.tar.gz`` / ``.tgz`` archive when ``out`` ends in one.

    The archive contains the same manifest and samples. Its sha256 matches the
    directory form because the hash is of ``samples.json``, not of the gzip bytes.
    """
    check_set_id(set_id)
    if source not in _SOURCES:
        raise ParityError(
            "A captured parity set comes from a dataset, a sim rollout, or a camera log."
        )
    if type(version) is not int or version < 1:
        raise ParityError("--version must be an integer >= 1.")
    if type(seed) is not int:
        raise ParityError("--seed must be an integer.")
    if not isinstance(model, str) or not model.strip():
        raise ParityError("A parity set needs a reference model.")
    if submodule is not None and (not isinstance(submodule, str) or not submodule):
        raise ParityError("submodule must be a string or omitted.")
    if source_path is not None and not isinstance(source_path, str):
        raise ParityError("The input path must be a string or omitted.")
    payload = canonical_samples(batches, outputs)
    count = _batch_count(next(iter(batches.values())))
    if type(n_requested) is not int or n_requested < count:
        raise ParityError("--n must be at least the number of stored samples.")
    if indices is None or len(indices) != count:
        raise ParityError("A captured parity set must list one index per sample.")
    if any(type(item) is not int or item < 0 for item in indices):
        raise ParityError("Sample indices must be non-negative integers.")
    digest = hashlib.sha256(payload).hexdigest()
    destination = Path(out)
    archive = _is_archive(destination)
    if archive:
        with tempfile.TemporaryDirectory(prefix="ohho-parity-set-") as tmp:
            directory = Path(tmp) / "set"
            _write_directory(
                directory,
                payload=payload,
                media_files=media_files,
                manifest=_manifest_body(
                    set_id=set_id,
                    version=version,
                    model=model.strip(),
                    submodule=submodule,
                    source=source,
                    source_path=source_path,
                    seed=seed,
                    n_requested=n_requested,
                    count=count,
                    indices=indices,
                    digest=digest,
                    media=[],
                ),
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            _pack(directory, destination)
        location = destination
    else:
        if destination.exists() and not destination.is_dir():
            raise ParityError(f"--out exists and is not a directory: {destination}")
        _write_directory(
            destination,
            payload=payload,
            media_files=media_files,
            manifest=_manifest_body(
                set_id=set_id,
                version=version,
                model=model.strip(),
                submodule=submodule,
                source=source,
                source_path=source_path,
                seed=seed,
                n_requested=n_requested,
                count=count,
                indices=indices,
                digest=digest,
                media=[],
            ),
        )
        location = destination
    return load_parity_set(location)


def load_parity_set(path: str | Path) -> ParitySet:
    """Load a directory, a ``manifest.json`` file, or a ``.tar.gz`` archive.

    The samples hash is checked against the manifest before the set is returned.
    """
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise ParityError(f"Parity set does not exist: {candidate}")
    if candidate.is_file() and _is_archive(candidate):
        return _load_archive(candidate)
    if candidate.is_file() and candidate.name == _MANIFEST_NAME:
        return _load_directory(candidate.parent, location=candidate.parent)
    if candidate.is_dir():
        return _load_directory(candidate, location=candidate)
    raise ParityError(
        f"{candidate} is not a parity set directory, manifest.json, or .tar.gz archive."
    )


def check_set_id(set_id: str) -> str:
    """Reject an empty, non-slug, or reserved parity set id."""
    if not isinstance(set_id, str) or _SLUG.fullmatch(set_id) is None:
        raise ParityError(
            f"Parity set id {set_id!r} must be a lowercase slug such as act-golden."
        )
    if set_id in _RESERVED_IDS:
        raise ParityError(f"Parity set id {set_id!r} is reserved.")
    return set_id


def _manifest_body(
    *,
    set_id: str,
    version: int,
    model: str,
    submodule: str | None,
    source: str,
    source_path: str | None,
    seed: int,
    n_requested: int,
    count: int,
    indices: list[int],
    digest: str,
    media: list[str | None],
) -> dict[str, Any]:
    return {
        "schema": SET_SCHEMA_ID,
        "id": set_id,
        "version": version,
        "model": model,
        "submodule": submodule,
        "source": source,
        "path": source_path,
        "synthetic": False,
        "seed": seed,
        "n_requested": n_requested,
        "n": count,
        "indices": list(indices),
        "dtype": "float32",
        "sha256": digest,
        "samples": _SAMPLES_NAME,
        "media": media,
    }


def _write_directory(
    directory: Path,
    *,
    payload: bytes,
    media_files: tuple[str | None, ...] | None,
    manifest: dict[str, Any],
) -> list[str | None]:
    directory.mkdir(parents=True, exist_ok=True)
    stored_media = _copy_media(directory, media_files)
    manifest["media"] = stored_media
    _write_bytes(directory / _SAMPLES_NAME, payload)
    _write_bytes(
        directory / _MANIFEST_NAME,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return stored_media


def _copy_media(
    directory: Path, media_files: tuple[str | None, ...] | None
) -> list[str | None]:
    if not media_files or all(item is None for item in media_files):
        stale = directory / "media"
        if stale.exists():
            shutil.rmtree(stale)
        return []
    media_dir = directory / "media"
    if media_dir.exists():
        shutil.rmtree(media_dir)
    media_dir.mkdir()
    stored: list[str | None] = []
    for index, item in enumerate(media_files):
        if item is None:
            stored.append(None)
            continue
        source = Path(item)
        if not source.is_file():
            raise ParityError(f"Camera frame does not exist: {source}")
        dest_name = f"{index:04d}{source.suffix.lower()}"
        shutil.copyfile(source, media_dir / dest_name)
        stored.append(f"media/{dest_name}")
    return stored


def _load_directory(directory: Path, *, location: Path) -> ParitySet:
    manifest_path = directory / _MANIFEST_NAME
    samples_path = directory / _SAMPLES_NAME
    if not manifest_path.is_file() or not samples_path.is_file():
        raise ParityError(
            f"{directory} is not a parity set (missing manifest.json or samples.json)."
        )
    manifest = _read_manifest(manifest_path)
    raw = samples_path.read_bytes()
    return _assemble(manifest, raw, location=str(location), directory=directory)


def _load_archive(archive: Path) -> ParitySet:
    try:
        handle = tarfile.open(archive, "r:gz")
    except (tarfile.TarError, OSError) as exc:
        raise ParityError(
            f"Could not read parity set archive {archive}: {exc}"
        ) from exc
    with handle:
        manifest_bytes = _tar_bytes(handle, _MANIFEST_NAME)
        samples_bytes = _tar_bytes(handle, _SAMPLES_NAME)
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ParityError(f"{archive} manifest.json is not JSON.") from exc
        _check_manifest(manifest)
        media = manifest.get("media")
        if isinstance(media, list):
            for item in media:
                if isinstance(item, str):
                    _tar_bytes(handle, item)
    # Media members were checked inside the archive. Do not require them on disk.
    return _assemble(manifest, samples_bytes, location=str(archive), directory=None)


def _assemble(
    manifest: dict[str, Any],
    samples: bytes,
    *,
    location: str,
    directory: Path | None,
) -> ParitySet:
    _check_manifest(manifest)
    digest = hashlib.sha256(samples).hexdigest()
    if digest != manifest["sha256"]:
        raise ParityError(
            f"Parity set sha256 does not match samples.json (got {digest})."
        )
    try:
        parsed = json.loads(samples.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParityError("samples.json is not JSON.") from exc
    if not isinstance(parsed, dict):
        raise ParityError("samples.json must be an object.")
    inputs = parsed.get("inputs")
    outputs = parsed.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ParityError("samples.json needs inputs and outputs objects.")
    extra = [key for key in parsed if key not in {"inputs", "outputs"}]
    if extra:
        raise ParityError("samples.json has unknown keys: " + ", ".join(sorted(extra)))
    count = _check_named_tensors(inputs, outputs)
    if count != manifest["n"]:
        raise ParityError("samples.json does not match manifest n.")
    media_root = directory if directory is not None and directory.is_dir() else None
    media = _check_media(manifest["media"], media_root)
    indices = manifest["indices"]
    return ParitySet(
        set_id=manifest["id"],
        version=manifest["version"],
        model=manifest["model"],
        submodule=manifest["submodule"],
        source=manifest["source"],
        path=manifest["path"],
        synthetic=False,
        seed=manifest["seed"],
        n_requested=manifest["n_requested"],
        n=count,
        indices=tuple(indices),
        inputs=inputs,
        outputs=outputs,
        sha256=digest,
        media=tuple(media),
        location=location,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParityError(f"Could not read {path}: {exc}") from exc
    _check_manifest(document)
    return document


def _check_manifest(document: object) -> None:
    if not isinstance(document, dict):
        raise ParityError("Parity set manifest must be an object.")
    missing = [key for key in _MANIFEST_KEYS if key not in document]
    extra = [key for key in document if key not in _MANIFEST_KEYS]
    if missing:
        raise ParityError("Parity set manifest missing " + ", ".join(missing))
    if extra:
        raise ParityError(
            "Parity set manifest has unknown keys: " + ", ".join(sorted(extra))
        )
    if document["schema"] != SET_SCHEMA_ID:
        raise ParityError(f"Parity set schema must be {SET_SCHEMA_ID}.")
    check_set_id(str(document["id"]))
    if type(document["version"]) is not int or document["version"] < 1:
        raise ParityError("Parity set version must be an integer >= 1.")
    if not isinstance(document["model"], str) or not document["model"]:
        raise ParityError("Parity set model must be a string.")
    submodule = document["submodule"]
    if submodule is not None and (not isinstance(submodule, str) or not submodule):
        raise ParityError("Parity set submodule must be a string or null.")
    if document["source"] not in _SOURCES:
        raise ParityError("Parity set source must be dataset, recorded, or camera.")
    path = document["path"]
    if path is not None and not isinstance(path, str):
        raise ParityError("Parity set path must be a string or null.")
    if document["synthetic"] is not False:
        raise ParityError("A stored parity set is not synthetic.")
    if type(document["seed"]) is not int:
        raise ParityError("Parity set seed must be an integer.")
    n_requested = document["n_requested"]
    count = document["n"]
    if type(n_requested) is not int or n_requested < 1:
        raise ParityError("Parity set n_requested must be an integer >= 1.")
    if type(count) is not int or count < 1 or count > n_requested:
        raise ParityError("Parity set n must be an integer from 1 to n_requested.")
    indices = document["indices"]
    if not isinstance(indices, list) or len(indices) != count:
        raise ParityError("Parity set indices must list every stored sample.")
    if any(type(item) is not int or item < 0 for item in indices):
        raise ParityError("Parity set indices must be non-negative integers.")
    if document["dtype"] != "float32":
        raise ParityError("Parity set dtype must be float32.")
    digest = document["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(ch not in _HEX for ch in digest)
    ):
        raise ParityError("Parity set sha256 must be 64 lowercase hex characters.")
    if document["samples"] != _SAMPLES_NAME:
        raise ParityError("Parity set samples must be samples.json.")
    if not isinstance(document["media"], list):
        raise ParityError("Parity set media must be a list.")


def _check_media(media: list[Any], directory: Path | None) -> list[str | None]:
    stored: list[str | None] = []
    for item in media:
        if item is None:
            stored.append(None)
            continue
        if not isinstance(item, str) or not item:
            raise ParityError(
                "Parity set media entries must be relative paths or null."
            )
        relative = Path(item)
        if relative.is_absolute() or ".." in relative.parts:
            raise ParityError(f"Parity set media path is not relative: {item}")
        if directory is not None and not (directory / relative).is_file():
            raise ParityError(f"Parity set is missing media file {item}.")
        stored.append(item)
    return stored


def _check_named_tensors(inputs: object, outputs: object) -> int:
    if not isinstance(inputs, dict) or not inputs:
        raise ParityError("Parity set inputs must name at least one tensor.")
    if not isinstance(outputs, dict) or not outputs:
        raise ParityError("Parity set outputs must name at least one tensor.")
    count: int | None = None
    for label, group in (("inputs", inputs), ("outputs", outputs)):
        for name, value in group.items():
            if not isinstance(name, str) or not name:
                raise ParityError(f"Parity set {label} need string names.")
            length = _check_tensor(value, f"{label}.{name}")
            if count is None:
                count = length
            elif length != count:
                raise ParityError(
                    "Parity set inputs and outputs must share one batch length."
                )
    assert count is not None
    return count


def _check_tensor(value: object, label: str) -> int:
    if not isinstance(value, list) or not value:
        raise ParityError(f"{label} must be a non-empty batch.")
    for item in value:
        _check_numbers(item, label)
    return len(value)


def _check_numbers(value: object, label: str) -> None:
    if isinstance(value, list):
        if not value:
            raise ParityError(f"{label} has an empty dimension.")
        for item in value:
            _check_numbers(item, label)
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityError(f"{label} must be numbers.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ParityError(f"{label} must be finite.")


def _batch_count(value: object) -> int:
    if not isinstance(value, list):
        raise ParityError("Parity set samples must be batches.")
    return len(value)


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tar.gz") or name.endswith(".tgz")


def _pack(directory: Path, archive: Path) -> None:
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            tar.add(path, arcname=path.relative_to(directory).as_posix())


def _tar_bytes(handle: tarfile.TarFile, name: str) -> bytes:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ParityError(f"Parity set archive path is not relative: {name}")
    try:
        member = handle.getmember(name)
    except KeyError as exc:
        raise ParityError(f"Parity set archive is missing {name}.") from exc
    if not member.isfile():
        raise ParityError(f"Parity set archive member {name} is not a file.")
    extracted = handle.extractfile(member)
    if extracted is None:
        raise ParityError(f"Parity set archive is missing {name}.")
    return extracted.read()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
