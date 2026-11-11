"""Local dataset-version store.

Layout (the store root is ``OHHO_DATASETS_ROOT`` or ``~/.ohho/datasets``)::

    <root>/versions/<version_id>/version.json
    <root>/versions/<version_id>/dataset/   # LeRobot tree

``version.json`` is one future ``dataset_versions`` row. The ``dataset``
directory is the object-storage prefix ``versions/<version_id>/dataset``.
Pinning is the only in-place write. A pinned version cannot be modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .model import (
    DatasetStats,
    DatasetVersion,
    DatasetVersionError,
    DatasetVersionFrozen,
    DatasetVersionNotFound,
    assert_identity,
    is_version_id,
    record_to_json,
    version_from_dict,
)


def resolve_store(store: str | Path | None = None) -> Path:
    """Absolute store root.

    An explicit ``store`` wins, then ``OHHO_DATASETS_ROOT``, then
    ``~/.ohho/datasets``. The directory is not created here.
    """
    if store is not None and str(store).strip():
        return Path(store).expanduser().resolve()
    env = os.environ.get("OHHO_DATASETS_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".ohho" / "datasets").resolve()


def version_record_path(root: Path, version_id: str) -> Path:
    """Path to ``version.json`` for ``version_id`` under ``root``."""
    if not is_version_id(version_id):
        raise DatasetVersionError(f"invalid dataset version id: {version_id}")
    return root / "versions" / version_id / "version.json"


def dataset_directory(root: Path, version_id: str) -> Path:
    """Materialised LeRobot directory for ``version_id``."""
    return version_record_path(root, version_id).parent / "dataset"


def read_version_file(path: Path) -> DatasetVersion:
    """Load and check one ``version.json`` file."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetVersionError(
            f"invalid version record: {path.parent.name}"
        ) from exc
    if not isinstance(document, dict):
        raise DatasetVersionError(f"invalid version record: {path.parent.name}")
    return version_from_dict(document)


def load_version(version_id: str, *, store: str | Path | None = None) -> DatasetVersion:
    """Load a dataset version from the local store."""
    path = version_record_path(resolve_store(store), version_id)
    if not path.is_file():
        raise DatasetVersionNotFound(f"dataset version {version_id} was not found")
    return read_version_file(path)


def list_versions(*, store: str | Path | None = None) -> list[DatasetVersion]:
    """Every version record in the store, ordered by version id."""
    versions = resolve_store(store) / "versions"
    if not versions.is_dir():
        return []
    found: list[DatasetVersion] = []
    for child in sorted(versions.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or not is_version_id(child.name):
            continue
        record_path = child / "version.json"
        if record_path.is_file():
            found.append(read_version_file(record_path))
    return found


def verify_content_hash(
    record: DatasetVersion, *, store: str | Path | None = None
) -> None:
    """Raise when ``meta/ohho_content.json`` does not match ``content_hash``."""
    path = dataset_directory(resolve_store(store), record.version_id)
    content = path / "meta" / "ohho_content.json"
    if not content.is_file():
        raise DatasetVersionError(
            f"dataset version {record.version_id} is missing its canonical content"
        )
    digest = "sha256:" + hashlib.sha256(content.read_bytes()).hexdigest()
    if digest != record.content_hash:
        raise DatasetVersionError(
            f"dataset version {record.version_id} content hash does not match"
        )


def create_version(
    record: DatasetVersion,
    content: bytes,
    populate: Callable[[Path], None],
    *,
    store: str | Path | None = None,
) -> DatasetVersion:
    """Create a version directory, or return the existing record unchanged.

    ``populate`` writes the LeRobot tree into the dataset directory. The
    canonical content bytes are stored beside it and must hash to
    ``record.content_hash``. An existing version is never rewritten.
    """
    root = resolve_store(store)
    path = version_record_path(root, record.version_id)
    if path.is_file():
        existing = read_version_file(path)
        if existing.content_hash != record.content_hash:
            raise DatasetVersionError(
                f"dataset version {record.version_id} collided with different content"
            )
        verify_content_hash(existing, store=root)
        return existing
    version_dir = path.parent
    if version_dir.exists():
        shutil.rmtree(version_dir)
    dataset_dir = version_dir / "dataset"
    try:
        populate(dataset_dir)
        content_path = dataset_dir / "meta" / "ohho_content.json"
        content_path.parent.mkdir(parents=True, exist_ok=True)
        content_path.write_bytes(content)
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        if digest != record.content_hash:
            raise DatasetVersionError(
                "content hash does not match the canonical payload"
            )
        _atomic_write_text(path, record_to_json(record))
    except Exception:
        if version_dir.exists():
            shutil.rmtree(version_dir, ignore_errors=True)
        raise
    return read_version_file(path)


def pin_version(version_id: str, *, store: str | Path | None = None) -> DatasetVersion:
    """Freeze a version. A second pin returns the same record and does not write."""
    root = resolve_store(store)
    path = version_record_path(root, version_id)
    if not path.is_file():
        raise DatasetVersionNotFound(f"dataset version {version_id} was not found")
    current = read_version_file(path)
    verify_content_hash(current, store=root)
    if current.pinned:
        return current
    updated = replace(current, pinned=True)
    _atomic_write_text(path, record_to_json(updated))
    return read_version_file(path)


def overwrite_version(
    record: DatasetVersion, *, store: str | Path | None = None
) -> None:
    """Refuse to replace a stored version in place.

    Pinned versions raise :class:`DatasetVersionFrozen`. Any other existing
    version raises :class:`DatasetVersionError`: spec edits allocate a new
    version id via :func:`ohho.datasets.build.edit_version`.
    """
    assert_identity(record)
    path = version_record_path(resolve_store(store), record.version_id)
    if not path.is_file():
        raise DatasetVersionNotFound(
            f"dataset version {record.version_id} was not found"
        )
    current = read_version_file(path)
    if current.pinned:
        raise DatasetVersionFrozen(
            f"dataset version {record.version_id} is pinned and cannot be modified"
        )
    raise DatasetVersionError(
        f"dataset version {record.version_id} already exists; edits create a new version"
    )


def version_stats(version_id: str, *, store: str | Path | None = None) -> DatasetStats:
    """Stats stored on the version record. This does not recompute or write."""
    return load_version(version_id, store=store).stats


def resolve_dataset_version(ref: str, *, store: str | Path | None = None) -> Path:
    """Resolve a dataset version id or a dataset path to a local LeRobot root.

    Training recipes take a filesystem path. Call this first so the same
    argument can be a version id (``dv_…``) or an existing dataset directory::

        root = resolve_dataset_version(version_or_path)
        finetune(str(root), policy="act", mock=True)

    This function does not import the trainer. A version id is checked
    against ``meta/ohho_content.json`` before the dataset path is returned.
    """
    text = ref.strip()
    if is_version_id(text):
        record = load_version(text, store=store)
        verify_content_hash(record, store=store)
        dataset = dataset_directory(resolve_store(store), text)
        if not (dataset / "meta" / "info.json").is_file():
            raise DatasetVersionNotFound(
                f"dataset version {text} has no materialised dataset"
            )
        return dataset
    path = Path(text).expanduser()
    if (path / "meta" / "info.json").is_file():
        return path.resolve()
    nested = path / "dataset" / "meta" / "info.json"
    if nested.is_file() and (path / "version.json").is_file():
        return (path / "dataset").resolve()
    raise DatasetVersionNotFound(f"no dataset version or local dataset at {ref}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
