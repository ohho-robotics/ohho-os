"""Local build cache for ``ohho optimize``.

The cache key is the sha256 of the model bytes (or the registry id), the
target profile, the precision, the applied flags, and the toolchain versions.
A hit copies the previous artifacts and skips export, calibration, and the
engine build. ``--no-cache`` skips the lookup and writes a new entry.

``RegistryCache`` is the hook for a console registry. This package does not
talk to one. Pass an object with ``find`` and ``publish`` to
:func:`ohho.optimize.engine.optimize` and it is consulted after a local miss
and offered the build after a local store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Protocol

from ohho.optimize.calibrator import sha256_path
from ohho.optimize.errors import OptimizeError
from ohho.optimize.manifest import manifest_problems, write_manifest

logger = logging.getLogger("ohho.optimize")


def default_cache_dir() -> Path:
    """``OHHO_OPTIMIZE_CACHE`` when set, otherwise ``~/.ohho/optimize-cache``."""
    override = os.environ.get("OHHO_OPTIMIZE_CACHE", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".ohho" / "optimize-cache"


def source_model_sha256(model: str) -> str:
    """sha256 of a model path, or of the registry id when that path does not exist."""
    text = model.strip()
    path = Path(text)
    if path.exists():
        return sha256_path(path)
    digest = hashlib.sha256()
    digest.update(b"ohho-registry\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def cache_key(
    *,
    model_sha256: str,
    target: dict[str, Any],
    precision: str,
    flags: dict[str, Any],
    versions: dict[str, Any],
) -> str:
    """sha256 of the model, target, precision, flags, and toolchain versions.

    Precision sits beside the flag object so two precisions that share every
    other flag still miss. ``versions`` is the toolchain recorded on the
    manifest, including TensorRT.
    """
    payload = {
        "model_sha256": model_sha256,
        "target": {
            "id": target.get("id"),
            "arch": target.get("arch"),
            "accelerator": target.get("accelerator"),
        },
        "precision": precision,
        "flags": flags,
        "versions": versions,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str | None:
    """sha256 of a file, or null when the path is not a file."""
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class RegistryCache(Protocol):
    """Lookup hook for a future console registry.

    The local directory cache is the only cache this package implements.
    A registry client is asked after a local miss and receives the finished
    build directory after a local store. Shipping that client is a separate
    change.
    """

    def find(self, key: str) -> str | None:
        """Return a directory that holds ``manifest.json``, or ``None``."""

    def publish(self, key: str, directory: str) -> None:
        """Record one finished build under ``key``."""


class LocalBuildCache:
    """One directory of builds, addressed by :func:`cache_key`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, key: str) -> Path:
        """Cache directory for ``key``. The key must already be 64 hex characters."""
        _check_key(key)
        return self.root / key

    def find(self, key: str) -> Path | None:
        """Return the cached directory when it contains ``manifest.json``."""
        directory = self.path_for(key)
        if (directory / "manifest.json").is_file():
            return directory
        return None

    def store(self, key: str, source: Path) -> Path:
        """Copy ``source`` over any previous entry for ``key``."""
        dest = self.path_for(key)
        staging = self.root / f".{key}.partial"
        if staging.exists():
            shutil.rmtree(staging)
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, staging)
        if dest.exists():
            shutil.rmtree(dest)
        staging.rename(dest)
        return dest


def read_cached_manifest(directory: Path) -> dict[str, Any] | None:
    """Return a schema-valid cached manifest, or ``None`` when it cannot be reused."""
    path = directory / "manifest.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict) or manifest_problems(document):
        return None
    return document


def restore_build(source: Path, out: Path, *, cache_source: str) -> dict[str, Any]:
    """Copy a cached build into ``out`` and mark the manifest as a hit.

    The stored copy keeps ``cache.hit`` false so the next lookup still
    describes the build that produced the bytes. The copy in ``out`` records
    the hit.
    """
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, out)
    document = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise OptimizeError(f"Cached manifest in {source} is not an object.")
    cache = document.get("cache")
    if not isinstance(cache, dict) or not isinstance(cache.get("key"), str):
        raise OptimizeError(f"Cached manifest in {source} has no cache key.")
    document["cache"] = {
        "key": cache["key"],
        "hit": True,
        "source": cache_source,
    }
    write_manifest(out, document)
    return document


def _check_key(key: str) -> None:
    if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
        raise OptimizeError("Optimize cache key must be 64 lowercase hex characters.")
