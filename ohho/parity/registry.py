"""Local parity-set registry and the model-version attachment hook.

Sets live under ``OHHO_PARITY_SETS`` or ``~/.ohho/parity-sets``. An id such as
``act-golden`` or ``act-golden@1`` resolves to a directory written by
:func:`attach_parity_set`.

:class:`ParitySetPublisher` is the upload hook. The default publisher returns a
``local://`` URI and does not upload. Console storage subclasses ``publish``
and returns the URI it stored. That upload is not implemented here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ohho.parity.errors import ParityError
from ohho.parity.sets import ParitySet, load_parity_set

INDEX_SCHEMA_ID = "ohho.parity_set_index/v1"
ATTACHMENT_SCHEMA_ID = "ohho.parity_set_attachment/v1"
_REF = re.compile(r"^([a-z][a-z0-9]*(?:-[a-z0-9]+)*)(?:@v?([1-9][0-9]*))?$")


@dataclass(frozen=True)
class ParitySetAttachment:
    """A captured set bound to one registry model version.

    ``path`` is the local set directory. ``manifest`` is the
    ``ohho.parity_set/v1`` document (samples stay in ``samples.json``).
    """

    model: str
    model_version: str
    set_id: str
    set_version: int
    sha256: str
    path: str
    manifest: dict[str, Any]

    def to_json(self, *, registry_uri: str | None = None) -> dict[str, Any]:
        """JSON record of the attachment. ``registry_uri`` is the publisher result."""
        return {
            "schema": ATTACHMENT_SCHEMA_ID,
            "model": self.model,
            "model_version": self.model_version,
            "set_id": self.set_id,
            "set_version": self.set_version,
            "sha256": self.sha256,
            "path": self.path,
            "manifest": self.manifest,
            "registry_uri": registry_uri,
        }


class ParitySetPublisher:
    """Uploads a parity set for a registry model version.

    Console storage overrides :meth:`publish`. The default records nothing on
    the network and returns ``local://<set id>@<version>``.
    """

    def publish(self, attachment: ParitySetAttachment) -> str:
        """Return the registry URI for ``attachment``. Do not upload by default."""
        return f"local://{attachment.set_id}@{attachment.set_version}"


def default_registry_root() -> Path:
    """``OHHO_PARITY_SETS`` when set, otherwise ``~/.ohho/parity-sets``."""
    override = os.environ.get("OHHO_PARITY_SETS", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ohho" / "parity-sets"


def resolve_parity_set(ref: str, *, root: str | Path | None = None) -> ParitySet:
    """Load ``ref`` from a path or from the local registry.

    A value that exists on disk, or that contains a path separator, is a
    filesystem set (directory, ``manifest.json``, or ``.tar.gz``). Anything
    else is a registry id, optionally ``id@1`` or ``id@v1``. The highest
    stored version is used when the version is omitted.
    """
    text = ref.strip()
    if not text:
        raise ParityError("--parity-set is required.")
    if _looks_like_path(text):
        path = Path(text).expanduser()
        if not path.exists():
            raise ParityError(f"Parity set does not exist: {path}")
        return load_parity_set(path)
    return _load_registered(text, _root(root))


def attach_parity_set(
    set_path: str | Path,
    *,
    model_ref: str,
    model_version: str = "current",
    publisher: ParitySetPublisher | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Copy a set into the local registry and attach it to a model version.

    ``publisher`` runs after the local copy is written. The default publisher
    does not upload. A console backend passes its own publisher and receives
    the local directory plus the manifest. The returned document includes
    ``registry_uri`` from the publisher.
    """
    model = _require_name(model_ref, "--model")
    version_name = _require_name(model_version, "--model-version")
    loaded = load_parity_set(set_path)
    registry = _root(root)
    destination = _install(Path(loaded.location), loaded, registry)
    index = _read_index(registry)
    versions = index["sets"].setdefault(loaded.set_id, {})
    versions[str(loaded.version)] = {
        "sha256": loaded.sha256,
        "model": loaded.model,
        "source": loaded.source,
    }
    by_model = index["attachments"].setdefault(model, {})
    by_model[version_name] = {
        "set_id": loaded.set_id,
        "set_version": loaded.version,
        "sha256": loaded.sha256,
    }
    _write_index(registry, index)
    attachment = ParitySetAttachment(
        model=model,
        model_version=version_name,
        set_id=loaded.set_id,
        set_version=loaded.version,
        sha256=loaded.sha256,
        path=str(destination),
        manifest=loaded.manifest_document(),
    )
    hook = publisher if publisher is not None else ParitySetPublisher()
    uri = hook.publish(attachment)
    if not isinstance(uri, str) or not uri.strip():
        raise ParityError("Parity set publisher must return a registry URI.")
    return attachment.to_json(registry_uri=uri)


def list_parity_sets(root: str | Path | None = None) -> list[str]:
    """Registered set ids in sorted order. An empty registry returns an empty list."""
    registry = _root(root)
    if not (registry / "index.json").is_file():
        return []
    index = _read_index(registry)
    return sorted(index["sets"])


def _load_registered(ref: str, registry: Path) -> ParitySet:
    match = _REF.fullmatch(ref)
    if match is None:
        raise ParityError(
            f"Unknown parity set {ref!r}. Give a directory, an archive, or a registry id."
        )
    set_id, version_text = match.group(1), match.group(2)
    index = _read_index(registry) if (registry / "index.json").is_file() else None
    versions = {} if index is None else index["sets"].get(set_id, {})
    if not versions:
        known = ", ".join(list_parity_sets(registry)) or "(none)"
        raise ParityError(
            f"Unknown parity set {ref!r}. Give a directory, an archive, or a registry id. "
            f"Registry: {known}."
        )
    if version_text is None:
        chosen = str(max(int(item) for item in versions))
    else:
        chosen = str(int(version_text))
        if chosen not in versions:
            raise ParityError(
                f"Parity set {set_id} has no version {chosen}. "
                f"Versions: {', '.join(sorted(versions, key=int))}."
            )
    directory = registry / "sets" / set_id / chosen
    if not directory.is_dir():
        raise ParityError(f"Parity set {set_id}@{chosen} is not in the registry.")
    loaded = load_parity_set(directory)
    recorded = versions[chosen].get("sha256")
    if recorded != loaded.sha256:
        raise ParityError(
            f"Parity set {set_id}@{chosen} does not match the registry index."
        )
    return loaded


def _install(source: Path, loaded: ParitySet, registry: Path) -> Path:
    destination = registry / "sets" / loaded.set_id / str(loaded.version)
    if destination.exists():
        existing = load_parity_set(destination)
        if existing.sha256 != loaded.sha256:
            raise ParityError(
                f"{loaded.set_id}@{loaded.version} is already registered "
                "with a different sha256."
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
        return destination
    if source.is_file() and source.name == "manifest.json":
        shutil.copytree(source.parent, destination)
        return destination
    _extract_archive(source, destination)
    confirmed = load_parity_set(destination)
    if confirmed.sha256 != loaded.sha256:
        raise ParityError("Installed parity set does not match the archive sha256.")
    return destination


def _extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    try:
        handle = tarfile.open(archive, "r:gz")
    except (tarfile.TarError, OSError) as exc:
        raise ParityError(
            f"Could not read parity set archive {archive}: {exc}"
        ) from exc
    with handle:
        for member in handle.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ParityError(f"Parity set archive is missing {member.name}.")
            target.write_bytes(extracted.read())


def _read_index(registry: Path) -> dict[str, Any]:
    path = registry / "index.json"
    if not path.is_file():
        return {"schema": INDEX_SCHEMA_ID, "sets": {}, "attachments": {}}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParityError(f"Could not read parity set index: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != INDEX_SCHEMA_ID:
        raise ParityError("Parity set index schema is not ohho.parity_set_index/v1.")
    sets = document.get("sets")
    attachments = document.get("attachments")
    if not isinstance(sets, dict) or not isinstance(attachments, dict):
        raise ParityError("Parity set index is missing sets or attachments.")
    return {"schema": INDEX_SCHEMA_ID, "sets": sets, "attachments": attachments}


def _write_index(registry: Path, document: dict[str, Any]) -> None:
    registry.mkdir(parents=True, exist_ok=True)
    path = registry / "index.json"
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _root(explicit: str | Path | None) -> Path:
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser()
    return default_registry_root()


def _require_name(value: str, flag: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(c in value for c in "/\\")
    ):
        raise ParityError(f"{flag} must be a name without a path separator.")
    return value.strip()


def _looks_like_path(text: str) -> bool:
    if any(sep in text for sep in ("/", "\\")):
        return True
    if text.startswith("."):
        return True
    return len(text) >= 2 and text[1] == ":"
