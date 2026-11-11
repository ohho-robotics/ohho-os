"""Import a LeRobot dataset from the Hugging Face Hub as a dataset version.

The import pins two facts on ``imports/hf/<repo hash>/<commit>.json``:

* ``revision`` is the commit the Hub resolved, never the branch name that
  was requested
* ``licence`` is the dataset-card licence at that commit

Both stay as they were written. A later import of the same commit that
reports a different licence is refused. Non-commercial licences are stored
and flagged; the import still completes.

``huggingface_hub`` is imported only when the default Hub client runs.
Tests pass a :class:`DatasetHub` and do not open a network connection.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .build import _augment, _load, _robot_type
from .lineage import record_version_lineage
from .model import (
    Augmentation,
    CanonicalEpisode,
    DatasetSpec,
    DatasetVersion,
    DatasetVersionError,
    DatasetVersionNotFound,
    SourceQuery,
    canonical_content_bytes,
    content_hash_of,
    is_version_id,
    stats_from_episodes,
    storage_prefix_for,
    version_id_for,
)
from .store import create_version, load_version, resolve_store, verify_content_hash

IMPORT_SCHEMA = "ohho.dataset_import.v1"
_REPO_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$"
)
_REVISION_RE = re.compile(r"[A-Za-z0-9._/-]+")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_TIME_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_NC_RE = re.compile(r"(^|[^a-z0-9])cc-by-nc(-sa|-nd)?([^a-z0-9]|$)")
_NON_COMMERCIAL_RE = re.compile(r"non-?commercial")
_NEUTRAL_LICENCES = {"unspecified", "unknown", "other", "none"}


class DatasetImportError(DatasetVersionError):
    """A Hugging Face dataset could not be imported."""


@dataclass(frozen=True)
class HubSnapshot:
    """A resolved Hub dataset sitting in a local LeRobot directory."""

    repo_id: str
    revision: str
    licence: str
    root: Path


@dataclass(frozen=True)
class DatasetImport:
    """One pinned Hub import. ``revision`` and ``licence`` do not change."""

    repo_id: str
    revision: str
    requested_revision: str | None
    licence: str
    non_commercial: bool
    version_id: str
    created_at: str
    schema: str = IMPORT_SCHEMA

    @property
    def warning(self) -> str | None:
        """Warning text when the pinned licence is non-commercial."""
        if self.non_commercial:
            return f"licence {self.licence} is non-commercial"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "requested_revision": self.requested_revision,
            "licence": self.licence,
            "non_commercial": self.non_commercial,
            "revision_pinned": True,
            "licence_pinned": True,
            "version_id": self.version_id,
            "created_at": self.created_at,
        }


class DatasetHub(Protocol):
    """Fetches one dataset revision. The default client talks to the Hub."""

    def fetch(self, repo_id: str, revision: str | None) -> HubSnapshot: ...


def is_non_commercial(licence: str) -> bool:
    """True for Creative Commons NonCommercial ids and "non-commercial" text.

    Apache, MIT, BSD, CC BY, CC BY-SA, CC BY-ND, and ``unspecified`` are not
    flagged. Comparison is case-insensitive.
    """
    text = licence.strip().lower().replace("_", "-")
    text = re.sub(r"\s+", "-", text)
    if not text or text in _NEUTRAL_LICENCES:
        return False
    if _NON_COMMERCIAL_RE.search(text):
        return True
    return _NC_RE.search(text) is not None


def resolved_licence(card_data: Any, readme_text: str | None) -> str:
    """Licence from a dataset card, then README front matter, else unspecified."""
    from_card = _licence_from_card(card_data)
    if from_card:
        return _clip_licence(from_card)
    if readme_text:
        from_readme = _licence_from_readme(readme_text)
        if from_readme:
            return _clip_licence(from_readme)
    return "unspecified"


def import_hf_dataset(
    repo_id: str,
    *,
    revision: str | None = None,
    store: str | Path | None = None,
    hub: DatasetHub | None = None,
) -> DatasetImport:
    """Import ``repo_id`` as a pinned dataset version.

    ``revision`` may be a branch, tag, or commit. The stored revision is the
    commit the Hub returns. ``hub`` defaults to :class:`HuggingFaceDatasetHub`.
    """
    repo = _check_repo_id(repo_id)
    requested = _check_revision(revision)
    client = hub if hub is not None else default_dataset_hub(store)
    snapshot = client.fetch(repo, requested)
    sha = _check_snapshot(snapshot, repo)
    licence = _normalize_licence(snapshot.licence)
    root = Path(snapshot.root).expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        raise DatasetImportError(
            f"Hugging Face dataset {repo}@{sha} has no meta/info.json"
        )
    loaded = _load((str(root),))
    canonical = _augment(loaded, Augmentation(), 0)
    if len(canonical) != len(loaded):
        raise DatasetImportError("imported episode count does not match the snapshot")
    content_hash = content_hash_of(canonical)
    spec = DatasetSpec(
        source_query=SourceQuery((f"hf://{repo}@{sha}",)),
        dataset_name=repo.rsplit("/", 1)[-1],
        licence=licence,
    )
    version_id = version_id_for(spec, content_hash)
    existing = _read_import(store, repo, sha)
    if existing is not None:
        if existing.licence != licence:
            raise DatasetImportError(
                f"pinned licence for {repo}@{sha} is {existing.licence}"
            )
        if existing.version_id != version_id:
            raise DatasetImportError(
                f"pinned dataset version for {repo}@{sha} does not match the snapshot"
            )
        _ensure_version(
            spec,
            canonical,
            content_hash,
            loaded,
            repo,
            store=store,
        )
        record_version_lineage(
            version_id,
            canonical,
            source=spec.source_query.roots[0],
            source_indexes=[item.source_index for item in loaded],
            store=store,
        )
        return existing
    _ensure_version(
        spec,
        canonical,
        content_hash,
        loaded,
        repo,
        store=store,
    )
    record_version_lineage(
        version_id,
        canonical,
        source=spec.source_query.roots[0],
        source_indexes=[item.source_index for item in loaded],
        store=store,
    )
    record = DatasetImport(
        repo_id=repo,
        revision=sha,
        requested_revision=requested,
        licence=licence,
        non_commercial=is_non_commercial(licence),
        version_id=version_id,
        created_at=_now(),
    )
    _write_new_import(store, record)
    stored = _read_import(store, repo, sha)
    if stored is None:
        raise DatasetImportError(f"import record for {repo}@{sha} was not written")
    if stored.licence != licence or stored.version_id != version_id:
        raise DatasetImportError(
            f"pinned import of {repo}@{sha} does not match this snapshot"
        )
    return stored


def default_dataset_hub(store: str | Path | None = None) -> HuggingFaceDatasetHub:
    """Hub client whose download cache lives under the dataset store."""
    return HuggingFaceDatasetHub(resolve_store(store) / ".cache" / "hf")


class HuggingFaceDatasetHub:
    """Resolve and download a dataset with ``huggingface_hub``.

    Constructing this class does not import ``huggingface_hub``. The import
    happens on :meth:`fetch`.
    """

    def __init__(self, cache: str | Path) -> None:
        self.cache = Path(cache)

    def fetch(self, repo_id: str, revision: str | None) -> HubSnapshot:
        repo = _check_repo_id(repo_id)
        requested = _check_revision(revision)
        api_factory, snapshot_download = _load_huggingface_hub()
        api = api_factory()
        info = _dataset_info(api, repo, requested)
        sha = str(getattr(info, "sha", "") or "")
        if _SHA_RE.fullmatch(sha) is None:
            raise DatasetImportError(
                f"Hugging Face did not resolve {repo} to a commit revision"
            )
        dest = self.cache / _repo_dir(repo) / sha
        if not (dest / "meta" / "info.json").is_file():
            _download_snapshot(snapshot_download, repo, sha, dest)
        readme = dest / "README.md"
        readme_text = _read_text(readme)
        licence = resolved_licence(getattr(info, "card_data", None), readme_text)
        return HubSnapshot(repo_id=repo, revision=sha, licence=licence, root=dest)


def _load_huggingface_hub() -> tuple[Any, Any]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise DatasetImportError(
            "Hugging Face import needs huggingface_hub "
            "(pip install 'ohho-os[datasets]')"
        ) from exc
    return HfApi, snapshot_download


def _dataset_info(api: Any, repo_id: str, revision: str | None) -> Any:
    kwargs: dict[str, Any] = {"repo_id": repo_id}
    if revision is not None:
        kwargs["revision"] = revision
    try:
        return api.dataset_info(**kwargs)
    except Exception as exc:
        raise DatasetImportError(
            f"could not resolve Hugging Face dataset {repo_id}"
        ) from exc


def _download_snapshot(
    snapshot_download: Any, repo_id: str, sha: str, dest: Path
) -> None:
    staging = dest.with_name(sha + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _call_snapshot(snapshot_download, repo_id, sha, staging)
    except DatasetImportError:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise DatasetImportError(
            f"could not download Hugging Face dataset {repo_id}@{sha}"
        ) from exc
    if not (staging / "meta" / "info.json").is_file():
        shutil.rmtree(staging, ignore_errors=True)
        raise DatasetImportError(
            f"Hugging Face dataset {repo_id}@{sha} has no meta/info.json"
        )
    if dest.exists():
        shutil.rmtree(dest)
    staging.replace(dest)


def _call_snapshot(snapshot_download: Any, repo_id: str, sha: str, dest: Path) -> None:
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": sha,
        "local_dir": str(dest),
    }
    try:
        snapshot_download(**kwargs, local_dir_use_symlinks=False)
    except TypeError:
        snapshot_download(**kwargs)


def _ensure_version(
    spec: DatasetSpec,
    canonical: Sequence[CanonicalEpisode],
    content_hash: str,
    loaded: Sequence[Any],
    repo_id: str,
    *,
    store: str | Path | None,
) -> DatasetVersion:
    version_id = version_id_for(spec, content_hash)
    root = resolve_store(store)
    try:
        current = load_version(version_id, store=root)
    except DatasetVersionNotFound:
        current = None
    except DatasetVersionError as exc:
        raise DatasetImportError(str(exc)) from exc
    if current is not None:
        _check_pinned_version(current, spec, content_hash)
        try:
            verify_content_hash(current, store=root)
        except DatasetVersionError as exc:
            raise DatasetImportError(str(exc)) from exc
        return current
    robot_type = _robot_type(loaded)
    record = DatasetVersion(
        version_id=version_id,
        content_hash=content_hash,
        source_query=spec.source_query,
        filters=spec.filters,
        split=spec.split,
        seed=spec.seed,
        augmentation=spec.augmentation,
        stats=stats_from_episodes(canonical),
        licence=spec.licence,
        dataset_name=spec.dataset_name,
        pinned=True,
        parent_version_id=None,
        storage_prefix=storage_prefix_for(version_id),
        created_at=_now(),
    )

    def populate(dest: Path) -> None:
        _write_imported_dataset(dest, canonical, repo_id, robot_type)

    try:
        created = create_version(
            record,
            canonical_content_bytes(canonical),
            populate,
            store=root,
        )
    except DatasetVersionError as exc:
        raise DatasetImportError(str(exc)) from exc
    _check_pinned_version(created, spec, content_hash)
    return created


def _check_pinned_version(
    current: DatasetVersion, spec: DatasetSpec, content_hash: str
) -> None:
    if current.content_hash != content_hash or current.licence != spec.licence:
        raise DatasetImportError(
            f"pinned dataset version {current.version_id} does not match the snapshot"
        )
    if current.source_query.roots != spec.source_query.roots:
        raise DatasetImportError(
            f"pinned dataset version {current.version_id} cites a different source"
        )
    if not current.pinned:
        raise DatasetImportError(
            f"dataset version {current.version_id} was imported without a pin"
        )


def _write_imported_dataset(
    dest: Path,
    episodes: Sequence[CanonicalEpisode],
    repo_id: str,
    robot_type: str,
) -> None:
    """Write a LeRobot tree. The Hub repo id is stored on the dataset card."""
    from ..data.recorder import Episode, Frame
    from ..data.writer import write_dataset

    tasks: list[str] = []
    task_index: dict[str, int] = {}
    written: list[Episode] = []
    fps = float(episodes[0].fps)
    for episode_index, episode in enumerate(episodes):
        if episode.task not in task_index:
            task_index[episode.task] = len(tasks)
            tasks.append(episode.task)
        index = task_index[episode.task]
        frames = [
            Frame(
                observation_state=[float(value) for value in frame.observation_state],
                action=[float(value) for value in frame.action],
                action_raw=(
                    None
                    if frame.action_raw is None
                    else [float(value) for value in frame.action_raw]
                ),
                timestamp=float(frame.timestamp),
                frame_index=frame.frame_index,
                episode_index=episode_index,
                task_index=index,
                next_done=frame.next_done,
            )
            for frame in episode.frames
        ]
        written.append(
            Episode(
                task=episode.task,
                frames=frames,
                episode_index=episode_index,
                task_index=index,
                fps=fps,
            )
        )
    write_dataset(
        written,
        str(dest),
        repo_id=repo_id,
        fps=fps,
        tasks=tasks or ["demonstration"],
        robot_type=robot_type,
    )


def _check_snapshot(snapshot: HubSnapshot, repo_id: str) -> str:
    if not isinstance(snapshot, HubSnapshot):
        raise DatasetImportError("dataset hub returned an unexpected snapshot")
    if snapshot.repo_id != repo_id:
        raise DatasetImportError("dataset hub returned a different repo id")
    revision = snapshot.revision
    if not isinstance(revision, str) or _SHA_RE.fullmatch(revision) is None:
        raise DatasetImportError(
            f"dataset hub did not resolve {repo_id} to a commit revision"
        )
    return revision


def _check_repo_id(repo_id: str) -> str:
    if not isinstance(repo_id, str) or _REPO_RE.fullmatch(repo_id) is None:
        raise DatasetImportError(f"invalid Hugging Face dataset repo id: {repo_id}")
    return repo_id


def _check_revision(revision: str | None) -> str | None:
    if revision is None:
        return None
    if not isinstance(revision, str) or revision != revision.strip() or not revision:
        raise DatasetImportError("revision must be a Hugging Face revision")
    if (
        len(revision) > 128
        or ".." in revision
        or "\\" in revision
        or revision.startswith("/")
        or _REVISION_RE.fullmatch(revision) is None
    ):
        raise DatasetImportError("revision must be a Hugging Face revision")
    return revision


def _normalize_licence(licence: str) -> str:
    if not isinstance(licence, str):
        raise DatasetImportError("dataset licence must be a string")
    return _clip_licence(licence) if licence.strip() else "unspecified"


def _clip_licence(value: str) -> str:
    text = value.strip()
    if not text:
        return "unspecified"
    if len(text) > 512 or any(ord(char) < 32 for char in text):
        raise DatasetImportError("dataset licence is not a single-line string")
    return text


def _licence_from_card(card: Any) -> str | None:
    if card is None:
        return None
    value = None
    if isinstance(card, Mapping):
        value = card.get("license", card.get("licence"))
    else:
        value = getattr(card, "license", None)
        if value is None:
            value = getattr(card, "licence", None)
        data = getattr(card, "data", None)
        if value is None and isinstance(data, Mapping):
            value = data.get("license", data.get("licence"))
    return _first_licence(value)


def _first_licence(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found = _first_licence(item)
            if found:
                return found
    return None


def _licence_from_readme(text: str) -> str | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    found: dict[str, str] = {}
    pending: str | None = None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[:1] in {" ", "\t"}:
            item = line.strip()
            if pending and item.startswith("- ") and pending not in found:
                value = item[2:].strip().strip("'\"")
                if value:
                    found[pending] = value
            continue
        if ":" not in line:
            pending = None
            continue
        key, raw = line.split(":", 1)
        key = key.strip().lower()
        value = raw.strip().strip("'\"")
        pending = key
        if value and value not in {"|", ">", "|-", ">-"}:
            found[key] = value
    return found.get("license") or found.get("licence")


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _repo_dir(repo_id: str) -> str:
    return hashlib.sha256(repo_id.encode("utf-8")).hexdigest()


def _import_path(store: str | Path | None, repo_id: str, revision: str) -> Path:
    return (
        resolve_store(store)
        / "imports"
        / "hf"
        / _repo_dir(repo_id)
        / f"{revision}.json"
    )


def _read_import(
    store: str | Path | None, repo_id: str, revision: str
) -> DatasetImport | None:
    path = _import_path(store, repo_id, revision)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetImportError(
            f"invalid import record for {repo_id}@{revision}"
        ) from exc
    record = _import_from_dict(document)
    if record.repo_id != repo_id or record.revision != revision:
        raise DatasetImportError(f"import record does not match {repo_id}@{revision}")
    return record


def _write_new_import(store: str | Path | None, record: DatasetImport) -> None:
    path = _import_path(store, record.repo_id, record.revision)
    if path.is_file():
        return
    text = json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, text)


def _import_from_dict(document: Any) -> DatasetImport:
    if not isinstance(document, Mapping):
        raise DatasetImportError("import record must be an object")
    required = (
        "created_at",
        "licence",
        "licence_pinned",
        "non_commercial",
        "repo_id",
        "requested_revision",
        "revision",
        "revision_pinned",
        "schema",
        "version_id",
    )
    unknown = sorted(set(document) - set(required))
    if unknown:
        names = ", ".join(unknown)
        raise DatasetImportError(f"import record has unknown fields: {names}")
    for key in required:
        if key not in document:
            raise DatasetImportError(f"import record is missing {key}")
    if document["schema"] != IMPORT_SCHEMA:
        raise DatasetImportError("import record schema is not ohho.dataset_import.v1")
    if (
        document["revision_pinned"] is not True
        or document["licence_pinned"] is not True
    ):
        raise DatasetImportError("import revision and licence must stay pinned")
    repo_id = _check_repo_id(document["repo_id"])
    revision = document["revision"]
    if not isinstance(revision, str) or _SHA_RE.fullmatch(revision) is None:
        raise DatasetImportError("import revision is not a pinned commit")
    requested = document["requested_revision"]
    if requested is not None:
        requested = _check_revision(requested)
    licence = _normalize_licence(document["licence"])
    non_commercial = document["non_commercial"]
    if not isinstance(non_commercial, bool):
        raise DatasetImportError("import non_commercial flag must be a boolean")
    if non_commercial != is_non_commercial(licence):
        raise DatasetImportError(
            "non_commercial flag does not match the pinned licence"
        )
    version_id = document["version_id"]
    if not isinstance(version_id, str) or not is_version_id(version_id):
        raise DatasetImportError("import record is missing a dataset version id")
    created_at = document["created_at"]
    if not isinstance(created_at, str) or _TIME_RE.fullmatch(created_at) is None:
        raise DatasetImportError("import created_at is not a UTC timestamp")
    return DatasetImport(
        repo_id=repo_id,
        revision=revision,
        requested_revision=requested,
        licence=licence,
        non_commercial=non_commercial,
        version_id=version_id,
        created_at=created_at,
    )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
