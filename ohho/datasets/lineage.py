"""Lineage edges for dataset versions, runs, and model versions.

The local store is one JSON document. It does not use SQLite. A console
database implements :class:`LineageStore` with the tables in
``docs/dataset-lineage.md``.

Edges:

* dataset version → the episodes that version materialised
* run → one dataset version (the run id is a reference, not a manifest)
* model version → the run that produced it

``lineage_from_model`` walks model version → run → dataset version →
episodes. ``models_trained_on`` walks the other way: one episode to every
model version trained on it.

Deleting an episode that a pinned dataset version still cites is refused as
a hard delete. The episode stays on those edges, ``deleted`` is set, and the
result carries a warning. A missing version record is treated as pinned.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .model import (
    CanonicalEpisode,
    DatasetVersionError,
    is_version_id,
)
from .store import load_version, resolve_store

SCHEMA_NAME = "ohho.lineage.v1"
EPISODE_ID_ALGORITHM = "ohho.episode_id.v1"
_EPISODE_ID_RE_TEXT = r"^ep_[0-9a-f]{32}$"
_CONTENT_HASH_RE_TEXT = r"^sha256:[0-9a-f]{64}$"
_REF_ID_RE_TEXT = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_TIME_RE_TEXT = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"

_EPISODE_ID_RE = re.compile(_EPISODE_ID_RE_TEXT)
_CONTENT_HASH_RE = re.compile(_CONTENT_HASH_RE_TEXT)
_REF_ID_RE = re.compile(_REF_ID_RE_TEXT)
_TIME_RE = re.compile(_TIME_RE_TEXT)

_GRAPH_KEYS = ("episodes", "models", "runs", "schema", "version_episodes")
_EPISODE_KEYS = (
    "content_hash",
    "delete_warning",
    "deleted",
    "deleted_at",
    "episode_id",
    "fps",
    "frame_count",
    "source",
    "source_index",
    "task",
)
_RUN_KEYS = ("dataset_version_id", "run_id")
_MODEL_KEYS = ("model_version_id", "run_id")


class LineageError(DatasetVersionError):
    """A lineage record or edge is invalid."""


class LineageNotFound(LineageError):
    """The lineage graph has no such episode, run, or model version."""


class LineageConflict(LineageError):
    """The edge contradicts an edge that is already stored."""


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode row in the lineage catalog.

    ``episode_id`` is ``ep_`` plus the first 32 hex chars of ``content_hash``.
    The hash is the canonical episode payload, so the same frames share an id
    wherever they were copied.
    """

    episode_id: str
    content_hash: str
    task: str
    fps: str
    frame_count: int
    source: str
    source_index: int
    deleted: bool = False
    deleted_at: str | None = None
    delete_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "content_hash": self.content_hash,
            "task": self.task,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "source": self.source,
            "source_index": self.source_index,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "delete_warning": self.delete_warning,
        }


@dataclass(frozen=True)
class ModelLineage:
    """Model version → run → dataset version → episodes."""

    model_version_id: str
    run_id: str
    dataset_version_id: str
    episode_ids: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeTraining:
    """One model version trained on an episode, through a run and a version."""

    episode_id: str
    dataset_version_id: str
    run_id: str
    model_version_id: str


@dataclass(frozen=True)
class EpisodeDeletion:
    """Result of :meth:`LineageStore.delete_episode`.

    ``blocked`` means a pinned dataset version still cites the episode, so
    the row was soft-deleted instead of removed. ``warning`` is set in that
    case. ``removed`` means the row and its version edges are gone.
    """

    episode_id: str
    removed: bool
    soft_deleted: bool
    blocked: bool
    warning: str | None
    pinned_version_ids: tuple[str, ...]


class LineageStore(Protocol):
    """Edges the local JSON file and the console database both implement.

    Run rows store a run id and a dataset version id. They are not a
    training manifest.
    """

    def put_episode(self, episode: EpisodeRecord) -> EpisodeRecord: ...

    def episode(self, episode_id: str) -> EpisodeRecord: ...

    def link_version_episodes(
        self, version_id: str, episode_ids: Sequence[str]
    ) -> None: ...

    def link_run(self, run_id: str, dataset_version_id: str) -> None: ...

    def link_model(self, model_version_id: str, run_id: str) -> None: ...

    def episodes_for_version(self, version_id: str) -> tuple[str, ...]: ...

    def versions_for_episode(self, episode_id: str) -> tuple[str, ...]: ...

    def lineage_from_model(self, model_version_id: str) -> ModelLineage: ...

    def models_trained_on(self, episode_id: str) -> tuple[EpisodeTraining, ...]: ...

    def delete_episode(self, episode_id: str) -> EpisodeDeletion: ...


def episode_from_canonical(
    episode: CanonicalEpisode,
    *,
    source: str,
    source_index: int,
) -> EpisodeRecord:
    """Build an episode row whose id is the hash of the canonical payload."""
    raw = _episode_bytes(episode)
    digest = hashlib.sha256(raw).hexdigest()
    task = " ".join(episode.task.splitlines())
    return EpisodeRecord(
        episode_id="ep_" + digest[:32],
        content_hash="sha256:" + digest,
        task=task,
        fps=episode.fps,
        frame_count=len(episode.frames),
        source=source,
        source_index=source_index,
    )


def record_version_lineage(
    version_id: str,
    episodes: Sequence[CanonicalEpisode],
    *,
    source: str,
    source_indexes: Sequence[int],
    store: str | Path | None = None,
) -> tuple[str, ...]:
    """Register ``episodes`` and link them, in order, to ``version_id``.

    Identical canonical episodes share one id. The first source index is kept.
    Linking the same list again does not rewrite the graph.
    """
    if len(episodes) != len(source_indexes):
        raise LineageError("episode index count does not match episodes")
    if not episodes:
        raise LineageError("a dataset version needs at least one episode")
    graph = JsonLineageStore(store)
    ids: list[str] = []
    seen: set[str] = set()
    for episode, index in zip(episodes, source_indexes):
        record = episode_from_canonical(episode, source=source, source_index=index)
        graph.put_episode(record)
        if record.episode_id not in seen:
            seen.add(record.episode_id)
            ids.append(record.episode_id)
    graph.link_version_episodes(version_id, ids)
    return tuple(ids)


class JsonLineageStore:
    """:class:`LineageStore` backed by ``<store>/lineage/graph.json``.

    The store root matches dataset versions (``--store``,
    ``OHHO_DATASETS_ROOT``, or ``~/.ohho/datasets``). Writes are atomic
    replacements of that one file. This process is a single writer; the
    console database is the concurrent implementation.
    """

    def __init__(self, store: str | Path | None = None) -> None:
        self.root = resolve_store(store)

    @property
    def path(self) -> Path:
        """Path of the graph document. It is created on the first write."""
        return self.root / "lineage" / "graph.json"

    def put_episode(self, episode: EpisodeRecord) -> EpisodeRecord:
        """Register an episode. The same id and content returns the stored row.

        A second registration does not clear a soft-delete and does not
        change the first source. Different content for the same id conflicts.
        """
        _validate_registration(episode)
        saved: dict[str, EpisodeRecord] = {}

        def mutate(document: dict[str, Any]) -> None:
            current = document["episodes"].get(episode.episode_id)
            if current is None:
                document["episodes"][episode.episode_id] = episode.to_dict()
                saved["episode"] = episode
                return
            existing = _episode_from_dict(current)
            if (
                existing.content_hash != episode.content_hash
                or existing.task != episode.task
                or existing.fps != episode.fps
                or existing.frame_count != episode.frame_count
            ):
                raise LineageConflict(
                    f"episode {episode.episode_id} is already registered "
                    "with different content"
                )
            saved["episode"] = existing

        self._update(mutate)
        return saved["episode"]

    def episode(self, episode_id: str) -> EpisodeRecord:
        """Return one episode row, including a soft-deleted row."""
        _check_episode_id(episode_id)
        raw = self._read()["episodes"].get(episode_id)
        if raw is None:
            raise LineageNotFound(f"episode {episode_id} was not found")
        return _episode_from_dict(raw)

    def link_version_episodes(
        self, version_id: str, episode_ids: Sequence[str]
    ) -> None:
        """Record the episode list for a dataset version.

        The list is ordered. The same list is a no-op. A different list
        conflicts. Episodes must already be registered.
        """
        _check_version_id(version_id)
        wanted = _episode_id_list(episode_ids)

        def mutate(document: dict[str, Any]) -> None:
            for episode_id in wanted:
                if episode_id not in document["episodes"]:
                    raise LineageError(f"episode {episode_id} is not registered")
            current = document["version_episodes"].get(version_id)
            if current is None:
                document["version_episodes"][version_id] = list(wanted)
                return
            if list(current) != list(wanted):
                raise LineageConflict(
                    f"dataset version {version_id} already lists different episodes"
                )

        self._update(mutate)

    def link_run(self, run_id: str, dataset_version_id: str) -> None:
        """Cite ``dataset_version_id`` from ``run_id``.

        One run cites one dataset version. The version must already list
        episodes. The stored object is the two ids and nothing else.
        """
        _check_ref(run_id, "run id")
        _check_version_id(dataset_version_id)

        def mutate(document: dict[str, Any]) -> None:
            episodes = document["version_episodes"].get(dataset_version_id)
            if not episodes:
                raise LineageError(
                    f"dataset version {dataset_version_id} has no episode edges"
                )
            current = document["runs"].get(run_id)
            if current is None:
                document["runs"][run_id] = {
                    "run_id": run_id,
                    "dataset_version_id": dataset_version_id,
                }
                return
            if current["dataset_version_id"] != dataset_version_id:
                raise LineageConflict(
                    f"run {run_id} is already linked to {current['dataset_version_id']}"
                )

        self._update(mutate)

    def link_model(self, model_version_id: str, run_id: str) -> None:
        """Cite ``run_id`` as the run that produced ``model_version_id``.

        The run must already cite a dataset version. One model version cites
        one run.
        """
        _check_ref(model_version_id, "model version id")
        _check_ref(run_id, "run id")

        def mutate(document: dict[str, Any]) -> None:
            if run_id not in document["runs"]:
                raise LineageError(f"run {run_id} has no dataset version")
            current = document["models"].get(model_version_id)
            if current is None:
                document["models"][model_version_id] = {
                    "model_version_id": model_version_id,
                    "run_id": run_id,
                }
                return
            if current["run_id"] != run_id:
                raise LineageConflict(
                    f"model version {model_version_id} is already linked to "
                    f"{current['run_id']}"
                )

        self._update(mutate)

    def episodes_for_version(self, version_id: str) -> tuple[str, ...]:
        """Episode ids in the order stored for ``version_id``."""
        _check_version_id(version_id)
        document = self._read()
        if version_id not in document["version_episodes"]:
            raise LineageNotFound(f"dataset version {version_id} has no episode edges")
        return tuple(document["version_episodes"][version_id])

    def versions_for_episode(self, episode_id: str) -> tuple[str, ...]:
        """Dataset versions that cite ``episode_id``, ordered by version id."""
        _check_episode_id(episode_id)
        document = self._read()
        if episode_id not in document["episodes"]:
            raise LineageNotFound(f"episode {episode_id} was not found")
        return _versions_citing(document, episode_id)

    def lineage_from_model(self, model_version_id: str) -> ModelLineage:
        """Walk model version → run → dataset version → episodes."""
        _check_ref(model_version_id, "model version id")
        document = self._read()
        model = document["models"].get(model_version_id)
        if model is None:
            raise LineageNotFound(f"model version {model_version_id} was not found")
        run_id = model["run_id"]
        run = document["runs"][run_id]
        version_id = run["dataset_version_id"]
        return ModelLineage(
            model_version_id=model_version_id,
            run_id=run_id,
            dataset_version_id=version_id,
            episode_ids=tuple(document["version_episodes"][version_id]),
        )

    def models_trained_on(self, episode_id: str) -> tuple[EpisodeTraining, ...]:
        """Every model version trained on ``episode_id``.

        A soft-deleted episode still answers, because the edges were kept.
        An episode that no run used returns an empty tuple.
        """
        _check_episode_id(episode_id)
        document = self._read()
        if episode_id not in document["episodes"]:
            raise LineageNotFound(f"episode {episode_id} was not found")
        found: list[EpisodeTraining] = []
        for version_id, episode_ids in document["version_episodes"].items():
            if episode_id not in episode_ids:
                continue
            for run_id, run in document["runs"].items():
                if run["dataset_version_id"] != version_id:
                    continue
                for model_id, model in document["models"].items():
                    if model["run_id"] != run_id:
                        continue
                    found.append(
                        EpisodeTraining(
                            episode_id=episode_id,
                            dataset_version_id=version_id,
                            run_id=run_id,
                            model_version_id=model_id,
                        )
                    )
        found.sort(
            key=lambda item: (
                item.dataset_version_id,
                item.run_id,
                item.model_version_id,
            )
        )
        return tuple(found)

    def delete_episode(self, episode_id: str) -> EpisodeDeletion:
        """Remove ``episode_id``, or soft-delete it when a pin still cites it.

        Pinned means ``version.json`` has ``pinned: true``, or the version
        file cannot be read. The dataset directory is not modified. Edges to
        a pinned version stay, so later lineage queries still see the episode.
        """
        _check_episode_id(episode_id)
        outcome: dict[str, EpisodeDeletion] = {}

        def mutate(document: dict[str, Any]) -> None:
            raw = document["episodes"].get(episode_id)
            if raw is None:
                raise LineageNotFound(f"episode {episode_id} was not found")
            current = _episode_from_dict(raw)
            versions = _versions_citing(document, episode_id)
            pinned = tuple(
                version_id for version_id in versions if self._held_by_pin(version_id)
            )
            if pinned:
                warning = _soft_warning(episode_id, pinned)
                updated = replace(
                    current,
                    deleted=True,
                    deleted_at=current.deleted_at or _now(),
                    delete_warning=warning,
                )
                document["episodes"][episode_id] = updated.to_dict()
                outcome["result"] = EpisodeDeletion(
                    episode_id=episode_id,
                    removed=False,
                    soft_deleted=True,
                    blocked=True,
                    warning=warning,
                    pinned_version_ids=pinned,
                )
                return
            del document["episodes"][episode_id]
            for version_id, episode_ids in list(document["version_episodes"].items()):
                document["version_episodes"][version_id] = [
                    item for item in episode_ids if item != episode_id
                ]
            outcome["result"] = EpisodeDeletion(
                episode_id=episode_id,
                removed=True,
                soft_deleted=False,
                blocked=False,
                warning=None,
                pinned_version_ids=(),
            )

        self._update(mutate)
        return outcome["result"]

    def _held_by_pin(self, version_id: str) -> bool:
        try:
            return load_version(version_id, store=self.root).pinned
        except DatasetVersionError:
            return True

    def _read(self) -> dict[str, Any]:
        path = self.path
        if not path.is_file():
            return _empty_graph()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LineageError("lineage graph is not valid JSON") from exc
        return _graph_from_document(document)

    def _update(self, mutate: Any) -> None:
        document = self._read()
        before = _dump(document)
        mutate(document)
        checked = _graph_from_document(document)
        after = _dump(checked)
        if after != before:
            _atomic_write_text(self.path, after)


def _episode_bytes(episode: CanonicalEpisode) -> bytes:
    payload = {"algorithm": EPISODE_ID_ALGORITHM, "episode": episode.to_dict()}
    text = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return (text + "\n").encode("utf-8")


def _empty_graph() -> dict[str, Any]:
    return {
        "schema": SCHEMA_NAME,
        "episodes": {},
        "version_episodes": {},
        "runs": {},
        "models": {},
    }


def _dump(document: Mapping[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _soft_warning(episode_id: str, pinned: Sequence[str]) -> str:
    label = "version" if len(pinned) == 1 else "versions"
    joined = ", ".join(pinned)
    return (
        f"soft-deleted: episode {episode_id} is used by pinned dataset "
        f"{label} {joined} and was not removed"
    )


def _versions_citing(document: Mapping[str, Any], episode_id: str) -> tuple[str, ...]:
    found = [
        version_id
        for version_id, episode_ids in document["version_episodes"].items()
        if episode_id in episode_ids
    ]
    return tuple(sorted(found))


def _graph_from_document(document: Any) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise LineageError("lineage graph must be an object")
    unknown = sorted(set(document) - set(_GRAPH_KEYS))
    if unknown:
        names = ", ".join(unknown)
        raise LineageError(f"lineage graph has unknown fields: {names}")
    for key in _GRAPH_KEYS:
        if key not in document:
            raise LineageError(f"lineage graph is missing {key}")
    if document["schema"] != SCHEMA_NAME:
        raise LineageError("lineage graph schema is not ohho.lineage.v1")
    episodes = _mapping(document["episodes"], "episodes")
    parsed_episodes = {
        episode_id: _episode_from_dict(raw).to_dict()
        for episode_id, raw in episodes.items()
    }
    for episode_id, raw in parsed_episodes.items():
        if raw["episode_id"] != episode_id:
            raise LineageError(f"episode key {episode_id} does not match its row")
    version_episodes = _version_edges(document["version_episodes"], parsed_episodes)
    runs = _run_rows(document["runs"], version_episodes)
    models = _model_rows(document["models"], runs)
    return {
        "schema": SCHEMA_NAME,
        "episodes": parsed_episodes,
        "version_episodes": version_episodes,
        "runs": runs,
        "models": models,
    }


def _version_edges(raw: Any, episodes: Mapping[str, Any]) -> dict[str, list[str]]:
    mapping = _mapping(raw, "version_episodes")
    edges: dict[str, list[str]] = {}
    for version_id, episode_ids in mapping.items():
        _check_version_id(version_id)
        if not isinstance(episode_ids, list):
            raise LineageError(f"dataset version {version_id} episodes must be a list")
        seen: set[str] = set()
        cleaned: list[str] = []
        for episode_id in episode_ids:
            _check_episode_id(episode_id)
            if episode_id in seen:
                raise LineageError(
                    f"dataset version {version_id} repeats episode {episode_id}"
                )
            if episode_id not in episodes:
                raise LineageError(
                    f"dataset version {version_id} cites missing episode {episode_id}"
                )
            seen.add(episode_id)
            cleaned.append(episode_id)
        edges[version_id] = cleaned
    return edges


def _run_rows(
    raw: Any, version_episodes: Mapping[str, list[str]]
) -> dict[str, dict[str, str]]:
    mapping = _mapping(raw, "runs")
    runs: dict[str, dict[str, str]] = {}
    for run_id, row in mapping.items():
        _check_ref(run_id, "run id")
        body = _exact_object(row, _RUN_KEYS, f"run {run_id}")
        if body["run_id"] != run_id:
            raise LineageError(f"run key {run_id} does not match its row")
        version_id = body["dataset_version_id"]
        _check_version_id(version_id)
        if version_id not in version_episodes:
            raise LineageError(
                f"run {run_id} cites dataset version {version_id} with no episode edges"
            )
        runs[run_id] = {"run_id": run_id, "dataset_version_id": version_id}
    return runs


def _model_rows(
    raw: Any, runs: Mapping[str, Mapping[str, str]]
) -> dict[str, dict[str, str]]:
    mapping = _mapping(raw, "models")
    models: dict[str, dict[str, str]] = {}
    for model_id, row in mapping.items():
        _check_ref(model_id, "model version id")
        body = _exact_object(row, _MODEL_KEYS, f"model version {model_id}")
        if body["model_version_id"] != model_id:
            raise LineageError(f"model version key {model_id} does not match its row")
        run_id = body["run_id"]
        _check_ref(run_id, "run id")
        if run_id not in runs:
            raise LineageError(f"model version {model_id} cites missing run {run_id}")
        models[model_id] = {"model_version_id": model_id, "run_id": run_id}
    return models


def _episode_from_dict(raw: Any) -> EpisodeRecord:
    body = _exact_object(raw, _EPISODE_KEYS, "episode")
    episode_id = body["episode_id"]
    content_hash = body["content_hash"]
    _check_episode_id(episode_id)
    _check_content_hash(content_hash)
    if episode_id != "ep_" + content_hash[len("sha256:") :][:32]:
        raise LineageError(f"episode {episode_id} does not match its content hash")
    deleted = body["deleted"]
    deleted_at = body["deleted_at"]
    warning = body["delete_warning"]
    if not isinstance(deleted, bool):
        raise LineageError(f"episode {episode_id} deleted flag must be a boolean")
    if deleted:
        if not isinstance(deleted_at, str) or _TIME_RE.fullmatch(deleted_at) is None:
            raise LineageError(f"episode {episode_id} is missing deleted_at")
        if not isinstance(warning, str) or not warning.strip():
            raise LineageError(f"episode {episode_id} is missing a delete warning")
        if len(warning) > 8192:
            raise LineageError(f"episode {episode_id} delete warning is too long")
    elif deleted_at is not None or warning is not None:
        raise LineageError(f"episode {episode_id} carries a delete marker")
    frame_count = body["frame_count"]
    source_index = body["source_index"]
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
    ):
        raise LineageError(
            f"episode {episode_id} frame_count must be a positive integer"
        )
    if (
        isinstance(source_index, bool)
        or not isinstance(source_index, int)
        or source_index < 0
    ):
        raise LineageError(
            f"episode {episode_id} source_index must be a non-negative integer"
        )
    return EpisodeRecord(
        episode_id=episode_id,
        content_hash=content_hash,
        task=_text(body["task"], "task", 4096, empty_ok=True),
        fps=_text(body["fps"], "fps", 64, empty_ok=False),
        frame_count=frame_count,
        source=_text(body["source"], "source", 4096, empty_ok=False),
        source_index=source_index,
        deleted=deleted,
        deleted_at=deleted_at,
        delete_warning=warning,
    )


def _validate_registration(episode: EpisodeRecord) -> None:
    if episode.deleted or episode.deleted_at is not None or episode.delete_warning:
        raise LineageError("register an episode before deleting it")
    _episode_from_dict(episode.to_dict())


def _episode_id_list(episode_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(episode_ids, (str, bytes)) or not isinstance(episode_ids, Sequence):
        raise LineageError("episode ids must be a list")
    if not episode_ids:
        raise LineageError("a dataset version needs at least one episode")
    seen: set[str] = set()
    cleaned: list[str] = []
    for episode_id in episode_ids:
        _check_episode_id(episode_id)
        if episode_id in seen:
            raise LineageError(f"episode {episode_id} is repeated")
        seen.add(episode_id)
        cleaned.append(episode_id)
    return tuple(cleaned)


def _check_episode_id(episode_id: str) -> None:
    if not isinstance(episode_id, str) or _EPISODE_ID_RE.fullmatch(episode_id) is None:
        raise LineageError(f"invalid episode id: {episode_id}")


def _check_version_id(version_id: str) -> None:
    if not isinstance(version_id, str) or not is_version_id(version_id):
        raise LineageError(f"invalid dataset version id: {version_id}")


def _check_content_hash(content_hash: str) -> None:
    if (
        not isinstance(content_hash, str)
        or _CONTENT_HASH_RE.fullmatch(content_hash) is None
    ):
        raise LineageError("episode content hash is not a sha256 digest")


def _check_ref(value: str, label: str) -> None:
    if not isinstance(value, str) or _REF_ID_RE.fullmatch(value) is None:
        raise LineageError(f"invalid {label}: {value}")


def _text(value: Any, label: str, limit: int, *, empty_ok: bool) -> str:
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise LineageError(f"episode {label} is invalid")
    if any(ord(char) < 32 for char in value):
        raise LineageError(f"episode {label} must be a single line")
    if not empty_ok and not value.strip():
        raise LineageError(f"episode {label} is empty")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LineageError(f"lineage {label} must be an object")
    for key in value:
        if not isinstance(key, str):
            raise LineageError(f"lineage {label} keys must be strings")
    return value


def _exact_object(value: Any, keys: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LineageError(f"{label} must be an object")
    unknown = sorted(set(value) - set(keys))
    if unknown:
        names = ", ".join(unknown)
        raise LineageError(f"{label} has unknown fields: {names}")
    for key in keys:
        if key not in value:
            raise LineageError(f"{label} is missing {key}")
    return value
