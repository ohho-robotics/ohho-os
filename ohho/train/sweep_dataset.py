"""Dataset-version axis for ``ohho.sweep/v1``.

A ``dataset`` dimension lists OHH-513 version ids. Each trial records the id
it selected. :class:`DatasetVersionCache` materialises that version once per
worker: a later trial with the same id is a cache hit and does not fetch or
copy again. :func:`group_by_dataset_version` buckets a trial table by that id.
``ohho sweep group`` prints the same grouping for a persisted run.

``ohho.run/v1`` is not defined in this package. A later manifest writer
copies ``dataset_version_id`` from the trial row.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..datasets.model import is_version_id
from .sweep import (
    SweepError,
    Trial,
    _assert_refs,
    _read_resource,
    _validate,
)

GROUPS_SCHEMA_ID = "ohho.sweep.dataset_groups/v1"
GROUPS_FILENAME = "sweep.dataset_groups.v1.json"


@dataclass(frozen=True)
class DatasetVersionGroup:
    """Trials that share one dataset version id, in proposal order.

    ``dataset_version_id`` is None when those trials did not select one.
    """

    dataset_version_id: str | None
    trials: tuple[dict[str, Any], ...]


class DatasetVersionCache:
    """Worker-local copy of dataset versions, keyed by version id.

    ``fetcher`` returns the source directory for a version id. The first
    request copies that tree into ``root/<version_id>``. A later request
    for the same id, including from a new cache pointed at the same root,
    returns the stored directory and does not call ``fetcher`` or copy
    again. ``fetch_count`` and ``copy_count`` count those misses.
    """

    def __init__(
        self,
        fetcher: Callable[[str], str | Path],
        root: str | Path,
    ) -> None:
        self._fetcher = fetcher
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._paths: dict[str, Path] = {}
        self.fetch_count = 0
        self.copy_count = 0

    def resolve(self, version_id: str) -> Path:
        """Return the cached dataset directory for ``version_id``."""

        cached = self._paths.get(version_id)
        if cached is not None:
            return cached
        dest = self._destination(version_id)
        if _nonempty_dir(dest):
            self._paths[version_id] = dest
            return dest
        source = self._fetch(version_id)
        if source.resolve() == dest:
            self._paths[version_id] = dest
            return dest
        self._copy_into(source, dest)
        self._paths[version_id] = dest
        return dest

    def _destination(self, version_id: str) -> Path:
        if not isinstance(version_id, str) or not is_version_id(version_id):
            raise SweepError(f"invalid dataset version id: {version_id}")
        dest = (self.root / version_id).resolve()
        if dest.parent != self.root:
            raise SweepError("dataset version id escapes the cache")
        return dest

    def _fetch(self, version_id: str) -> Path:
        source = Path(self._fetcher(version_id)).expanduser()
        self.fetch_count += 1
        if not source.is_dir():
            raise SweepError(
                f"dataset version {version_id} did not resolve to a directory"
            )
        return source

    def _copy_into(self, source: Path, dest: Path) -> None:
        if dest.exists():
            if dest.is_dir():
                dest.rmdir()
            else:
                dest.unlink()
        staging = dest.with_name(f".{dest.name}.{os.getpid()}.partial")
        if staging.exists():
            shutil.rmtree(staging)
        try:
            shutil.copytree(source, staging)
            self.copy_count += 1
            os.replace(staging, dest)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise


def worker_dataset_cache(
    root: str | Path | None = None,
    *,
    fetcher: Callable[[str], str | Path] | None = None,
    store: str | Path | None = None,
) -> DatasetVersionCache:
    """Cache for one worker.

    ``root`` defaults to ``OHHO_SWEEP_DATASET_CACHE`` or
    ``~/.ohho/sweep-datasets``. ``fetcher`` defaults to
    :func:`ohho.datasets.resolve_dataset_version` with ``store``.
    """

    if fetcher is None:
        fetcher = _store_fetcher(store)
    if root is None:
        env = os.environ.get("OHHO_SWEEP_DATASET_CACHE", "").strip()
        root = env if env else Path.home() / ".ohho" / "sweep-datasets"
    return DatasetVersionCache(fetcher, root)


def resolve_trial_dataset(
    trial: Trial | Mapping[str, Any],
    cache: DatasetVersionCache,
) -> Path:
    """Resolve the dataset version recorded on ``trial`` through ``cache``."""

    version_id = trial_dataset_version_id(trial)
    if version_id is None:
        raise SweepError("trial has no dataset version")
    return cache.resolve(version_id)


def resolve_trial_datasets(
    trials: Sequence[Trial | Mapping[str, Any]],
    cache: DatasetVersionCache,
) -> tuple[Path, ...]:
    """Resolve each trial's dataset version, sharing ``cache``."""

    return tuple(resolve_trial_dataset(trial, cache) for trial in trials)


def trial_dataset_version_id(trial: Trial | Mapping[str, Any]) -> str | None:
    """Version id stored on ``trial``, or None when the trial has none."""

    if isinstance(trial, Trial):
        return trial.dataset_version_id
    if isinstance(trial, Mapping):
        value = trial.get("dataset_version_id")
        if value is None:
            return None
        if isinstance(value, str) and is_version_id(value):
            return value
        raise SweepError("dataset_version_id must be a dataset version id")
    raise SweepError("trial must be a trial or a mapping")


def group_by_dataset_version(
    trials: Sequence[Trial | Mapping[str, Any]],
) -> tuple[DatasetVersionGroup, ...]:
    """Group ``trials`` by dataset version id, in first-seen order."""

    order: list[str | None] = []
    buckets: dict[str | None, list[dict[str, Any]]] = {}
    for trial in trials:
        version_id = trial_dataset_version_id(trial)
        if version_id not in buckets:
            order.append(version_id)
            buckets[version_id] = []
        buckets[version_id].append(_trial_row(trial))
    return tuple(
        DatasetVersionGroup(version_id, tuple(buckets[version_id]))
        for version_id in order
    )


def groups_as_dict(
    sweep_id: str,
    groups: Sequence[DatasetVersionGroup],
) -> dict[str, Any]:
    """JSON-ready ``ohho.sweep.dataset_groups/v1`` document."""

    payload = {
        "schema": GROUPS_SCHEMA_ID,
        "id": sweep_id,
        "groups": [
            {
                "dataset_version_id": group.dataset_version_id,
                "trials": [copy.deepcopy(row) for row in group.trials],
            }
            for group in groups
        ],
    }
    _check_schema(payload, _groups_schema())
    return payload


def format_dataset_groups(
    sweep_id: str,
    groups: Sequence[DatasetVersionGroup],
) -> str:
    """Text for ``ohho sweep group``."""

    lines = [
        f"schema: {GROUPS_SCHEMA_ID}",
        f"id: {sweep_id}",
        f"groups: {len(groups)}",
    ]
    for group in groups:
        label = group.dataset_version_id or "(none)"
        lines.append("---")
        lines.append(f"dataset_version_id: {label}")
        lines.append(f"trials: {len(group.trials)}")
        lines.extend(_format_grouped_trial(trial) for trial in group.trials)
    return "\n".join(lines) + "\n"


def load_groups_schema() -> dict[str, Any]:
    """Return a copy of the ``ohho.sweep.dataset_groups/v1`` JSON Schema."""

    return copy.deepcopy(_groups_schema())


def _store_fetcher(store: str | Path | None) -> Callable[[str], Path]:
    def fetch(version_id: str) -> Path:
        from ..datasets import resolve_dataset_version

        return resolve_dataset_version(version_id, store=store)

    return fetch


def _trial_row(trial: Trial | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trial, Trial):
        row: dict[str, Any] = {
            "index": trial.index,
            "seed": trial.seed,
            "overrides": copy.deepcopy(trial.overrides),
            "sets": list(trial.sets),
            "recipe": copy.deepcopy(trial.recipe),
            "estimated_gpu_hours": float(trial.estimated_gpu_hours),
            "values": None,
        }
        if trial.dataset_version_id is not None:
            row["dataset_version_id"] = trial.dataset_version_id
        return row
    if isinstance(trial, Mapping):
        return copy.deepcopy(dict(trial))
    raise SweepError("trial must be a trial or a mapping")


def _format_grouped_trial(trial: Mapping[str, Any]) -> str:
    index = trial.get("index", trial.get("trial"))
    parts = [f"  trial: {index}"]
    if "seed" in trial:
        parts.append(f"seed: {trial['seed']}")
    values = trial.get("values")
    rendered = (
        "null"
        if values is None
        else json.dumps(values, sort_keys=True, ensure_ascii=True)
    )
    parts.append(f"values: {rendered}")
    return " ".join(parts)


def _nonempty_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.iterdir())


def _groups_schema() -> dict[str, Any]:
    schema = getattr(_groups_schema, "_value", None)
    if schema is None:
        schema = json.loads(_read_resource("schemas", GROUPS_FILENAME))
        _assert_refs(schema, schema, "$")
        _groups_schema._value = schema  # type: ignore[attr-defined]
    return schema


def _check_schema(instance: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    _validate(dict(instance), schema, schema, "$", 0)


__all__ = [
    "GROUPS_SCHEMA_ID",
    "DatasetVersionCache",
    "DatasetVersionGroup",
    "format_dataset_groups",
    "group_by_dataset_version",
    "groups_as_dict",
    "load_groups_schema",
    "resolve_trial_dataset",
    "resolve_trial_datasets",
    "trial_dataset_version_id",
    "worker_dataset_cache",
]
