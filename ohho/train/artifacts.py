"""Per-run artifact layout, retention, and storage totals.

The run directory is the folder that contains ``run_manifest.json``. Retention
reads and writes ``artifact_index.json`` beside it. It does not rewrite the
manifest or ``metrics.json``. Checkpoint files it removes are recorded in the
sidecar. See ``docs/train/artifacts.md``.

This module does not open a database, read credentials, or use the network.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .run_manifest import MANIFEST_NAME
from .schema_validate import validate_instance

SCHEMA_ID = "ohho.artifacts/v1"
SCHEMA_FILENAME = "ohho.artifacts.v1.json"
INDEX_NAME = "artifact_index.json"
CHECKPOINTS_DIR = "checkpoints"
LOGS_DIR = "logs"
EVAL_DIR = "eval"
METRICS_NAME = "metrics.json"
DEFAULT_MAX_AGE_DAYS = 14

LAYOUT = {
    "manifest": MANIFEST_NAME,
    "metrics": METRICS_NAME,
    "index": INDEX_NAME,
    "checkpoints": CHECKPOINTS_DIR,
    "logs": LOGS_DIR,
    "eval": EVAL_DIR,
}

_SCHEMA: dict[str, Any] | None = None
_EXPERIMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STEP = re.compile(r"step[-_]\d+")
_EPOCH = re.compile(r"epoch[-_]\d+")
_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".ckpt",
    ".onnx",
    ".pth",
    ".bin",
    ".json",
    ".pt",
)
_SKIP_DIRS = {".git", "__pycache__"}
_PROTECTED = {MANIFEST_NAME, METRICS_NAME, INDEX_NAME}
_ROOT_CHECKPOINT_FILES = {"policy.pt", "checkpoint.json"}
_ROOT_CHECKPOINT_DIRS = {"model_final"}


class _Leave:
    """A ``set_run_retention`` field the caller did not pass."""


_LEAVE = _Leave()


class ArtifactError(ValueError):
    """A retention or storage request cannot be completed."""


@dataclass(frozen=True)
class RetentionPolicy:
    """Delete intermediate checkpoints of pruned trials after ``max_age_days``.

    Best and last checkpoints, logs, eval reports, the manifest, and metrics
    are always kept. Pinned and promoted runs and checkpoints are kept.
    """

    max_age_days: int = DEFAULT_MAX_AGE_DAYS

    def __post_init__(self) -> None:
        if type(self.max_age_days) is not int or self.max_age_days < 0:
            raise ArtifactError("max_age_days must be an integer >= 0")
        try:
            timedelta(days=self.max_age_days)
        except OverflowError as exc:
            raise ArtifactError("max_age_days is too large") from exc


@dataclass(frozen=True)
class ArtifactDecision:
    """One checkpoint retention decided to delete or to keep."""

    path: str
    role: str
    size_bytes: int
    reason: str
    deleted_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "path": self.path,
            "role": self.role,
            "bytes": self.size_bytes,
            "reason": self.reason,
        }
        if self.deleted_at is not None:
            body["deleted_at"] = self.deleted_at
        return body


@dataclass(frozen=True)
class RunRetentionResult:
    """Retention outcome for one run directory."""

    run_dir: str
    experiment: str | None
    pinned: bool
    promoted: bool
    pruned: bool
    deleted: tuple[ArtifactDecision, ...]
    kept: tuple[ArtifactDecision, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "experiment": self.experiment,
            "pinned": self.pinned,
            "promoted": self.promoted,
            "pruned": self.pruned,
            "deleted": [item.to_dict() for item in self.deleted],
            "kept": [item.to_dict() for item in self.kept],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class RetentionResult:
    """Retention outcome for every run the job visited."""

    dry_run: bool
    max_age_days: int
    runs: tuple[RunRetentionResult, ...]

    @property
    def deleted_bytes(self) -> int:
        return sum(item.size_bytes for run in self.runs for item in run.deleted)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(error for run in self.runs for error in run.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "max_age_days": self.max_age_days,
            "deleted_bytes": self.deleted_bytes,
            "runs": [run.to_dict() for run in self.runs],
        }


@dataclass(frozen=True)
class StorageUsage:
    """Bytes still on disk for one run, plus bytes recorded as deleted."""

    run_dir: str
    experiment: str | None
    total_bytes: int
    checkpoints_bytes: int
    logs_bytes: int
    eval_bytes: int
    metrics_bytes: int
    manifest_bytes: int
    other_bytes: int
    deleted_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "experiment": self.experiment,
            "total_bytes": self.total_bytes,
            "checkpoints_bytes": self.checkpoints_bytes,
            "logs_bytes": self.logs_bytes,
            "eval_bytes": self.eval_bytes,
            "metrics_bytes": self.metrics_bytes,
            "manifest_bytes": self.manifest_bytes,
            "other_bytes": self.other_bytes,
            "deleted_bytes": self.deleted_bytes,
        }


@dataclass(frozen=True)
class ExperimentStorage:
    """Sum of :func:`storage_used` across the runs under one directory."""

    experiment: str | None
    root: str
    runs: tuple[StorageUsage, ...]

    @property
    def total_bytes(self) -> int:
        return sum(run.total_bytes for run in self.runs)

    @property
    def deleted_bytes(self) -> int:
        return sum(run.deleted_bytes for run in self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "root": self.root,
            "total_bytes": self.total_bytes,
            "deleted_bytes": self.deleted_bytes,
            "runs": [run.to_dict() for run in self.runs],
        }


def schema_path() -> Path:
    """Path to the packaged ``ohho.artifacts/v1`` schema."""
    return Path(__file__).resolve().parent / "schema" / SCHEMA_FILENAME


def schema_document() -> dict[str, Any]:
    """Load the artifact index schema."""
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(schema_path().read_text(encoding="utf-8"))
    return _SCHEMA


def validate_artifact_index(document: Any) -> list[str]:
    """Return schema errors. An empty list means ``document`` is valid."""
    if not isinstance(document, dict):
        return ["$: expected object"]
    return validate_instance(document, schema_document())


def empty_artifact_index() -> dict[str, Any]:
    """An index for a run that has not been pinned, promoted, or pruned."""
    return {
        "schema": SCHEMA_ID,
        "experiment": None,
        "pinned": False,
        "promoted": False,
        "pruned": False,
        "pruned_at": None,
        "checkpoints": [],
        "deletions": [],
    }


def checkpoint_role(name: str) -> str:
    """Role of a checkpoint file or directory name.

    One weight suffix is removed first (``.pt``, ``.pth``, ``.ckpt``,
    ``.safetensors``, ``.onnx``, ``.bin``, ``.json``). ``best`` and ``last``
    are kept. ``step-<n>`` and ``epoch-<n>`` (hyphen or underscore) are
    intermediate roles. Every other name under ``checkpoints/`` is ``other``.
    """
    stem = name
    for suffix in _WEIGHT_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            stem = name[: -len(suffix)]
            break
    if stem == "best":
        return "best"
    if stem == "last":
        return "last"
    if _STEP.fullmatch(stem):
        return "step"
    if _EPOCH.fullmatch(stem):
        return "epoch"
    return "other"


def ensure_layout(run_dir: str | Path) -> Path:
    """Create ``checkpoints/``, ``logs/``, and ``eval/`` inside ``run_dir``."""
    root = Path(run_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for name in (CHECKPOINTS_DIR, LOGS_DIR, EVAL_DIR):
        (root / name).mkdir(exist_ok=True)
    return root


def load_artifact_index(run_dir: str | Path) -> dict[str, Any]:
    """Return the sidecar, or an empty index when the file is absent.

    Does not create the file. Raises :class:`ArtifactError` when the file
    exists and is not valid ``ohho.artifacts/v1``.
    """
    root = _directory(run_dir)
    path = root / INDEX_NAME
    if not path.exists():
        return empty_artifact_index()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"could not read artifact index: {exc}") from exc
    errors = validate_artifact_index(document)
    if errors:
        message = "artifact index is not valid ohho.artifacts/v1: " + "; ".join(errors)
        raise ArtifactError(message)
    return document


def set_run_retention(
    run_dir: str | Path,
    *,
    experiment: str | None | _Leave = _LEAVE,
    pinned: bool | None = None,
    promoted: bool | None = None,
    pruned: bool | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Set pin, promote, prune, or experiment on a run. Writes the sidecar.

    ``experiment=None`` clears the id. Omitting it leaves the stored id.
    Marking a trial pruned records ``pruned_at`` once. Clearing prune clears
    ``pruned_at``. The manifest and metrics file are not modified.
    """
    root = _require_run(run_dir)
    pinned = _check_bool("pinned", pinned)
    promoted = _check_bool("promoted", promoted)
    pruned = _check_bool("pruned", pruned)
    index = load_artifact_index(root)
    if not isinstance(experiment, _Leave):
        index["experiment"] = _check_experiment(experiment)
    if pinned is not None:
        index["pinned"] = pinned
    if promoted is not None:
        index["promoted"] = promoted
    if pruned is True:
        index["pruned"] = True
        if index.get("pruned_at") is None:
            index["pruned_at"] = _format_time(_as_utc(now))
    elif pruned is False:
        index["pruned"] = False
        index["pruned_at"] = None
    index["checkpoints"] = _checkpoint_entries(root, index)
    _write_index(root / INDEX_NAME, index)
    return load_artifact_index(root)


def set_checkpoint_retention(
    run_dir: str | Path,
    relative_path: str,
    *,
    pinned: bool | None = None,
    promoted: bool | None = None,
) -> dict[str, Any]:
    """Pin or promote one checkpoint under ``checkpoints/``.

    ``relative_path`` is ``checkpoints/<name>`` or the checkpoint's file name
    when that name is unique. The manifest and metrics file are not modified.
    """
    root = _require_run(run_dir)
    pinned = _check_bool("pinned", pinned)
    promoted = _check_bool("promoted", promoted)
    if pinned is None and promoted is None:
        raise ArtifactError("set pinned or promoted")
    index = load_artifact_index(root)
    key = _resolve_checkpoint_key(root, relative_path)
    entries = _checkpoint_entries(root, index)
    found = False
    for entry in entries:
        if entry["path"] != key:
            continue
        if pinned is not None:
            entry["pinned"] = pinned
        if promoted is not None:
            entry["promoted"] = promoted
        found = True
        break
    if not found:
        raise ArtifactError(f"checkpoint not found: {relative_path}")
    index["checkpoints"] = entries
    _write_index(root / INDEX_NAME, index)
    return load_artifact_index(root)


def storage_used(run_dir: str | Path) -> StorageUsage:
    """Bytes on disk for one run directory.

    The console calls this for a run page. ``deleted_bytes`` is the sum of
    deletion records in the sidecar and is not part of ``total_bytes``.
    """
    root = _require_run(run_dir)
    return _usage(root)


def storage_used_experiment(
    root: str | Path,
    experiment: str | None = None,
) -> ExperimentStorage:
    """Sum :func:`storage_used` for each run under ``root``.

    A directory that itself contains ``run_manifest.json`` is one run.
    Otherwise each nested run is included and the walk does not descend into
    a run. When ``experiment`` is set, a run matches if its sidecar records
    that id, or if that field is null and the parent directory has that name.
    """
    anchor, runs = _runs_under(root)
    experiment = _check_experiment(experiment) if experiment is not None else None
    selected: list[StorageUsage] = []
    for run_dir in runs:
        index = load_artifact_index(run_dir)
        if not _matches(run_dir, index, experiment):
            continue
        selected.append(_usage(run_dir, index))
    if experiment is not None and not selected:
        raise ArtifactError(f"no run matches experiment {experiment!r}")
    return ExperimentStorage(
        experiment=experiment,
        root=str(anchor),
        runs=tuple(selected),
    )


def apply_retention(
    path: str | Path,
    *,
    policy: RetentionPolicy | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
    experiment: str | None = None,
) -> RetentionResult:
    """Delete eligible checkpoints, or report them when ``dry_run`` is set.

    An intermediate checkpoint under ``checkpoints/`` is deleted only when
    the trial is pruned, the run is not pinned or promoted, the checkpoint
    is not pinned or promoted, and its newest mtime is strictly older than
    ``policy.max_age_days`` (default 14). Best and last checkpoints, logs,
    eval reports, the manifest, metrics, and run-root trainer files stay.

    ``dry_run`` does not remove files and does not write the sidecar.
    Deletions that do happen are appended to ``artifact_index.json``.
    """
    policy = RetentionPolicy() if policy is None else policy
    if not isinstance(policy, RetentionPolicy):
        raise ArtifactError("policy must be a RetentionPolicy")
    moment = _as_utc(now)
    experiment = _check_experiment(experiment) if experiment is not None else None
    _anchor, runs = _runs_under(path)
    outcomes: list[RunRetentionResult] = []
    matched = 0
    for run_dir in runs:
        try:
            index = load_artifact_index(run_dir)
        except ArtifactError as exc:
            outcomes.append(_error_run(run_dir, str(exc)))
            continue
        if not _matches(run_dir, index, experiment):
            continue
        matched += 1
        outcomes.append(
            _retain_one(
                run_dir,
                index,
                policy,
                moment,
                dry_run=dry_run,
            )
        )
    if (
        experiment is not None
        and matched == 0
        and not any(run.errors for run in outcomes)
    ):
        raise ArtifactError(f"no run matches experiment {experiment!r}")
    return RetentionResult(
        dry_run=dry_run,
        max_age_days=policy.max_age_days,
        runs=tuple(outcomes),
    )


def format_retention(result: RetentionResult) -> str:
    """Human report for ``ohho run retain``."""
    count = sum(len(run.deleted) for run in result.runs)
    noun = "checkpoint" if count == 1 else "checkpoints"
    label = "dry-run" if result.dry_run else "deleted"
    lines = [
        f"{label}: {count} {noun}, {result.deleted_bytes} bytes",
        "kept: best checkpoints, last checkpoints, logs, eval reports, "
        "manifest, metrics, pinned, promoted",
    ]
    for run in result.runs:
        for item in run.deleted:
            lines.append(
                f"{item.path}\t{item.size_bytes}\t{item.reason}\t{run.run_dir}"
            )
        for error in run.errors:
            lines.append(f"error: {error}\t{run.run_dir}")
    return "\n".join(lines) + "\n"


def format_storage(usage: ExperimentStorage) -> str:
    """Human report for ``ohho run storage``."""
    label = usage.experiment if usage.experiment is not None else "all"
    lines = [
        f"experiment: {label}",
        f"runs: {len(usage.runs)}",
        f"total_bytes: {usage.total_bytes}",
        f"deleted_bytes: {usage.deleted_bytes}",
    ]
    for run in usage.runs:
        lines.append(
            f"{run.run_dir}\ttotal_bytes={run.total_bytes}\t"
            f"checkpoints_bytes={run.checkpoints_bytes}\t"
            f"logs_bytes={run.logs_bytes}\t"
            f"eval_bytes={run.eval_bytes}\t"
            f"metrics_bytes={run.metrics_bytes}\t"
            f"manifest_bytes={run.manifest_bytes}\t"
            f"other_bytes={run.other_bytes}\t"
            f"deleted_bytes={run.deleted_bytes}"
        )
    return "\n".join(lines) + "\n"


def add_run_commands(subparsers: Any) -> None:
    """Register ``retain`` and ``storage`` on the ``ohho run`` parser."""
    retain = subparsers.add_parser(
        "retain",
        help="apply checkpoint retention to pruned trials",
    )
    retain.add_argument(
        "path",
        help="run directory, manifest file, or a directory of runs",
    )
    retain.add_argument(
        "--dry-run",
        action="store_true",
        help="report deletions without removing files or writing the index",
    )
    retain.add_argument(
        "--days",
        type=int,
        default=DEFAULT_MAX_AGE_DAYS,
        help=f"delete intermediate checkpoints older than this many days "
        f"(default {DEFAULT_MAX_AGE_DAYS})",
    )
    retain.add_argument(
        "--experiment",
        default=None,
        help="only runs in this experiment",
    )
    retain.add_argument(
        "--json",
        action="store_true",
        help="print the retention result as JSON",
    )
    retain.set_defaults(func=_cmd_retain)

    storage = subparsers.add_parser(
        "storage",
        help="print bytes used by a run or an experiment",
    )
    storage.add_argument(
        "path",
        help="run directory, manifest file, or a directory of runs",
    )
    storage.add_argument(
        "--experiment",
        default=None,
        help="only runs in this experiment",
    )
    storage.add_argument(
        "--json",
        action="store_true",
        help="print storage totals as JSON",
    )
    storage.set_defaults(func=_cmd_storage)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry for ``python -m ohho.train.artifacts``."""
    parser = argparse.ArgumentParser(prog="ohho.train.artifacts")
    sub = parser.add_subparsers(dest="command", required=True)
    add_run_commands(sub)
    args = parser.parse_args(None if argv is None else list(argv))
    return int(args.func(args))


def _cmd_retain(args: argparse.Namespace) -> int:
    try:
        policy = RetentionPolicy(max_age_days=args.days)
    except ArtifactError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        result = apply_retention(
            args.path,
            policy=policy,
            dry_run=bool(args.dry_run),
            experiment=args.experiment,
        )
    except ArtifactError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(format_retention(result))
    if result.errors:
        for line in result.errors:
            print(line, file=sys.stderr)
        return 1
    return 0


def _cmd_storage(args: argparse.Namespace) -> int:
    try:
        usage = storage_used_experiment(args.path, experiment=args.experiment)
    except ArtifactError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        sys.stdout.write(json.dumps(usage.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(format_storage(usage))
    return 0


def _usage(run_dir: Path, index: Mapping[str, Any] | None = None) -> StorageUsage:
    if index is None:
        index = load_artifact_index(run_dir)
    totals = {
        "checkpoints": 0,
        "logs": 0,
        "eval": 0,
        "metrics": 0,
        "manifest": 0,
        "other": 0,
    }
    for file in _iter_files(run_dir):
        relative = file.relative_to(run_dir).as_posix()
        try:
            size = file.lstat().st_size
        except OSError:
            continue
        totals[_category(relative)] += size
    deleted = 0
    for entry in index["deletions"]:
        deleted += int(entry["bytes"])
    total = sum(totals.values())
    experiment = index["experiment"]
    if not isinstance(experiment, str):
        experiment = None
    return StorageUsage(
        run_dir=str(run_dir),
        experiment=experiment,
        total_bytes=total,
        checkpoints_bytes=totals["checkpoints"],
        logs_bytes=totals["logs"],
        eval_bytes=totals["eval"],
        metrics_bytes=totals["metrics"],
        manifest_bytes=totals["manifest"],
        other_bytes=totals["other"],
        deleted_bytes=deleted,
    )


def _retain_one(
    run_dir: Path,
    index: dict[str, Any],
    policy: RetentionPolicy,
    moment: datetime,
    *,
    dry_run: bool,
) -> RunRetentionResult:
    decisions = _decide(run_dir, index, policy, moment)
    deleting = [item for item in decisions if item.deleted_at is not None]
    keeping = [item for item in decisions if item.deleted_at is None]
    stamp = _format_time(moment)
    planned = [
        ArtifactDecision(
            path=item.path,
            role=item.role,
            size_bytes=item.size_bytes,
            reason=item.reason,
            deleted_at=stamp,
        )
        for item in deleting
    ]
    probe = _index_after(run_dir, index, planned)
    errors = validate_artifact_index(probe)
    if errors:
        message = "refusing to delete because the artifact index would be invalid: "
        return _result_from_index(
            run_dir,
            index,
            deleted=(),
            kept=tuple(keeping),
            errors=(message + "; ".join(errors),),
        )
    if dry_run:
        return _result_from_index(
            run_dir,
            index,
            deleted=tuple(planned),
            kept=tuple(keeping),
            errors=(),
        )
    deleted: list[ArtifactDecision] = []
    failures: list[str] = []
    for item in planned:
        try:
            _delete_checkpoint(run_dir, item.path)
        except (OSError, ArtifactError) as exc:
            failures.append(f"{item.path}: {exc}")
            continue
        deleted.append(item)
    if deleted:
        updated = _index_after(run_dir, index, deleted)
        try:
            _write_index(run_dir / INDEX_NAME, updated)
        except ArtifactError as exc:
            failures.append(str(exc))
    return _result_from_index(
        run_dir,
        index,
        deleted=tuple(deleted),
        kept=tuple(keeping),
        errors=tuple(failures),
    )


def _decide(
    run_dir: Path,
    index: Mapping[str, Any],
    policy: RetentionPolicy,
    moment: datetime,
) -> list[ArtifactDecision]:
    flags = _flags_by_path(index)
    cutoff = moment - timedelta(days=policy.max_age_days)
    decisions: list[ArtifactDecision] = []
    for path in _discover_checkpoints(run_dir):
        relative = _rel(run_dir, path)
        role = checkpoint_role(path.name)
        pinned, promoted = flags.get(relative, (False, False))
        size = _tree_bytes(path)
        delete, reason = _eligible(
            index=index,
            role=role,
            pinned=pinned,
            promoted=promoted,
            path=path,
            cutoff=cutoff,
            max_age_days=policy.max_age_days,
        )
        decisions.append(
            ArtifactDecision(
                path=relative,
                role=role,
                size_bytes=size,
                reason=reason,
                deleted_at="" if delete else None,
            )
        )
    decisions.sort(key=lambda item: item.path)
    return decisions


def _eligible(
    *,
    index: Mapping[str, Any],
    role: str,
    pinned: bool,
    promoted: bool,
    path: Path,
    cutoff: datetime,
    max_age_days: int,
) -> tuple[bool, str]:
    if index.get("pinned") is True:
        return False, "pinned run"
    if index.get("promoted") is True:
        return False, "promoted run"
    if pinned:
        return False, "pinned checkpoint"
    if promoted:
        return False, "promoted checkpoint"
    if role == "best":
        return False, "best checkpoint"
    if role == "last":
        return False, "last checkpoint"
    if index.get("pruned") is not True:
        return False, "trial is not pruned"
    newest = _newest_mtime(path)
    stamp = None if newest is None else _mtime_to_datetime(newest)
    if stamp is None:
        return False, "unreadable checkpoint time"
    if stamp >= cutoff:
        return False, f"newer than {max_age_days} days"
    return True, f"pruned trial checkpoint older than {max_age_days} days"


def _index_after(
    run_dir: Path,
    index: Mapping[str, Any],
    deleted: Sequence[ArtifactDecision],
) -> dict[str, Any]:
    """Sidecar after ``deleted`` checkpoints are gone.

    A path already listed in ``deletions`` is updated in place so a retry
    does not count those bytes twice.
    """
    gone = {item.path for item in deleted}
    entries = [
        entry
        for entry in _checkpoint_entries(run_dir, index)
        if entry["path"] not in gone
    ]
    by_path: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for entry in index["deletions"]:
        by_path[str(entry["path"])] = dict(entry)
        order.append(str(entry["path"]))
    for item in sorted(deleted, key=lambda entry: entry.path):
        record = {
            "path": item.path,
            "role": item.role,
            "bytes": item.size_bytes,
            "deleted_at": item.deleted_at,
            "reason": item.reason,
        }
        if item.path not in by_path:
            order.append(item.path)
        by_path[item.path] = record
    document = dict(index)
    document["schema"] = SCHEMA_ID
    document["checkpoints"] = entries
    document["deletions"] = [by_path[path] for path in order]
    return _normalize(document)


def _checkpoint_entries(
    run_dir: Path, index: Mapping[str, Any]
) -> list[dict[str, Any]]:
    flags = _flags_by_path(index)
    entries: list[dict[str, Any]] = []
    for path in _discover_checkpoints(run_dir):
        relative = _rel(run_dir, path)
        pinned, promoted = flags.get(relative, (False, False))
        entries.append(
            {
                "path": relative,
                "role": checkpoint_role(path.name),
                "pinned": pinned,
                "promoted": promoted,
            }
        )
    entries.sort(key=lambda item: item["path"])
    return entries


def _flags_by_path(index: Mapping[str, Any]) -> dict[str, tuple[bool, bool]]:
    flags: dict[str, tuple[bool, bool]] = {}
    for entry in index.get("checkpoints", []):
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        flags[path] = (entry.get("pinned") is True, entry.get("promoted") is True)
    return flags


def _discover_checkpoints(run_dir: Path) -> list[Path]:
    root = run_dir / CHECKPOINTS_DIR
    if root.is_symlink() or not root.is_dir():
        return []
    found: list[Path] = []
    for child in root.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_symlink() or child.is_file() or child.is_dir():
            found.append(child)
    found.sort(key=lambda item: item.name)
    return found


def _resolve_checkpoint_key(run_dir: Path, text: str) -> str:
    raw = text.strip().replace("\\", "/")
    parts = PurePosixPath(raw).parts
    if (
        not raw
        or raw.startswith("/")
        or ".." in parts
        or re.match(r"^[A-Za-z]:", raw) is not None
    ):
        raise ArtifactError("checkpoint path must stay inside the run directory")
    discovered = {_rel(run_dir, path) for path in _discover_checkpoints(run_dir)}
    if raw in discovered:
        return raw
    matches = [key for key in discovered if PurePosixPath(key).name == raw]
    if len(matches) == 1:
        return matches[0]
    raise ArtifactError(f"checkpoint not found: {text}")


def _delete_checkpoint(run_dir: Path, relative: str) -> None:
    if relative in _PROTECTED or PurePosixPath(relative).name in _PROTECTED:
        raise ArtifactError(f"refusing to delete {relative}")
    path = _safe_child(run_dir, relative)
    if path is None:
        raise ArtifactError(f"refusing to delete {relative}")
    for name in _PROTECTED:
        guard = run_dir / name
        if not guard.exists() or path.is_symlink() or not path.exists():
            continue
        try:
            same = path.samefile(guard)
        except OSError:
            same = False
        if same:
            raise ArtifactError(f"refusing to delete {relative}")
    if path.is_symlink():
        path.unlink()
        return
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _safe_child(run_dir: Path, relative: str) -> Path | None:
    parts = PurePosixPath(relative).parts
    if len(parts) != 2 or parts[0] != CHECKPOINTS_DIR or ".." in parts:
        return None
    if parts[1].startswith(".") or not parts[1]:
        return None
    candidate = run_dir.joinpath(*parts)
    root_text = os.path.normcase(str(run_dir.resolve()))
    try:
        if candidate.is_symlink():
            parent = os.path.normcase(str(candidate.parent.resolve()))
            Path(parent).relative_to(root_text)
            return candidate
        target = os.path.normcase(str(candidate.resolve()))
        Path(target).relative_to(root_text)
    except (OSError, ValueError):
        return None
    return candidate


def _result_from_index(
    run_dir: Path,
    index: Mapping[str, Any],
    *,
    deleted: tuple[ArtifactDecision, ...],
    kept: tuple[ArtifactDecision, ...],
    errors: tuple[str, ...],
) -> RunRetentionResult:
    experiment = index.get("experiment")
    if not isinstance(experiment, str):
        experiment = None
    return RunRetentionResult(
        run_dir=str(run_dir),
        experiment=experiment,
        pinned=index.get("pinned") is True,
        promoted=index.get("promoted") is True,
        pruned=index.get("pruned") is True,
        deleted=deleted,
        kept=kept,
        errors=errors,
    )


def _error_run(run_dir: Path, message: str) -> RunRetentionResult:
    return RunRetentionResult(
        run_dir=str(run_dir),
        experiment=None,
        pinned=False,
        promoted=False,
        pruned=False,
        deleted=(),
        kept=(),
        errors=(message,),
    )


def _runs_under(path: str | Path) -> tuple[Path, list[Path]]:
    raw = Path(path).expanduser()
    if not raw.exists():
        raise ArtifactError(f"path does not exist: {raw}")
    if raw.is_file():
        if raw.name != MANIFEST_NAME:
            raise ArtifactError(f"not a run directory: {raw}")
        raw = raw.parent
    if not raw.is_dir():
        raise ArtifactError(f"not a run directory: {raw}")
    anchor = raw.resolve()
    if (anchor / MANIFEST_NAME).is_file():
        return anchor, [anchor]
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(anchor):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
        if MANIFEST_NAME in filenames:
            found.append(Path(dirpath).resolve())
            dirnames[:] = []
    found.sort()
    if not found:
        raise ArtifactError(f"no run manifest under {anchor}")
    return anchor, found


def _require_run(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_file() and raw.name == MANIFEST_NAME:
        raw = raw.parent
    if not raw.is_dir() or not (raw / MANIFEST_NAME).is_file():
        raise ArtifactError(f"no run manifest in {raw}")
    return raw.resolve()


def _directory(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_file() and raw.name == MANIFEST_NAME:
        raw = raw.parent
    if not raw.is_dir():
        raise ArtifactError(f"not a run directory: {raw}")
    return raw.resolve()


def _matches(run_dir: Path, index: Mapping[str, Any], experiment: str | None) -> bool:
    if experiment is None:
        return True
    recorded = index.get("experiment")
    if isinstance(recorded, str) and recorded:
        return recorded == experiment
    return run_dir.parent.name == experiment


def _category(relative: str) -> str:
    if relative == MANIFEST_NAME:
        return "manifest"
    if relative == METRICS_NAME:
        return "metrics"
    if relative.startswith(f"{LOGS_DIR}/"):
        return "logs"
    if relative.startswith(f"{EVAL_DIR}/"):
        return "eval"
    if relative.startswith(f"{CHECKPOINTS_DIR}/"):
        return "checkpoints"
    if relative in _ROOT_CHECKPOINT_FILES or relative in _ROOT_CHECKPOINT_DIRS:
        return "checkpoints"
    if relative.startswith("model_final/"):
        return "checkpoints"
    return "other"


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
        for name in filenames:
            files.append(Path(dirpath) / name)
    return files


def _tree_bytes(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        try:
            return path.lstat().st_size
        except OSError:
            return 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
        for name in filenames:
            file = Path(dirpath) / name
            try:
                total += file.lstat().st_size
            except OSError:
                continue
    return total


def _newest_mtime(path: Path) -> float | None:
    try:
        newest = path.lstat().st_mtime
    except OSError:
        return None
    if path.is_symlink() or path.is_file():
        return newest
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
        for name in list(dirnames) + filenames:
            file = Path(dirpath) / name
            try:
                newest = max(newest, file.lstat().st_mtime)
            except OSError:
                continue
    return newest


def _mtime_to_datetime(mtime: float) -> datetime | None:
    try:
        return datetime.fromtimestamp(mtime, timezone.utc).replace(microsecond=0)
    except (OSError, OverflowError, ValueError):
        return None


def _write_index(path: Path, document: Mapping[str, Any]) -> None:
    body = _normalize(document)
    errors = validate_artifact_index(body)
    if errors:
        message = "refusing to write an invalid artifact index: " + "; ".join(errors)
        raise ArtifactError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(body, indent=2) + "\n"
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


def _normalize(document: Mapping[str, Any]) -> dict[str, Any]:
    checkpoints = []
    for entry in document["checkpoints"]:
        checkpoints.append(
            {
                "path": entry["path"],
                "role": entry["role"],
                "pinned": entry["pinned"] is True,
                "promoted": entry["promoted"] is True,
            }
        )
    deletions = []
    for entry in document["deletions"]:
        deletions.append(
            {
                "path": entry["path"],
                "role": entry["role"],
                "bytes": entry["bytes"],
                "deleted_at": entry["deleted_at"],
                "reason": entry["reason"],
            }
        )
    experiment = document["experiment"]
    if experiment is not None and not isinstance(experiment, str):
        experiment = None
    pruned_at = document["pruned_at"]
    if pruned_at is not None and not isinstance(pruned_at, str):
        pruned_at = None
    return {
        "schema": SCHEMA_ID,
        "experiment": experiment,
        "pinned": document["pinned"] is True,
        "promoted": document["promoted"] is True,
        "pruned": document["pruned"] is True,
        "pruned_at": pruned_at,
        "checkpoints": checkpoints,
        "deletions": deletions,
    }


def _check_experiment(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _EXPERIMENT.fullmatch(value) is None:
        raise ArtifactError(
            "experiment must be 1-128 characters of letters, digits, '.', '_' or '-'"
        )
    return value


def _check_bool(name: str, value: bool | None) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise ArtifactError(f"{name} must be true or false")
    return value


def _as_utc(moment: datetime | None) -> datetime:
    if moment is None:
        moment = datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).replace(microsecond=0)


def _format_time(moment: datetime) -> str:
    return _as_utc(moment).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(run_dir: Path, path: Path) -> str:
    return path.relative_to(run_dir).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
