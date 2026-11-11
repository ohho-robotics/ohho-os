"""Persisted sweep runs: sampler state plus the trial table.

``ohho sweep start`` creates ``ohho.sweep.run/v1`` under
``$OHHO_SWEEP_DIR/<id>/run.json`` (default ``~/.ohho/sweeps``).
``ohho sweep resume <id>`` loads that file and proposes the next trials.
Trial indexes already in the table are not proposed again. Each row's
``seed`` is the training seed from :func:`ohho.train.sweep_seed.trial_training_seed`.

``ohho.run/v1`` is not defined here. :func:`ohho.train.sweep_seed.derived_trial_seed`
reads the stored training seed for a later manifest. When the spec has a
dataset dimension, each trial row also stores ``dataset_version_id``.

:meth:`SweepRun.report` records an intermediate metric.
:meth:`SweepRun.decide` returns ``continue`` or ``stop``. A stop marks the
trial ``pruned``, keeps its last checkpoint and metrics, and is distinct
from :meth:`SweepRun.mark_failed`. Resume keeps both statuses. The
controller that fans trials out across jobs is not in this package; it
calls these methods.
"""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .sweep import (
    SCHEMA_ID,
    SweepError,
    SweepSpec,
    Trial,
    _apply,
    _assert_refs,
    _format_set,
    _read_resource,
    _validate,
    dataset_version_for,
    load_sweep,
)
from .sweep_early import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PRUNED,
    EarlyStopping,
    StopDecision,
)
from .sweep_sampler import SAMPLER_SCHEMA_ID, SweepSampler
from .sweep_seed import derived_trial_seed, sampler_seed, trial_training_seed

RUN_SCHEMA_ID = "ohho.sweep.run/v1"
RUN_FILENAME = "sweep.run.v1.json"
RUN_FILE = "run.json"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def default_sweep_root() -> Path:
    """Directory that holds one subdirectory per sweep id.

    ``OHHO_SWEEP_DIR`` overrides ``~/.ohho/sweeps``.
    """

    raw = os.environ.get("OHHO_SWEEP_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".ohho" / "sweeps"


def load_run_schema() -> dict[str, Any]:
    """Return a copy of the ``ohho.sweep.run/v1`` JSON Schema."""

    return copy.deepcopy(_cached_run_schema())


def start_sweep(
    source: str | Path | Mapping[str, Any],
    *,
    sweep_id: str | None = None,
    root: str | Path | None = None,
) -> SweepRun:
    """Validate ``source`` and persist a new run that has not proposed a trial."""

    return SweepRun.start(source, sweep_id=sweep_id, root=root)


def resume_sweep(sweep_id: str, *, root: str | Path | None = None) -> SweepRun:
    """Load a stopped run. The next :meth:`SweepRun.ask` continues it."""

    return SweepRun.resume(sweep_id, root=root)


class SweepRun:
    """One sweep id: the sampler, its saved state, and the trial table."""

    def __init__(
        self,
        sweep_id: str,
        root: Path,
        sampler: SweepSampler,
        trials: list[dict[str, Any]],
    ) -> None:
        self.sweep_id = sweep_id
        self.root = root
        self._sampler = sampler
        self._trials = trials
        self._path = _run_path(root, sweep_id)
        early = sampler.spec.early_stopping
        self._early = None if early is None else EarlyStopping(early)

    @property
    def spec(self) -> SweepSpec:
        """Validated sweep spec stored in the sampler state."""

        return self._sampler.spec

    @property
    def sampler_seed(self) -> int:
        """Sampler seed derived from the spec's sweep seed."""

        return sampler_seed(self.spec.seed)

    @classmethod
    def start(
        cls,
        source: str | Path | Mapping[str, Any],
        *,
        sweep_id: str | None = None,
        root: str | Path | None = None,
    ) -> SweepRun:
        """Create ``root/id/run.json``. Raises if that id already exists."""

        spec = load_sweep(source)
        if spec.schema != SCHEMA_ID:
            raise SweepError(f"sweep schema must be {SCHEMA_ID!r}")
        chosen = _new_id() if sweep_id is None else _check_id(sweep_id)
        directory = _root(root)
        path = _run_path(directory, chosen)
        if path.is_file():
            raise SweepError(f"sweep id already exists: {chosen}")
        sampler = SweepSampler(spec)
        run = cls(chosen, directory, sampler, [])
        _check_consistency(run)
        run.save()
        return run

    @classmethod
    def resume(cls, sweep_id: str, *, root: str | Path | None = None) -> SweepRun:
        """Load a run. Does not propose a trial until :meth:`ask`."""

        directory = _root(root)
        path = _run_path(directory, _check_id(sweep_id))
        if not path.is_file():
            raise SweepError(f"sweep id not found: {sweep_id}")
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SweepError(f"could not read sweep {sweep_id}: {exc}") from exc
        if not isinstance(data, dict):
            raise SweepError("sweep run must be a mapping")
        _check_schema(data, _cached_run_schema())
        if data.get("schema") != RUN_SCHEMA_ID:
            raise SweepError(f"sweep run schema must be {RUN_SCHEMA_ID!r}")
        if data.get("id") != sweep_id:
            raise SweepError("sweep id does not match the saved run")
        try:
            sampler_payload = data["sampler"]
            raw_trials = data["trials"]
            recorded_seed = data["sampler_seed"]
        except KeyError as exc:
            raise SweepError(f"sweep run is missing {exc.args[0]!r}") from exc
        if not isinstance(sampler_payload, Mapping):
            raise SweepError("sweep run sampler must be a mapping")
        if sampler_payload.get("schema") != SAMPLER_SCHEMA_ID:
            raise SweepError(f"sweep run sampler schema must be {SAMPLER_SCHEMA_ID!r}")
        sampler = SweepSampler.load_state(sampler_payload)
        if not isinstance(raw_trials, list):
            raise SweepError("sweep run trials must be an array")
        trials = [_copy_row(row) for row in raw_trials]
        run = cls(sweep_id, directory, sampler, trials)
        if recorded_seed != run.sampler_seed:
            raise SweepError("sampler seed does not match the sweep seed")
        _check_consistency(run)
        return run

    def ask(self, n: int = 1) -> tuple[Trial, ...]:
        """Propose up to ``n`` new trials and persist them.

        Indexes already in the trial table are not returned. A failure leaves
        the saved run unchanged.
        """

        before = self._sampler.dump_state()
        saved_rows = copy.deepcopy(self._trials)
        try:
            trials = self._sampler.ask(n)
            self._append_new(trials)
            self.save()
        except Exception:
            self._sampler = SweepSampler.load_state(before)
            self._trials = saved_rows
            raise
        return trials

    def tell(self, index: int, values: Mapping[str, Any]) -> None:
        """Record objective values for a proposed trial and persist them.

        The trial status becomes ``completed``. A ``pruned`` or ``failed``
        trial is refused, so those statuses stay distinct.
        """

        self._reject_stopped(index)
        before = self._sampler.dump_state()
        saved_rows = copy.deepcopy(self._trials)
        try:
            self._sampler.tell(index, values)
            self._copy_values(index)
            self._set_status(index, STATUS_COMPLETED)
            self.save()
        except Exception:
            self._sampler = SweepSampler.load_state(before)
            self._trials = saved_rows
            raise

    def report(
        self,
        index: int,
        step: int,
        value: int | float,
        *,
        checkpoint: str | None = None,
    ) -> None:
        """Record an intermediate metric and persist the trial row.

        This is the call a trial runner makes at a configured step. It does
        not decide and it does not launch a job. The sweep controller calls
        :meth:`decide` after the reports it wants in the comparison.
        """

        stopper = self._require_early()
        saved_rows = copy.deepcopy(self._trials)
        try:
            stopper.report(self._trials, index, step, value, checkpoint=checkpoint)
            self.save()
        except Exception:
            self._trials = saved_rows
            raise

    def decide(self, index: int) -> StopDecision:
        """Return ``continue`` or ``stop`` and persist a stop as ``pruned``.

        A stopped trial keeps the checkpoint and metrics already stored.
        ``pruned`` is not ``failed``. The same call after resume returns
        stop for a trial that was already pruned.
        """

        stopper = self._require_early()
        saved_rows = copy.deepcopy(self._trials)
        try:
            decision = stopper.decide(self._trials, index)
            self.save()
        except Exception:
            self._trials = saved_rows
            raise
        return decision

    def mark_failed(self, index: int, error: str) -> None:
        """Mark a trial ``failed``, keep its checkpoint and metrics, and persist.

        This is distinct from a prune. A pruned trial cannot be marked
        failed, and a failed trial cannot be marked pruned.
        """

        stopper = self._require_early()
        saved_rows = copy.deepcopy(self._trials)
        try:
            stopper.mark_failed(self._trials, index, error)
            self.save()
        except Exception:
            self._trials = saved_rows
            raise

    def trial_table(self) -> list[dict[str, Any]]:
        """Copy of the trial table. Each row's ``seed`` is the training seed."""

        return copy.deepcopy(self._trials)

    def document(self) -> dict[str, Any]:
        """JSON-ready ``ohho.sweep.run/v1`` document."""

        payload = {
            "schema": RUN_SCHEMA_ID,
            "id": self.sweep_id,
            "sampler_seed": self.sampler_seed,
            "sampler": self._sampler.dump_state(),
            "trials": copy.deepcopy(self._trials),
        }
        _check_schema(payload, _cached_run_schema())
        _check_consistency(self)
        return payload

    def save(self) -> Path:
        """Write the run document atomically. Returns the path."""

        payload = self.document()
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        _atomic_write(self._path, text)
        return self._path

    def _append_new(self, trials: tuple[Trial, ...]) -> None:
        existing = {row["index"] for row in self._trials}
        for trial in trials:
            if trial.index in existing:
                raise SweepError(f"trial {trial.index} already recorded")
            if trial.seed != trial_training_seed(self.spec.seed, trial.index):
                raise SweepError(
                    f"trial {trial.index} seed does not match the sweep seed"
                )
            self._trials.append(_row_from_trial(trial))
            existing.add(trial.index)

    def _copy_values(self, index: int) -> None:
        state_rows = self._sampler.dump_state()["trials"]
        state = next((row for row in state_rows if row["index"] == index), None)
        if state is None:
            raise SweepError(f"trial {index} has not been proposed")
        for row in self._trials:
            if row["index"] == index:
                row["values"] = (
                    None if state["values"] is None else dict(state["values"])
                )
                return
        raise SweepError(f"trial {index} is missing from the trial table")

    def _require_early(self) -> EarlyStopping:
        if self._early is None:
            raise SweepError("sweep has no early_stopping config")
        return self._early

    def _reject_stopped(self, index: int) -> None:
        for row in self._trials:
            if row.get("index") == index and row.get("status") in (
                STATUS_PRUNED,
                STATUS_FAILED,
            ):
                raise SweepError(f"trial {index} is {row['status']}")

    def _set_status(self, index: int, status: str) -> None:
        for row in self._trials:
            if row.get("index") == index:
                row["status"] = status
                return
        raise SweepError(f"trial {index} is missing from the trial table")


def format_start(run: SweepRun) -> str:
    """Text for ``ohho sweep start``."""

    return (
        f"schema: {RUN_SCHEMA_ID}\n"
        f"id: {run.sweep_id}\n"
        f"sampler_seed: {run.sampler_seed}\n"
        f"trials: {len(run.trial_table())}\n"
    )


def format_resume(run: SweepRun, proposed: tuple[Trial, ...]) -> str:
    """Text for ``ohho sweep resume``. ``proposed`` are the new trials only."""

    lines = [
        f"schema: {RUN_SCHEMA_ID}",
        f"id: {run.sweep_id}",
        f"sampler_seed: {run.sampler_seed}",
        f"trials: {len(run.trial_table())}",
        f"proposed: {len(proposed)}",
    ]
    for trial in proposed:
        lines.append("---")
        lines.append(f"trial: {trial.index}")
        lines.append(f"seed: {derived_trial_seed({'seed': trial.seed})}")
        if trial.dataset_version_id is not None:
            lines.append(f"dataset_version_id: {trial.dataset_version_id}")
        lines.append("sets:")
        if trial.sets:
            lines.extend(f"  {item}" for item in trial.sets)
        else:
            lines.append("  (none)")
    return "\n".join(lines) + "\n"


def resume_as_dict(run: SweepRun, proposed: tuple[Trial, ...]) -> dict[str, Any]:
    """JSON for ``ohho sweep resume --json``."""

    return {
        "schema": RUN_SCHEMA_ID,
        "id": run.sweep_id,
        "sampler_seed": run.sampler_seed,
        "trials": len(run.trial_table()),
        "proposed": [_proposed_dict(trial) for trial in proposed],
        "trial_table": [
            {"index": row["index"], "seed": derived_trial_seed(row)}
            for row in run.trial_table()
        ],
    }


def _proposed_dict(trial: Trial) -> dict[str, Any]:
    row = {
        "index": trial.index,
        "seed": trial.seed,
        "overrides": dict(trial.overrides),
        "sets": list(trial.sets),
        "recipe": trial.recipe,
        "estimated_gpu_hours": trial.estimated_gpu_hours,
    }
    if trial.dataset_version_id is not None:
        row["dataset_version_id"] = trial.dataset_version_id
    return row


def _row_from_trial(trial: Trial) -> dict[str, Any]:
    row = {
        "index": trial.index,
        "seed": trial.seed,
        "overrides": copy.deepcopy(trial.overrides),
        "sets": list(trial.sets),
        "recipe": copy.deepcopy(trial.recipe),
        "estimated_gpu_hours": float(trial.estimated_gpu_hours),
        "values": None,
        "status": STATUS_PENDING,
        "checkpoint": None,
        "metrics": [],
        "decision": None,
        "error": None,
    }
    if trial.dataset_version_id is not None:
        row["dataset_version_id"] = trial.dataset_version_id
    return row


def _copy_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise SweepError("sweep trial must be an object")
    values = raw.get("values")
    status = raw.get("status")
    if status is None:
        status = STATUS_COMPLETED if values is not None else STATUS_PENDING
    metrics = raw.get("metrics")
    decision = raw.get("decision", None)
    row = {
        "index": raw.get("index"),
        "seed": raw.get("seed"),
        "overrides": copy.deepcopy(dict(raw.get("overrides") or {})),
        "sets": list(raw.get("sets") or []),
        "recipe": copy.deepcopy(dict(raw.get("recipe") or {})),
        "estimated_gpu_hours": raw.get("estimated_gpu_hours"),
        "values": None if values is None else copy.deepcopy(dict(values)),
        "status": status,
        "checkpoint": raw.get("checkpoint"),
        "metrics": [] if metrics is None else copy.deepcopy(list(metrics)),
        "decision": None if decision is None else copy.deepcopy(dict(decision)),
        "error": raw.get("error"),
    }
    version_id = raw.get("dataset_version_id")
    if version_id is not None:
        row["dataset_version_id"] = version_id
    return row


def _check_consistency(run: SweepRun) -> None:
    state = run._sampler.dump_state()
    rows = state["trials"]
    table = run._trials
    if len(table) != len(rows):
        raise SweepError("trial table does not match the sampler state")
    seen: set[int] = set()
    for row, saved in zip(table, rows):
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise SweepError("trial index must be an integer")
        if index in seen:
            raise SweepError(f"duplicate trial {index}")
        seen.add(index)
        if index != saved["index"]:
            raise SweepError("trial table does not match the sampler state")
        if row.get("seed") != saved.get("seed"):
            raise SweepError(f"trial {index} seed does not match the sampler state")
        if row.get("seed") != trial_training_seed(run.spec.seed, index):
            raise SweepError(f"trial {index} seed does not match the sweep seed")
        derived_trial_seed(row)
        if row.get("overrides") != saved.get("overrides"):
            raise SweepError(f"trial {index} overrides do not match the sampler state")
        if row.get("dataset_version_id") != saved.get("dataset_version_id"):
            raise SweepError(
                f"trial {index} dataset version does not match the sampler state"
            )
        expected_version = dataset_version_for(run.spec, row["overrides"])
        if row.get("dataset_version_id") != expected_version:
            raise SweepError(
                f"trial {index} dataset version does not match its overrides"
            )
        if list(row.get("sets") or []) != list(saved.get("sets") or []):
            raise SweepError(f"trial {index} sets do not match the sampler state")
        if row.get("values") != saved.get("values"):
            raise SweepError(f"trial {index} values do not match the sampler state")
        hours = row.get("estimated_gpu_hours")
        saved_hours = saved.get("estimated_gpu_hours")
        if (
            isinstance(hours, bool)
            or isinstance(saved_hours, bool)
            or not isinstance(hours, (int, float))
            or not isinstance(saved_hours, (int, float))
            or float(hours) != float(saved_hours)
        ):
            raise SweepError(
                f"trial {index} estimated_gpu_hours does not match the sampler state"
            )
        expected_sets = {
            _format_set(path, value) for path, value in row["overrides"].items()
        }
        if set(row["sets"]) != expected_sets or len(row["sets"]) != len(expected_sets):
            raise SweepError(f"trial {index} sets do not match its overrides")
        expected_recipe = _apply(run.spec.recipe, list(row["overrides"].items()))
        if row.get("recipe") != expected_recipe:
            raise SweepError(f"trial {index} recipe does not match its overrides")
    if seen != set(range(len(table))):
        raise SweepError("trial indexes must be 0..n-1")


def _cached_run_schema() -> dict[str, Any]:
    schema = getattr(_cached_run_schema, "_value", None)
    if schema is None:
        schema = json.loads(_read_resource("schemas", RUN_FILENAME))
        _assert_refs(schema, schema, "$")
        _cached_run_schema._value = schema  # type: ignore[attr-defined]
    return schema


def _check_schema(instance: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    _validate(dict(instance), schema, schema, "$", 0)


def _root(root: str | Path | None) -> Path:
    if root is None:
        return default_sweep_root()
    return Path(root)


def _check_id(sweep_id: str) -> str:
    if not isinstance(sweep_id, str) or _ID_RE.fullmatch(sweep_id) is None:
        raise SweepError("sweep id must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    return sweep_id


def _new_id() -> str:
    return "s" + secrets.token_hex(4)


def _run_path(root: Path, sweep_id: str) -> Path:
    _check_id(sweep_id)
    base = root.expanduser().resolve()
    path = (base / sweep_id / RUN_FILE).resolve()
    if path.parent != (base / sweep_id).resolve():
        raise SweepError("sweep id escapes the sweep directory")
    try:
        path.parent.relative_to(base)
    except ValueError as exc:
        raise SweepError("sweep id escapes the sweep directory") from exc
    return path


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


__all__ = [
    "RUN_SCHEMA_ID",
    "SweepRun",
    "default_sweep_root",
    "load_run_schema",
    "start_sweep",
    "resume_sweep",
    "format_start",
    "format_resume",
    "resume_as_dict",
]
