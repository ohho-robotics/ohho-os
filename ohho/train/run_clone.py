"""Clone a training run and diff two runs' resolved configs.

``ohho run clone`` copies ``resolved_config`` from an ``ohho.run/v1`` manifest,
applies repeatable dotted ``--set`` overrides, and validates the result as
``ohho.recipe/v1``. It writes a new run directory with ``recipe.json`` and
``run_manifest.json``. The new manifest records ``cloned_from``. Clone does
not start training.

``ohho run diff`` prints :func:`diff_run_configs`. The console uses that
function, so its text and JSON match the CLI.

Exit codes match ``ohho run show``: ``0`` success, ``1`` a missing or invalid
run or an invalid clone, ``2`` a bad command line.
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .recipe_schema import (
    RecipeConfigError,
    apply_defaults,
    apply_overrides,
    validate_recipe,
)
from .run_manifest import (
    MANIFEST_NAME,
    ClonedFrom,
    RunManifest,
    RunManifestError,
    declared_hooks,
    load_run_manifest,
    resolve_manifest_path,
    write_run_manifest,
)

DIFF_SCHEMA_ID = "ohho.run.diff/v1"
RECIPE_NAME = "recipe.json"
OUTPUT_DIR_NOTE = (
    "note: output.dir is the source run; set output.dir before training this recipe"
)

_Op = Literal["add", "remove", "replace"]


class RunConfigError(ValueError):
    """A clone or config diff could not be completed."""

    def __init__(self, message: str, errors: Sequence[str] | None = None) -> None:
        self.summary = message
        self.errors = list(errors or [])
        text = message
        if self.errors:
            text = message + ": " + "; ".join(self.errors)
        super().__init__(text)


@dataclass(frozen=True)
class ConfigChange:
    """One dotted-path change between two resolved configs.

    ``op`` is ``add``, ``remove``, or ``replace``. ``before`` is absent for
    ``add``. ``after`` is absent for ``remove``. Paths use the same dotted
    form as ``--set``, including numeric list indexes.
    """

    path: str
    op: _Op
    before: Any = None
    after: Any = None

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"path": self.path, "op": self.op}
        if self.op != "add":
            body["before"] = self.before
        if self.op != "remove":
            body["after"] = self.after
        return body


@dataclass(frozen=True)
class RunConfigDiff:
    """Config diff of two runs. ``left`` is the before side."""

    left_run: str
    right_run: str
    left_manifest: str
    right_manifest: str
    changes: tuple[ConfigChange, ...]

    @property
    def equal(self) -> bool:
        """True when the resolved configs have no changes."""
        return not self.changes

    def to_dict(self) -> dict[str, Any]:
        """JSON document shared by ``ohho run diff --json`` and the console."""
        return {
            "schema": DIFF_SCHEMA_ID,
            "left": {"run": self.left_run, "manifest": self.left_manifest},
            "right": {"run": self.right_run, "manifest": self.right_manifest},
            "equal": self.equal,
            "changes": [change.to_dict() for change in self.changes],
        }


@dataclass(frozen=True)
class CloneResult:
    """A run directory written by :func:`clone_run`."""

    source_run: str
    source_manifest: str
    dest: str
    recipe_path: str
    manifest_path: str
    config: dict[str, Any]
    manifest: RunManifest
    diff: RunConfigDiff
    note: str | None = None


def cloned_config(
    resolved_config: Mapping[str, Any], overrides: Sequence[str] | None = None
) -> dict[str, Any]:
    """Return ``resolved_config`` with ``--set`` overrides applied.

    Overrides use the same dotted keys and typed values as ``ohho train``.
    The result is validated against ``ohho.recipe/v1``, then recipe defaults
    are filled. A ``null`` value deletes a mapping key.
    """
    if not isinstance(resolved_config, dict):
        raise RunConfigError("resolved config is not an object")
    items = _override_list(overrides)
    try:
        updated = apply_overrides(dict(resolved_config), items)
    except RecipeConfigError as exc:
        raise RunConfigError(exc.summary, list(exc.errors)) from exc
    errors = validate_recipe(updated)
    if errors:
        raise RunConfigError(
            "cloned config is not a valid ohho.recipe/v1 recipe", errors
        )
    try:
        parsed = json.loads(json.dumps(apply_defaults(updated)))
    except (TypeError, ValueError) as exc:
        raise RunConfigError(f"cloned config is not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RunConfigError("cloned config is not an object")
    return parsed


def clone_run(
    run_id: str | Path,
    overrides: Sequence[str] | None = None,
    dest: str | Path | None = None,
) -> CloneResult:
    """Write a new run from ``run_id``'s resolved config.

    ``dest`` defaults to a sibling directory named ``<source>-clone``. The
    directory contains ``recipe.json`` and ``run_manifest.json``. The manifest
    is the source manifest with the tweaked config, the ``--set`` list, and
    ``cloned_from`` pointing at the source run. Execution fields (status,
    seeds, loss, checkpoints) stay as recorded on the source. This function
    does not train.
    """
    source_run, source_manifest, source = _open_run(run_id)
    items = _override_list(overrides)
    config = cloned_config(source.resolved_config, items)
    dest_path = _dest_path(source_run, dest)
    _ensure_dest_available(dest_path, source_run)
    recipe_path = dest_path / RECIPE_NAME
    manifest = _manifest_for_clone(
        source,
        config,
        items,
        recipe_path,
        ClonedFrom(run=str(source_run), manifest=str(source_manifest)),
    )
    created = not dest_path.exists()
    dest_path.mkdir(parents=True, exist_ok=True)
    try:
        _write_json(recipe_path, config)
        manifest_path = write_run_manifest(dest_path / MANIFEST_NAME, manifest)
    except Exception:
        _cleanup(dest_path, created)
        raise
    return CloneResult(
        source_run=str(source_run),
        source_manifest=str(source_manifest),
        dest=str(dest_path),
        recipe_path=str(recipe_path),
        manifest_path=str(manifest_path),
        config=config,
        manifest=manifest,
        diff=diff_run_configs(source_run, dest_path),
        note=_output_note(config, source_run),
    )


def diff_configs(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_run: str = "",
    right_run: str = "",
    left_manifest: str = "",
    right_manifest: str = "",
) -> RunConfigDiff:
    """Diff two resolved config objects. ``left`` is the before side."""
    changes: list[ConfigChange] = []
    _diff_values(left, right, "", changes)
    return RunConfigDiff(
        left_run=left_run,
        right_run=right_run,
        left_manifest=left_manifest,
        right_manifest=right_manifest,
        changes=tuple(changes),
    )


def diff_run_configs(left_id: str | Path, right_id: str | Path) -> RunConfigDiff:
    """Diff ``resolved_config`` for two run directories or manifest paths.

    The console and ``ohho run diff`` both render this value. Changes are
    only inside the resolved configs. Status, loss, and ``cloned_from`` are
    not compared.
    """
    left_run, left_manifest, left = _open_run(left_id)
    right_run, right_manifest, right = _open_run(right_id)
    return diff_configs(
        left.resolved_config,
        right.resolved_config,
        left_run=str(left_run),
        right_run=str(right_run),
        left_manifest=str(left_manifest),
        right_manifest=str(right_manifest),
    )


def format_config_diff(diff: RunConfigDiff) -> str:
    """Human config diff. The change list matches :meth:`RunConfigDiff.to_dict`."""
    lines = [
        DIFF_SCHEMA_ID,
        f"left: {diff.left_run}",
        f"left_manifest: {diff.left_manifest}",
        f"right: {diff.right_run}",
        f"right_manifest: {diff.right_manifest}",
        f"equal: {'true' if diff.equal else 'false'}",
    ]
    for change in diff.changes:
        lines.append(_format_change(change))
    return "\n".join(lines) + "\n"


def format_clone_result(result: CloneResult) -> str:
    """Human summary of a clone, including the config diff against the source."""
    lines = [
        f"cloned: {result.dest}",
        f"recipe: {result.recipe_path}",
        f"manifest: {result.manifest_path}",
        f"cloned_from: {result.source_run}",
    ]
    if result.note:
        lines.append(result.note)
    return "\n".join(lines) + "\n" + format_config_diff(result.diff)


def clone_command(
    run_id: str,
    overrides: Sequence[str] | None = None,
    dest: str | None = None,
) -> int:
    """CLI entry for ``ohho run clone``. Return a process exit code."""
    try:
        result = clone_run(run_id, overrides, dest=dest)
    except (RunConfigError, RunManifestError) as exc:
        _print_error(exc)
        return 1
    sys.stdout.write(format_clone_result(result))
    return 0


def diff_command(left: str, right: str, *, as_json: bool = False) -> int:
    """CLI entry for ``ohho run diff``. Return a process exit code."""
    try:
        diff = diff_run_configs(left, right)
    except (RunConfigError, RunManifestError) as exc:
        _print_error(exc)
        return 1
    if as_json:
        sys.stdout.write(json.dumps(diff.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(format_config_diff(diff))
    return 0


def _manifest_for_clone(
    source: RunManifest,
    config: dict[str, Any],
    overrides: list[str],
    recipe_path: Path,
    cloned_from: ClonedFrom,
) -> RunManifest:
    family = config.get("family")
    device = config.get("device")
    precision = config.get("precision")
    precision_str = (
        precision if isinstance(precision, str) and precision else source.precision
    )
    from .precision import PRECISIONS, backend_for

    loss_scaling, backend = backend_for(precision_str)
    if precision_str == source.precision:
        precision_metrics = dict(source.precision_metrics)
    else:
        precision_metrics = {name: None for name in PRECISIONS}
        precision_metrics[precision_str] = {
            "throughput_steps_per_s": None,
            "peak_memory_bytes": None,
        }
    return RunManifest(
        schema=source.schema,
        status=source.status,
        started_at=source.started_at,
        finished_at=source.finished_at,
        family=family if isinstance(family, str) and family else source.family,
        device=device if isinstance(device, str) and device else source.device,
        precision=precision_str,
        loss_scaling=loss_scaling,
        precision_backend=backend,
        resolved_config=config,
        seeds=source.seeds,
        git=source.git,
        dataset=source.dataset,
        base_weights=source.base_weights,
        container_image_digest=source.container_image_digest,
        library_versions=dict(source.library_versions),
        gpu=source.gpu,
        final_loss=source.final_loss,
        error=source.error,
        recipe_path=str(recipe_path),
        overrides=list(overrides),
        checkpoint_dir=source.checkpoint_dir,
        metrics_path=source.metrics_path,
        declared_hooks=declared_hooks(config),
        precision_metrics=precision_metrics,
        qat=getattr(source, "qat", None),
        cloned_from=cloned_from,
    )


def _open_run(run_id: str | Path) -> tuple[Path, Path, RunManifest]:
    path = resolve_manifest_path(run_id)
    if path is None:
        raise RunConfigError(f"no run manifest in {Path(run_id).expanduser()}")
    manifest = load_run_manifest(path)
    return path.parent.resolve(), path.resolve(), manifest


def _dest_path(source_run: Path, dest: str | Path | None) -> Path:
    if dest is not None and str(dest).strip():
        return Path(dest).expanduser().resolve()
    if not source_run.name:
        raise RunConfigError("pass --dest; the source run has no directory name")
    return source_run.with_name(source_run.name + "-clone")


def _ensure_dest_available(dest: Path, source_run: Path) -> None:
    if dest == source_run:
        raise RunConfigError("destination is the source run")
    if not dest.exists():
        return
    if not dest.is_dir():
        raise RunConfigError(f"destination is not a directory: {dest}")
    if any(dest.iterdir()):
        raise RunConfigError(f"destination already exists: {dest}")


def _output_note(config: Mapping[str, Any], source_run: Path) -> str | None:
    output = config.get("output")
    if not isinstance(output, dict):
        return None
    raw = output.get("dir")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        target = Path(raw).expanduser().resolve()
    except OSError:
        return None
    if target == source_run:
        return OUTPUT_DIR_NOTE
    return None


def _override_list(overrides: Sequence[str] | None) -> list[str]:
    if not overrides:
        return []
    items: list[str] = []
    for item in overrides:
        if not isinstance(item, str):
            raise RunConfigError(f"--set expects k=v, got {item!r}")
        items.append(item)
    return items


def _diff_values(left: Any, right: Any, path: str, out: list[ConfigChange]) -> None:
    if _same(left, right):
        return
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left:
            child = f"{path}.{key}" if path else str(key)
            if key not in right:
                out.append(_change(child, "remove", before=left[key]))
            else:
                _diff_values(left[key], right[key], child, out)
        for key in right:
            if key in left:
                continue
            child = f"{path}.{key}" if path else str(key)
            out.append(_change(child, "add", after=right[key]))
        return
    if isinstance(left, list) and isinstance(right, list):
        for index in range(max(len(left), len(right))):
            child = f"{path}.{index}" if path else str(index)
            if index >= len(left):
                out.append(_change(child, "add", after=right[index]))
            elif index >= len(right):
                out.append(_change(child, "remove", before=left[index]))
            else:
                _diff_values(left[index], right[index], child, out)
        return
    label = path or "$"
    out.append(_change(label, "replace", before=left, after=right))


def _change(
    path: str, op: _Op, *, before: Any = None, after: Any = None
) -> ConfigChange:
    return ConfigChange(
        path=path,
        op=op,
        before=None if op == "add" else copy.deepcopy(before),
        after=None if op == "remove" else copy.deepcopy(after),
    )


def _same(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return False
        return all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same(item, other) for item, other in zip(left, right)
        )
    return left == right


def _format_change(change: ConfigChange) -> str:
    if change.op == "add":
        return f"+ {change.path}: {_json_value(change.after)}"
    if change.op == "remove":
        return f"- {change.path}: {_json_value(change.before)}"
    before = _json_value(change.before)
    after = _json_value(change.after)
    return f"~ {change.path}: {before} -> {after}"


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(", ", ": "))


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
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


def _cleanup(dest: Path, created: bool) -> None:
    for name in (
        RECIPE_NAME,
        MANIFEST_NAME,
        f"{RECIPE_NAME}.tmp",
        f"{MANIFEST_NAME}.tmp",
    ):
        try:
            (dest / name).unlink(missing_ok=True)
        except OSError:
            pass
    if not created:
        return
    try:
        dest.rmdir()
    except OSError:
        pass


def _print_error(exc: Exception) -> None:
    errors = [str(item) for item in getattr(exc, "errors", []) or []]
    summary = getattr(exc, "summary", None)
    if summary is None:
        text = str(exc)
        if errors and ": " in text:
            summary = text.split(": ", 1)[0]
        else:
            summary = text
    print(summary, file=sys.stderr)
    for line in errors:
        if line != summary:
            print(line, file=sys.stderr)
