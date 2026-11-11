"""Optional W&B sweep backend for ``ohho.sweep/v1``.

:func:`to_wandb_sweep_config` is a pure translation. It does not import
``wandb`` and it does not build a sweep URL.

:func:`launch_sweep` chooses a backend:

* ``native`` (the default) persists an ``ohho.sweep.run/v1`` run and
  returns no URL.
* ``wandb`` creates the sweep through the W&B API when an API key is
  configured and returns the absolute URL that client returned.
* ``auto`` uses ``wandb`` only when that key is configured. Otherwise it
  uses ``native``.

The sweep command runs ``python -m ohho.train.sweep_agent``. Each agent
trial is one OhhO train job. ``wandb`` is imported only inside the real
client, and only when a launch is about to call the API.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .sweep import (
    SCHEMA_ID,
    SweepError,
    SweepSpec,
    _grid_levels,
    load_sweep,
)
from .sweep_run import SweepRun, format_start, start_sweep

WANDB_METHOD = {
    "grid": "grid",
    "random": "random",
    "tpe": "bayes",
    "bayesian": "bayes",
    "multi": "bayes",
}
OHHO_TRAIN_AGENT_MODULE = "ohho.train.sweep_agent"
OHHO_TRAIN_AGENT_COMMAND = (
    "${env}",
    "${interpreter}",
    "-m",
    OHHO_TRAIN_AGENT_MODULE,
    "${args}",
)
_BACKENDS = ("native", "wandb", "auto")
_OFFLINE_MODES = frozenset({"disabled", "offline"})
_METHOD_NOTES = {
    "tpe": (
        "ohho strategy tpe is the built-in TPE sampler; "
        "W&B method bayes is W&B's search"
    ),
    "bayesian": (
        "ohho strategy bayesian is the built-in TPE sampler; "
        "W&B method bayes is W&B's search"
    ),
    "multi": (
        "ohho strategy multi is multi-objective TPE; "
        "W&B method bayes optimizes one metric"
    ),
}


class WandbUnavailable(SweepError):
    """The W&B package is not installed, or the client cannot be used."""


@dataclass(frozen=True)
class WandbSweepTranslation:
    """W&B sweep body for one ``ohho.sweep/v1`` spec.

    ``early_terminate`` is the Hyperband block, or ``None`` when the spec's
    early-stopping fields are not a Hyperband schedule. ``config`` never
    contains a URL.
    """

    config: dict[str, Any]
    notes: tuple[str, ...]
    early_terminate: dict[str, Any] | None
    agent_count: int | None


@dataclass(frozen=True)
class WandbSweepCreated:
    """Id and absolute URL returned by a W&B client."""

    sweep_id: str
    url: str


@dataclass(frozen=True)
class SweepLaunch:
    """Result of starting a sweep on one backend.

    ``url`` is set only when the W&B client returned an absolute URL.
    """

    backend: str
    sweep_id: str
    url: str | None
    config: dict[str, Any] | None
    notes: tuple[str, ...]
    agent_count: int | None
    project: str | None
    entity: str | None
    run: SweepRun | None


class WandbSweepCreator(Protocol):
    """Creates one sweep and returns the id and URL the service assigned."""

    def create(
        self,
        config: Mapping[str, Any],
        *,
        entity: str | None,
        project: str,
    ) -> WandbSweepCreated:
        """Create ``config`` in ``project`` and return the client's URL."""


def to_wandb_sweep_config(
    source: str | os.PathLike[str] | Mapping[str, Any],
) -> WandbSweepTranslation:
    """Translate a validated ``ohho.sweep/v1`` spec into a W&B sweep config.

    The result's ``config`` is the mapping a W&B client accepts. This
    function does not contact W&B and does not invent a sweep URL.
    """

    spec = load_sweep(source)
    if spec.schema != SCHEMA_ID:
        raise SweepError(f"sweep schema must be {SCHEMA_ID!r}")
    method = WANDB_METHOD.get(spec.strategy)
    if method is None:
        raise SweepError(f"unknown strategy {spec.strategy!r}")
    notes: list[str] = []
    method_note = _METHOD_NOTES.get(spec.strategy)
    if method_note is not None:
        notes.append(method_note)
    metric, metric_notes = _metric(spec)
    notes.extend(metric_notes)
    parameters = _parameters(spec, method, notes)
    # TODO(OHH-520): map the early-stopping module onto W&B early_terminate
    # when that module exists. ohho.sweep/v1 early_stopping is patience,
    # min_delta, and mode. W&B early_terminate is Hyperband (type, min_iter,
    # eta, s, max_iter). Those fields do not correspond, so this leaves the
    # block unset instead of inventing a Hyperband schedule.
    early_terminate = _early_terminate(spec)
    if spec.early_stopping is not None:
        early = spec.early_stopping
        notes.append(
            "TODO(OHH-520): early_stopping "
            f"(metric={early['metric']}, patience={early['patience']}, "
            f"min_delta={json.dumps(early['min_delta'])}, mode={early['mode']}) "
            "is not mapped to W&B early_terminate. "
            "W&B early_terminate is Hyperband and those fields are not "
            "a Hyperband schedule."
        )
    _budget_notes(spec, notes)
    config: dict[str, Any] = {
        "program": OHHO_TRAIN_AGENT_MODULE,
        "command": list(OHHO_TRAIN_AGENT_COMMAND),
        "method": method,
        "parameters": parameters,
        "description": (
            "OhhO training sweep. Each agent trial is one OhhO train job "
            f"(python -m {OHHO_TRAIN_AGENT_MODULE})."
        ),
    }
    if metric is not None:
        config["metric"] = metric
    if early_terminate is not None:
        config["early_terminate"] = early_terminate
    agent_count = spec.budget.get("max_trials")
    return WandbSweepTranslation(
        config=config,
        notes=tuple(notes),
        early_terminate=early_terminate,
        agent_count=None if agent_count is None else int(agent_count),
    )


def wandb_connection_configured(
    *,
    api_key: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """True when a W&B API key is set and W&B is not disabled or offline.

    The key is not returned and not logged. ``api_key`` overrides
    ``WANDB_API_KEY`` when it is passed, including an empty string.
    """

    return _connection_problem(api_key, _env(environ), for_error=False) is None


def launch_sweep(
    source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    backend: str = "native",
    sweep_id: str | None = None,
    root: str | os.PathLike[str] | None = None,
    project: str | None = None,
    entity: str | None = None,
    api_key: str | None = None,
    environ: Mapping[str, str] | None = None,
    client: WandbSweepCreator | None = None,
    importer: Any = None,
) -> SweepLaunch:
    """Start ``source`` on ``backend``.

    ``native`` writes a local run and sets ``url`` to ``None``. ``wandb``
    calls ``client`` (or the real W&B client) and copies the absolute URL
    that call returned. A missing key, a missing project, or a client that
    does not return an absolute URL raises :class:`SweepError` and does not
    substitute a URL. ``auto`` uses native when the key is unset or the
    ``wandb`` package cannot be imported. ``importer`` replaces the loader
    used for that import.
    """

    if backend not in _BACKENDS:
        raise SweepError(f"unknown sweep backend {backend!r}")
    env = _env(environ)
    problem = _connection_problem(api_key, env, for_error=backend == "wandb")
    if backend == "native" or (backend == "auto" and problem is not None):
        notes = () if backend == "native" or problem is None else (problem,)
        return _native_result(source, sweep_id=sweep_id, root=root, notes=notes)
    if problem is not None:
        raise SweepError(problem)
    if client is None:
        try:
            _load_wandb(None, importer=importer)
        except WandbUnavailable:
            if backend != "auto":
                raise
            return _native_result(
                source,
                sweep_id=sweep_id,
                root=root,
                notes=(
                    "the wandb package is not installed, so the native backend is used",
                ),
            )
    if sweep_id is not None:
        raise SweepError(
            "W&B assigns the sweep id. Omit the local sweep id. "
            "No sweep URL was created."
        )
    translation = to_wandb_sweep_config(source)
    if translation.config["method"] == "bayes" and "metric" not in translation.config:
        raise SweepError(
            "W&B bayes needs one metric. Set a primary objective. "
            "No sweep URL was created."
        )
    chosen_project = _name("project", _project(project, env))
    chosen_entity = _optional_name("entity", _entity(entity, env))
    creator = client if client is not None else WandbApiClient()
    sent = copy.deepcopy(translation.config)
    try:
        if isinstance(creator, WandbApiClient):
            created = creator.create(
                sent,
                entity=chosen_entity,
                project=chosen_project,
                api_key=_chosen_key(api_key, env),
                importer=importer,
            )
        else:
            created = creator.create(sent, entity=chosen_entity, project=chosen_project)
    except SweepError:
        raise
    except Exception as exc:
        raise SweepError(
            f"W&B sweep API call failed ({type(exc).__name__}). "
            "No sweep URL was created."
        ) from exc
    url = _absolute_url(created.url)
    if not isinstance(created.sweep_id, str) or created.sweep_id == "":
        raise SweepError(
            "W&B client did not return a sweep id. No sweep URL was created."
        )
    return SweepLaunch(
        backend="wandb",
        sweep_id=created.sweep_id,
        url=url,
        config=copy.deepcopy(translation.config),
        notes=translation.notes,
        agent_count=translation.agent_count,
        project=chosen_project,
        entity=chosen_entity,
        run=None,
    )


class WandbApiClient:
    """W&B API client. ``wandb`` is imported here, not at module import.

    The URL written on the result is the ``url`` attribute of the sweep
    object the API returns. This class does not format a host, entity, or
    sweep id into a URL.
    """

    def create(
        self,
        config: Mapping[str, Any],
        *,
        entity: str | None,
        project: str,
        api_key: str | None = None,
        wandb_module: Any = None,
        importer: Any = None,
    ) -> WandbSweepCreated:
        """Create a sweep and return the id and URL the client exposes."""

        wandb = _load_wandb(wandb_module, importer=importer)
        previous = os.environ.get("WANDB_API_KEY")
        injected = False
        try:
            if api_key and previous != api_key:
                os.environ["WANDB_API_KEY"] = api_key
                injected = True
            sweep_id = wandb.sweep(dict(config), entity=entity, project=project)
            if not isinstance(sweep_id, str) or sweep_id == "":
                raise SweepError(
                    "W&B sweep API did not return a sweep id. No sweep URL was created."
                )
            ident = sweep_id
            if "://" not in sweep_id and "/" not in sweep_id:
                if entity:
                    ident = f"{entity}/{project}/{sweep_id}"
                else:
                    ident = f"{project}/{sweep_id}"
            found = wandb.Api().sweep(ident)
        except SweepError:
            raise
        except Exception as exc:
            raise SweepError(
                f"W&B sweep API call failed ({type(exc).__name__}). "
                "No sweep URL was created."
            ) from exc
        finally:
            if injected:
                if previous is None:
                    os.environ.pop("WANDB_API_KEY", None)
                else:
                    os.environ["WANDB_API_KEY"] = previous
        url = getattr(found, "url", None)
        return WandbSweepCreated(sweep_id=sweep_id, url=_absolute_url(url))


def _native_result(
    source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    sweep_id: str | None,
    root: str | os.PathLike[str] | None,
    notes: tuple[str, ...],
) -> SweepLaunch:
    run = start_sweep(source, sweep_id=sweep_id, root=root)
    return SweepLaunch(
        backend="native",
        sweep_id=run.sweep_id,
        url=None,
        config=None,
        notes=notes,
        agent_count=None,
        project=None,
        entity=None,
        run=run,
    )


def format_launch(result: SweepLaunch) -> str:
    """Text for ``ohho sweep start``. Native output has no URL line."""

    if result.backend == "native":
        if result.run is None:
            raise SweepError("native sweep launch is missing its run")
        return format_start(result.run)
    url = _absolute_url(result.url)
    lines = [
        "backend: wandb",
        f"sweep_id: {result.sweep_id}",
        f"url: {url}",
        f"agent: python -m {OHHO_TRAIN_AGENT_MODULE}",
    ]
    if result.project:
        lines.append(f"project: {result.project}")
    if result.entity:
        lines.append(f"entity: {result.entity}")
    if result.agent_count is not None:
        lines.append(f"agent_count: {result.agent_count}")
    lines.extend(f"note: {note}" for note in result.notes)
    return "\n".join(lines) + "\n"


def launch_as_dict(result: SweepLaunch) -> dict[str, Any]:
    """JSON for a launch. The native payload is the run document, with no URL."""

    if result.backend == "native":
        if result.run is None:
            raise SweepError("native sweep launch is missing its run")
        payload = result.run.document()
        if "url" in payload:
            raise SweepError("native sweep run must not carry a URL")
        return payload
    url = _absolute_url(result.url)
    return {
        "backend": "wandb",
        "sweep_id": result.sweep_id,
        "url": url,
        "project": result.project,
        "entity": result.entity,
        "agent": f"python -m {OHHO_TRAIN_AGENT_MODULE}",
        "agent_count": result.agent_count,
        "notes": list(result.notes),
        "config": result.config,
    }


def _early_terminate(spec: SweepSpec) -> dict[str, Any] | None:
    """Return a W&B ``early_terminate`` block, or ``None`` when unmapped.

    TODO(OHH-520): the early-stopping module is not in this tree. When it
    exists, map its config here. Patience, ``min_delta``, and ``mode`` are
    not Hyperband inputs, so a spec that only has those fields stays
    unmapped.
    """

    del spec
    return None


def _metric(spec: SweepSpec) -> tuple[dict[str, str] | None, tuple[str, ...]]:
    objectives = spec.objectives
    if not objectives:
        return None, ("no objective is set, so the W&B config has no metric",)
    primary = [obj for obj in objectives if obj.get("primary")]
    notes: list[str] = []
    if len(objectives) == 1:
        chosen = objectives[0]
    elif primary:
        chosen = primary[0]
        notes.append(
            "W&B metric is the primary objective; "
            "other objectives are not the W&B metric"
        )
    else:
        return None, ("W&B has one metric and this sweep has no primary objective",)
    goal = "minimize" if chosen["direction"] == "minimize" else "maximize"
    return {"name": str(chosen["name"]), "goal": goal}, tuple(notes)


def _parameters(spec: SweepSpec, method: str, notes: list[str]) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for dim in spec.dimensions:
        parameters[dim.path] = _parameter(dim, method)
        if dim.when is not None:
            notes.append(
                f"dimension {dim.name} is conditional (when) and "
                "W&B samples it on every trial"
            )
    if spec.constraints:
        notes.append(
            "constraints are not sent to W&B, so invalid combinations are not pruned"
        )
    leaves, skipped = _recipe_leaves(spec.recipe)
    for path in sorted(leaves):
        if path not in parameters:
            parameters[path] = {"value": leaves[path]}
    for path, reason in sorted(skipped):
        notes.append(f"recipe field {path} {reason} and is not a W&B parameter")
    return parameters


def _parameter(dim: Any, method: str) -> dict[str, Any]:
    if _discrete(dim, method):
        levels = list(_grid_levels(dim))
        return {"values": levels}
    if dim.low is None or dim.high is None:
        raise SweepError(f"dimension {dim.name}: nothing to map")
    if dim.type == "int":
        return {
            "distribution": "int_uniform",
            "min": int(dim.low),
            "max": int(dim.high),
        }
    if dim.type == "float":
        return {
            "distribution": "uniform",
            "min": float(dim.low),
            "max": float(dim.high),
        }
    if dim.type == "log_uniform":
        return {
            "distribution": "log_uniform_values",
            "min": float(dim.low),
            "max": float(dim.high),
        }
    raise SweepError(f"dimension {dim.name}: unknown type {dim.type}")


def _discrete(dim: Any, method: str) -> bool:
    if dim.values is not None or dim.n is not None or dim.type == "categorical":
        return True
    if dim.type == "int":
        step = 1 if dim.step is None else int(dim.step)
        return method == "grid" or step != 1
    return False


def _recipe_leaves(
    recipe: Mapping[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    leaves: dict[str, Any] = {}
    skipped: list[tuple[str, str]] = []

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            if prefix and not value:
                skipped.append((prefix, "is not a scalar"))
                return
            for key in value:
                child = f"{prefix}.{key}" if prefix else str(key)
                walk(value[key], child)
            return
        if not prefix:
            return
        if value is None:
            skipped.append((prefix, "is null"))
            return
        if isinstance(value, bool) or isinstance(value, (str, int)):
            leaves[prefix] = value
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                skipped.append((prefix, "is not a finite number"))
                return
            leaves[prefix] = value
            return
        skipped.append((prefix, "is not a scalar"))

    walk(dict(recipe), "")
    return leaves, skipped


def _budget_notes(spec: SweepSpec, notes: list[str]) -> None:
    if spec.strategy != "grid":
        notes.append("the sweep seed is not part of the W&B config")
    if "max_trials" in spec.budget:
        notes.append(
            "budget.max_trials is not a W&B field; agent_count carries that cap"
        )
    if "max_gpu_hours" in spec.budget:
        notes.append("budget.max_gpu_hours is not sent to W&B")
    if spec.parallelism != 1:
        notes.append("parallelism is not a W&B field")


def _env(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    if environ is None:
        return os.environ
    return environ


def _connection_problem(
    api_key: str | None, env: Mapping[str, str], *, for_error: bool
) -> str | None:
    mode = str(env.get("WANDB_MODE", "")).strip().lower()
    if mode in _OFFLINE_MODES:
        if for_error:
            return "W&B mode is disabled or offline. No sweep URL was created."
        return "W&B mode is disabled or offline, so the native backend is used"
    key = api_key if api_key is not None else env.get("WANDB_API_KEY", "")
    if not isinstance(key, str) or not key.strip():
        if for_error:
            return "W&B sweep backend needs WANDB_API_KEY. No sweep URL was created."
        return "W&B is not configured, so the native backend is used"
    return None


def _chosen_key(api_key: str | None, env: Mapping[str, str]) -> str | None:
    if api_key is not None:
        return api_key if api_key.strip() else None
    raw = env.get("WANDB_API_KEY", "")
    return raw if isinstance(raw, str) and raw.strip() else None


def _project(project: str | None, env: Mapping[str, str]) -> str:
    chosen = project if project is not None else env.get("WANDB_PROJECT", "")
    if not isinstance(chosen, str) or not chosen.strip():
        raise SweepError(
            "W&B sweep backend needs a project (WANDB_PROJECT or project). "
            "No sweep URL was created."
        )
    return chosen.strip()


def _entity(entity: str | None, env: Mapping[str, str]) -> str | None:
    chosen = entity if entity is not None else env.get("WANDB_ENTITY")
    if chosen is None:
        return None
    if not isinstance(chosen, str) or not chosen.strip():
        return None
    return chosen.strip()


def _optional_name(label: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _name(label, value)


def _name(label: str, value: str) -> str:
    if (
        not value
        or value.strip() != value
        or "/" in value
        or "\\" in value
        or "://" in value
        or any(char.isspace() for char in value)
    ):
        raise SweepError(
            f"W&B {label} must be a single name. No sweep URL was created."
        )
    return value


def _absolute_url(url: Any) -> str:
    """Return ``url`` unchanged when it is absolute. Never build a replacement."""

    if isinstance(url, str) and "://" in url:
        return url
    raise SweepError(
        "W&B client did not return an absolute sweep URL. No sweep URL was created."
    )


def _load_wandb(module: Any, *, importer: Any = None) -> Any:
    if module is not None:
        return module
    load = importer if importer is not None else __import__
    try:
        if importer is None:
            import wandb
        else:
            wandb = load("wandb")
    except ImportError as exc:
        raise WandbUnavailable(
            "W&B sweep backend needs the wandb package: "
            "pip install 'ohho-os[wandb]'. No sweep URL was created."
        ) from exc
    return wandb


__all__ = [
    "WANDB_METHOD",
    "OHHO_TRAIN_AGENT_MODULE",
    "OHHO_TRAIN_AGENT_COMMAND",
    "WandbUnavailable",
    "WandbSweepTranslation",
    "WandbSweepCreated",
    "SweepLaunch",
    "WandbSweepCreator",
    "to_wandb_sweep_config",
    "wandb_connection_configured",
    "launch_sweep",
    "WandbApiClient",
    "format_launch",
    "launch_as_dict",
]
