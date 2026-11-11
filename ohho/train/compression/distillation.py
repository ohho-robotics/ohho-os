"""Knowledge distillation from a teacher checkpoint into an IL student.

The recipe family is the student. The teacher is a registry model version
or a local tiny CPU checkpoint. This build trains a tiny regressor on CPU
data. A full SmolVLA teacher and a full ACT student need a GPU and are not
run here.

Teacher actions and features are cached by teacher version and dataset
version so a later trial with the same pair does not run the teacher again.
The student run manifest records that teacher version on ``lineage.teacher``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ohho.data.reader import DatasetReader
from ohho.train.recipe_registry import Emit, TrainResult, get_family
from ohho.train.recipe_schema import DatasetUnavailable, RecipeConfigError
from ohho.train.run_manifest import hash_tree, register_lineage_contributor

CACHE_SCHEMA = "ohho.teacher-cache/v1"
TINY_TEACHER_POLICY = "tiny-teacher"
TINY_STUDENT_POLICY = "tiny-student"
BUILTIN_IL_FAMILIES = ("act", "diffusion", "smolvla", "openvla")
LOSSES = ("action_regression", "kl")

_REGRESSOR: type | None = None


class DistillationUnsupported(RecipeConfigError):
    """This distillation path is not available in this build.

    Raised before training. A missing registry client, a non-local registry
    result, and a teacher that is not a tiny CPU checkpoint all use this
    error. No fallback teacher is loaded. Distilling SmolVLA into ACT needs
    a GPU.
    """


class ModelRegistry(Protocol):
    """Client that turns a registry model version into a local checkpoint.

    ``resolve`` returns a mapping with ``checkpoint`` (a local path) and may
    return ``version`` (the canonical teacher version). It must not download
    weights. ohho-os does not ship a client. Install one with
    :func:`set_model_registry`.
    """

    def resolve(self, model: str, version: str) -> Mapping[str, Any]:
        """Return the local checkpoint for ``model`` at ``version``."""


_REGISTRY: ModelRegistry | None = None


def set_model_registry(registry: ModelRegistry | None) -> None:
    """Install the process-local registry client. ``None`` clears it."""
    global _REGISTRY
    _REGISTRY = registry


def get_model_registry() -> ModelRegistry | None:
    """Return the installed registry client, or ``None``."""
    return _REGISTRY


@dataclass(frozen=True)
class DistillationSettings:
    """Knobs from a recipe ``distillation`` block, with defaults filled."""

    loss: str
    mix_weight: float
    temperature: float
    feature_matching: bool
    feature_weight: float
    cache_dir: Path | None


@dataclass(frozen=True)
class ResolvedTeacher:
    """A teacher reference after checkpoint or registry resolution."""

    version: str
    checkpoint: Path
    source: str
    model: str | None

    def as_dict(self) -> dict[str, Any]:
        """Lineage object stored on the student run manifest."""
        return {
            "version": self.version,
            "source": self.source,
            "model": self.model,
            "checkpoint": str(self.checkpoint),
        }


@dataclass(frozen=True)
class LossParts:
    """One distillation step, split so tests can see each term."""

    total: Any
    supervised: Any
    distill: Any
    feature: Any


@dataclass(frozen=True)
class TeacherOutputs:
    """Cached teacher actions and features for one dataset."""

    actions: list[list[float]]
    features: list[list[float]]
    feature_dim: int
    reused: bool
    forward_passes: int
    key: str
    path: Path
    teacher_version: str
    dataset_version: str


def distillation_requested(recipe: Mapping[str, Any]) -> bool:
    """True when the recipe carries a ``distillation`` object."""
    return isinstance(recipe.get("distillation"), dict)


def require_il_student(recipe: Mapping[str, Any], family: Any) -> None:
    """Refuse a student that is not an imitation-learning recipe.

    Built-in IL families are ``act``, ``diffusion``, ``smolvla``, and
    ``openvla``. A plugin family is IL when ``metadata()`` sets
    ``imitation`` to true. The check runs before the availability check, so
    a non-IL family is exit 1 even when that family also has no trainer.
    """
    if _is_il(recipe, family):
        return
    family_id = _family_id(recipe, family)
    names = ", ".join(BUILTIN_IL_FAMILIES)
    if family is None:
        raise RecipeConfigError(
            f"distillation student {family_id!r} is not a registered "
            f"imitation-learning recipe. Built-in IL families: {names}. "
            "A plugin family must register and set metadata imitation to true."
        )
    raise RecipeConfigError(
        "distillation student must be an imitation-learning recipe "
        f"({names}, or a family whose metadata sets imitation to true); "
        f"got {family_id!r}"
    )


def prepare_distillation(recipe: Mapping[str, Any]) -> ResolvedTeacher:
    """Resolve the teacher and refuse a checkpoint this build cannot run.

    Does not train and does not read the dataset. A failure here is a
    preflight error: no ``train_start`` and no run manifest.
    """
    block = recipe.get("distillation")
    if not isinstance(block, dict):
        raise RecipeConfigError("distillation block must be a mapping")
    settings = parse_settings(block)
    if settings.cache_dir is not None and settings.cache_dir.exists():
        if not settings.cache_dir.is_dir():
            raise RecipeConfigError(
                f"distillation.cache_dir is not a directory: {settings.cache_dir}"
            )
    teacher = resolve_teacher(block.get("teacher"))
    _require_tiny_teacher(teacher.checkpoint)
    return teacher


def parse_settings(block: Mapping[str, Any]) -> DistillationSettings:
    """Read loss, mix weight, temperature, and the optional cache directory."""
    loss = block.get("loss", "action_regression")
    if loss not in LOSSES:
        listed = ", ".join(LOSSES)
        raise RecipeConfigError(
            f"distillation.loss must be one of {listed}; got {loss!r}"
        )
    mix = _unit_number(block.get("mix_weight", 0.5), "distillation.mix_weight")
    temperature = block.get("temperature", 1.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise RecipeConfigError(
            "distillation.temperature must be a number greater than 0"
        )
    if float(temperature) <= 0:
        raise RecipeConfigError("distillation.temperature must be greater than 0")
    matching = block.get("feature_matching", False)
    if not isinstance(matching, bool):
        raise RecipeConfigError("distillation.feature_matching must be a boolean")
    weight = block.get("feature_weight", 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
        raise RecipeConfigError(
            "distillation.feature_weight must be a number greater than or equal to 0"
        )
    cache = block.get("cache_dir")
    cache_dir = None
    if cache is not None:
        if not isinstance(cache, str) or not cache.strip():
            raise RecipeConfigError("distillation.cache_dir must be a directory path")
        cache_dir = Path(cache).expanduser()
    return DistillationSettings(
        loss=str(loss),
        mix_weight=mix,
        temperature=float(temperature),
        feature_matching=matching,
        feature_weight=float(weight),
        cache_dir=cache_dir,
    )


def resolve_teacher(ref: Any) -> ResolvedTeacher:
    """Resolve a checkpoint path or a registry model version.

    A checkpoint with no ``version`` uses ``sha256:`` plus the sha256 of
    ``policy.pt``. A registry reference uses the client's version when it
    returns one, otherwise the requested version.
    """
    if not isinstance(ref, dict):
        raise RecipeConfigError(
            "distillation.teacher must be a checkpoint path or a registry model version"
        )
    has_checkpoint = "checkpoint" in ref
    has_registry = "registry" in ref
    if has_checkpoint and has_registry:
        raise RecipeConfigError(
            "distillation.teacher must set checkpoint or registry, not both"
        )
    if has_checkpoint:
        return _resolve_checkpoint(ref)
    if has_registry:
        return _resolve_registry(ref)
    raise RecipeConfigError(
        "distillation.teacher must set a local checkpoint path or a registry "
        "model version"
    )


def teacher_cache_key(teacher_version: str, dataset_version: str) -> str:
    """sha256 hex of the teacher version and the dataset version.

    The two strings are separated by a NUL byte so a boundary in one value
    cannot collide with the other.
    """
    digest = hashlib.sha256()
    digest.update(teacher_version.encode("utf-8"))
    digest.update(b"\0")
    digest.update(dataset_version.encode("utf-8"))
    return digest.hexdigest()


def dataset_version_of(recipe: Mapping[str, Any]) -> str:
    """Dataset identity used in the teacher-output cache key.

    A recipe ``version_id`` becomes ``version:<id>``. A local directory
    becomes ``sha256:`` plus :func:`ohho.train.run_manifest.hash_tree`.
    Training in this build requires the local directory.
    """
    dataset = recipe.get("dataset")
    if not isinstance(dataset, dict):
        raise DatasetUnavailable("recipe dataset is missing")
    version_id = dataset.get("version_id")
    if isinstance(version_id, str) and version_id.strip():
        return "version:" + version_id.strip()
    raw = dataset.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise DatasetUnavailable(
            "distillation needs a local dataset path to cache teacher outputs"
        )
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise DatasetUnavailable(f"dataset path does not exist: {path}")
    return "sha256:" + hash_tree(path)


def default_cache_dir() -> Path:
    """Shared teacher-output cache, reused across trials and output dirs."""
    return Path.home() / ".ohho" / "cache" / "teacher-outputs"


def combine_loss(
    student_actions: Any,
    teacher_actions: Any,
    target_actions: Any,
    student_features: Any,
    teacher_features: Any,
    settings: DistillationSettings,
) -> LossParts:
    """Supervised imitation loss mixed with a teacher loss.

    ``mix_weight`` is the teacher share. ``1 - mix_weight`` stays on mean
    squared error against the dataset actions. ``action_regression`` is mean
    squared error against teacher actions. ``kl`` is ``T^2`` times the KL
    between temperature-scaled softmaxes of the two action vectors.
    Temperature is recorded for action regression and is not applied to it.
    Feature matching adds ``feature_weight`` times the feature MSE.
    """
    torch = _torch()
    mse = torch.nn.functional.mse_loss
    teacher_actions = teacher_actions.detach()
    teacher_features = teacher_features.detach()
    supervised = mse(student_actions, target_actions)
    if settings.loss == "kl":
        distill = _kl(student_actions, teacher_actions, settings.temperature)
    else:
        distill = mse(student_actions, teacher_actions)
    mix = settings.mix_weight
    total = (1.0 - mix) * supervised + mix * distill
    feature = student_actions.new_zeros(())
    if settings.feature_matching:
        if student_features.shape[-1] != teacher_features.shape[-1]:
            raise RecipeConfigError(
                "feature matching needs the student hidden size to equal the "
                f"teacher hidden size ({int(teacher_features.shape[-1])}); "
                f"the student hidden size is {int(student_features.shape[-1])}"
            )
        feature = mse(student_features, teacher_features)
        total = total + settings.feature_weight * feature
    return LossParts(
        total=total, supervised=supervised, distill=distill, feature=feature
    )


def save_tiny_teacher(
    directory: str | Path,
    *,
    state_dim: int,
    action_dim: int,
    hidden: int = 8,
    seed: int = 0,
) -> Path:
    """Write a tiny CPU teacher (``config.json`` plus ``policy.pt``).

    The checkpoint policy name is ``tiny-teacher``. This is the only teacher
    format this build loads.
    """
    if state_dim < 1 or action_dim < 1 or hidden < 1:
        raise RecipeConfigError(
            "tiny teacher state_dim, action_dim, and hidden must be positive"
        )
    torch = _torch()
    out = Path(directory).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = _regressor_cls()(state_dim, action_dim, hidden)
    torch.save(model.state_dict(), out / "policy.pt")
    config = {
        "policy": TINY_TEACHER_POLICY,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden": hidden,
        "seed": seed,
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return out.resolve()


def load_tiny_teacher(directory: str | Path, device: str = "cpu") -> Any:
    """Load a ``tiny-teacher`` checkpoint onto ``device``."""
    checkpoint = _checkpoint_dir(Path(directory))
    config = _read_teacher_config(checkpoint)
    torch = _torch()
    model = _regressor_cls()(
        int(config["state_dim"]),
        int(config["action_dim"]),
        int(config["hidden"]),
    )
    weights = _weights_file(checkpoint)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()
    return model


def train_distilled(recipe: dict[str, Any], emit: Emit) -> TrainResult:
    """Train a tiny student against cached teacher outputs.

    The recipe family is the student. ACT hyperparameters are checked with
    the ACT rules. The weights that are written are a tiny regressor, not
    TinyACT and not SmolVLA. ``train_start`` is emitted after the teacher
    cache is ready and before the student step loop.
    """
    block = recipe.get("distillation")
    if not isinstance(block, dict):
        raise RecipeConfigError("distillation block must be a mapping")
    settings = parse_settings(block)
    teacher = resolve_teacher(block.get("teacher"))
    _require_tiny_teacher(teacher.checkpoint)
    family_id = _family_id(recipe, None)
    plan = _precision_plan(recipe)
    dataset = _student_dataset(recipe, family_id)
    torch = _torch()
    _emit_preamble(recipe, emit)
    states, targets = _load_pairs(dataset)
    if not states:
        raise DatasetUnavailable(f"no recorded frames found in dataset at {dataset}")
    data_version = dataset_version_of(recipe)
    cache = load_teacher_outputs(
        teacher=teacher,
        states=states,
        action_dim=len(targets[0]),
        teacher_version=teacher.version,
        dataset_version=data_version,
        cache_dir=_cache_dir(settings),
    )
    hidden = _student_hidden(recipe)
    if settings.feature_matching and hidden != cache.feature_dim:
        raise RecipeConfigError(
            "feature matching needs the student hidden size to equal the "
            f"teacher hidden size ({cache.feature_dim}); "
            f"hyperparameters.d_model is {hidden}. "
            "Set d_model to the teacher hidden size, or turn feature matching off."
        )
    hyperparameters = recipe.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        hyperparameters = {}
    batch_size = _positive_int(
        hyperparameters.get("batch_size", 8), "hyperparameters.batch_size"
    )
    log_every = _positive_int(
        hyperparameters.get("log_every", 25), "hyperparameters.log_every"
    )
    lr = _positive_float(hyperparameters.get("lr", 1e-4), "hyperparameters.lr")
    train_steps = _train_steps(recipe)
    _apply_student_seed(torch, recipe, plan.device)
    emit(
        {
            "event": "train_start",
            "family": family_id,
            "device": plan.device,
            "precision": plan.precision,
            "train_steps": train_steps,
        }
    )
    try:
        return _fit_student(
            recipe=recipe,
            emit_device=plan.device,
            plan=plan,
            torch=torch,
            states=states,
            targets=targets,
            cache=cache,
            teacher=teacher,
            settings=settings,
            hidden=hidden,
            train_steps=train_steps,
            data_version=data_version,
            batch_size=batch_size,
            log_every=log_every,
            lr=lr,
        )
    except (RecipeConfigError, DatasetUnavailable):
        raise
    except Exception as exc:
        raise _train_failed(exc) from exc


def load_teacher_outputs(
    *,
    teacher: ResolvedTeacher,
    states: Sequence[Sequence[float]],
    action_dim: int,
    teacher_version: str,
    dataset_version: str,
    cache_dir: Path,
) -> TeacherOutputs:
    """Return cached teacher outputs, building the file on a miss.

    A hit does not construct the teacher network and does not call its
    forward. The teacher forward that fills a miss runs on CPU. ``action_dim``
    is the dataset action width.
    """
    if not states:
        raise DatasetUnavailable("distillation needs at least one dataset frame")
    if cache_dir.exists() and not cache_dir.is_dir():
        raise RecipeConfigError(
            f"distillation.cache_dir is not a directory: {cache_dir}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = teacher_cache_key(teacher_version, dataset_version)
    path = cache_dir / f"{key}.json"
    state_dim = len(states[0])
    found = _read_json(path)
    if isinstance(found, dict) and _cache_usable(
        found,
        teacher_version,
        dataset_version,
        len(states),
        state_dim,
        action_dim,
    ):
        return TeacherOutputs(
            actions=found["actions"],
            features=found["features"],
            feature_dim=int(found["feature_dim"]),
            reused=True,
            forward_passes=0,
            key=key,
            path=path,
            teacher_version=teacher_version,
            dataset_version=dataset_version,
        )
    model = load_tiny_teacher(teacher.checkpoint, device="cpu")
    if model.state_dim != state_dim or model.action_dim != action_dim:
        raise RecipeConfigError(
            "teacher checkpoint dims "
            f"(state {model.state_dim}, action {model.action_dim}) "
            f"do not match the dataset (state {state_dim}, action {action_dim})"
        )
    torch = _torch()
    tensor = torch.tensor(list(states), dtype=torch.float32)
    with torch.no_grad():
        actions, features = model(tensor)
    action_rows = _rows(actions)
    feature_rows = _rows(features)
    payload = {
        "schema": CACHE_SCHEMA,
        "teacher_version": teacher_version,
        "dataset_version": dataset_version,
        "state_dim": state_dim,
        "action_dim": model.action_dim,
        "feature_dim": model.hidden,
        "num_frames": len(states),
        "actions": action_rows,
        "features": feature_rows,
    }
    _atomic_json(path, payload)
    return TeacherOutputs(
        actions=action_rows,
        features=feature_rows,
        feature_dim=model.hidden,
        reused=False,
        forward_passes=int(model.forward_passes),
        key=key,
        path=path,
        teacher_version=teacher_version,
        dataset_version=dataset_version,
    )


def teacher_lineage(recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    """Manifest lineage for a distillation recipe, or ``None`` otherwise."""
    if not distillation_requested(recipe):
        return None
    block = recipe.get("distillation")
    teacher = block.get("teacher") if isinstance(block, dict) else None
    return {"teacher": resolve_teacher(teacher).as_dict()}


def _is_il(recipe: Mapping[str, Any], family: Any) -> bool:
    meta = _family_metadata(family)
    if "imitation" in meta:
        return bool(meta["imitation"])
    return _family_id(recipe, family) in BUILTIN_IL_FAMILIES


def _family_metadata(family: Any) -> Mapping[str, Any]:
    if family is None:
        return {}
    meta_fn = getattr(family, "metadata", None)
    if not callable(meta_fn):
        return {}
    meta = meta_fn()
    if isinstance(meta, Mapping):
        return meta
    return {}


def _family_id(recipe: Mapping[str, Any], family: Any) -> str:
    raw = recipe.get("family")
    if isinstance(raw, str) and raw:
        return raw
    family_id = getattr(family, "id", "")
    return family_id if isinstance(family_id, str) else ""


def _unit_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeConfigError(f"{label} must be a number from 0 to 1")
    number = float(value)
    if number < 0 or number > 1:
        raise RecipeConfigError(f"{label} must be from 0 to 1")
    return number


def _resolve_checkpoint(ref: Mapping[str, Any]) -> ResolvedTeacher:
    raw = ref.get("checkpoint")
    if not isinstance(raw, str) or not raw.strip():
        raise RecipeConfigError("distillation.teacher.checkpoint must be a path")
    checkpoint = _checkpoint_dir(Path(raw).expanduser())
    version = ref.get("version")
    if isinstance(version, str) and version.strip():
        version_text = version.strip()
    else:
        version_text = "sha256:" + _sha256_file(_weights_file(checkpoint))
    return ResolvedTeacher(
        version=version_text,
        checkpoint=checkpoint,
        source="checkpoint",
        model=None,
    )


def _resolve_registry(ref: Mapping[str, Any]) -> ResolvedTeacher:
    registry = ref.get("registry")
    if not isinstance(registry, dict):
        raise RecipeConfigError("distillation.teacher.registry must be a mapping")
    model = registry.get("model")
    version = registry.get("version")
    if not isinstance(model, str) or not model.strip():
        raise RecipeConfigError("distillation.teacher.registry.model is required")
    if not isinstance(version, str) or not version.strip():
        raise RecipeConfigError("distillation.teacher.registry.version is required")
    model_id = model.strip()
    requested = version.strip()
    client = get_model_registry()
    if client is None:
        raise DistillationUnsupported(
            "no model registry client is configured for "
            f"{model_id}@{requested}. Set a local "
            "distillation.teacher.checkpoint, or install a registry with "
            "ohho.train.compression.distillation.set_model_registry(). "
            "This build does not download teacher weights."
        )
    try:
        resolved = client.resolve(model_id, requested)
    except RecipeConfigError:
        raise
    except Exception as exc:
        raise RecipeConfigError(
            f"model registry could not resolve {model_id}@{requested}: {exc}"
        ) from exc
    if not isinstance(resolved, Mapping):
        raise RecipeConfigError("model registry resolve() must return a mapping")
    raw_path = resolved.get("checkpoint")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RecipeConfigError(
            "model registry resolve() must return a local checkpoint path"
        )
    if "://" in raw_path or raw_path.startswith("local:"):
        raise DistillationUnsupported(
            "model registry returned a non-local checkpoint "
            f"{raw_path!r}. This build does not download teacher weights."
        )
    checkpoint = _checkpoint_dir(Path(raw_path).expanduser())
    canonical = resolved.get("version", requested)
    if not isinstance(canonical, str) or not canonical.strip():
        canonical_text = requested
    else:
        canonical_text = canonical.strip()
    return ResolvedTeacher(
        version=canonical_text,
        checkpoint=checkpoint,
        source="registry",
        model=model_id,
    )


def _checkpoint_dir(path: Path) -> Path:
    if not path.exists():
        raise RecipeConfigError(f"teacher checkpoint not found: {path}")
    directory = path.parent if path.is_file() else path
    if not directory.is_dir():
        raise RecipeConfigError(f"teacher checkpoint not found: {path}")
    return directory.resolve()


def _weights_file(checkpoint: Path) -> Path:
    weights = checkpoint / "policy.pt"
    if not weights.is_file():
        raise DistillationUnsupported(_not_tiny(checkpoint))
    return weights


def _require_tiny_teacher(checkpoint: Path) -> dict[str, Any]:
    return _read_teacher_config(checkpoint)


def _read_teacher_config(checkpoint: Path) -> dict[str, Any]:
    _weights_file(checkpoint)
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise DistillationUnsupported(_not_tiny(checkpoint))
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DistillationUnsupported(
            _not_tiny(checkpoint) + f" config.json could not be read ({exc})."
        ) from exc
    if not isinstance(config, dict) or config.get("policy") != TINY_TEACHER_POLICY:
        found = config.get("policy") if isinstance(config, dict) else None
        raise DistillationUnsupported(_not_tiny(checkpoint, found))
    for key in ("state_dim", "action_dim", "hidden"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RecipeConfigError(f"teacher config {key} must be a positive integer")
    return config


def _not_tiny(checkpoint: Path, found: Any = None) -> str:
    found_text = ""
    if found not in (None, ""):
        found_text = f" Found policy {found!r}."
    return (
        f"teacher checkpoint {checkpoint} is not a tiny CPU teacher "
        f"(expected config.json policy {TINY_TEACHER_POLICY!r}).{found_text} "
        "Distilling a full model such as SmolVLA into ACT needs a GPU and is "
        "not available in this build. No fallback teacher is loaded."
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kl(student: Any, teacher: Any, temperature: float) -> Any:
    torch = _torch()
    scale = float(temperature)
    log_p = torch.nn.functional.log_softmax(student / scale, dim=-1)
    target = torch.nn.functional.softmax(teacher / scale, dim=-1)
    return torch.nn.functional.kl_div(log_p, target, reduction="batchmean") * (
        scale * scale
    )


def _precision_plan(recipe: Mapping[str, Any]) -> Any:
    from ohho.train.precision import prepare_precision

    return prepare_precision(recipe, family=get_family(_family_id(recipe, None)))


def _student_dataset(recipe: Mapping[str, Any], family_id: str) -> Path:
    from ohho.train.builtin_recipes import _act_kwargs, _dataset_dir

    if family_id == "act":
        _act_kwargs(recipe)
    return _dataset_dir(recipe)


def _apply_student_seed(torch: Any, recipe: Mapping[str, Any], device: str) -> None:
    from ohho.train.builtin_recipes import _apply_seed

    _apply_seed(torch, recipe, device)


def _emit_preamble(recipe: Mapping[str, Any], emit: Emit) -> None:
    weights = recipe.get("base_weights")
    if isinstance(weights, dict) and weights:
        emit(
            {
                "event": "base_weights",
                "status": "not_loaded",
                "uri": weights.get("uri", ""),
                "reason": (
                    "the distillation student in this build starts from a tiny "
                    "CPU regressor and does not load base weights"
                ),
            }
        )
    for hook_type in ("compression_hooks", "eval_hooks"):
        hooks = recipe.get(hook_type) or []
        if not isinstance(hooks, list):
            continue
        for hook in hooks:
            if not isinstance(hook, dict):
                continue
            emit(
                {
                    "event": "hook",
                    "hook_type": hook_type,
                    "name": hook.get("name", ""),
                    "status": "skipped",
                    "reason": "hooks are not executed in this build",
                }
            )


def _load_pairs(dataset: Path) -> tuple[list[list[float]], list[list[float]]]:
    reader = DatasetReader(str(dataset))
    states: list[list[float]] = []
    actions: list[list[float]] = []
    for _meta, frames in reader.iter_episodes():
        for frame in frames:
            states.append([float(value) for value in frame.observation_state])
            actions.append([float(value) for value in frame.action])
    return states, actions


def _cache_dir(settings: DistillationSettings) -> Path:
    if settings.cache_dir is not None:
        return settings.cache_dir
    return default_cache_dir()


def _student_hidden(recipe: Mapping[str, Any]) -> int:
    hyperparameters = recipe.get("hyperparameters")
    if not isinstance(hyperparameters, dict) or "d_model" not in hyperparameters:
        return 8
    value = hyperparameters["d_model"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RecipeConfigError("hyperparameters.d_model must be a positive integer")
    return value


def _train_steps(recipe: Mapping[str, Any]) -> int:
    hyperparameters = recipe.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        hyperparameters = {}
    if "train_steps" in hyperparameters:
        return _positive_int(
            hyperparameters["train_steps"], "hyperparameters.train_steps"
        )
    if "num_epochs" in hyperparameters:
        epochs = _positive_int(
            hyperparameters["num_epochs"], "hyperparameters.num_epochs"
        )
        return epochs * 2
    return 200


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RecipeConfigError(f"{label} must be a positive integer")
    return value


def _fit_student(
    *,
    recipe: Mapping[str, Any],
    emit_device: str,
    plan: Any,
    torch: Any,
    states: list[list[float]],
    targets: list[list[float]],
    cache: TeacherOutputs,
    teacher: ResolvedTeacher,
    settings: DistillationSettings,
    hidden: int,
    train_steps: int,
    data_version: str,
    batch_size: int,
    log_every: int,
    lr: float,
) -> TrainResult:
    from ohho.train.precision import PrecisionSession

    state_dim = len(states[0])
    action_dim = len(targets[0])
    dev = torch.device(plan.device)
    student = _regressor_cls()(state_dim, action_dim, hidden)
    student.to(dev)
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    tensor_s = torch.tensor(states, dtype=torch.float32)
    tensor_a = torch.tensor(targets, dtype=torch.float32)
    tensor_teacher = torch.tensor(cache.actions, dtype=torch.float32)
    tensor_features = torch.tensor(cache.features, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(
        tensor_s, tensor_a, tensor_teacher, tensor_features
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    session = PrecisionSession(plan, torch_module=torch)
    session.begin()
    loss_curve: list[tuple[int, float]] = []
    step = 0
    while step < train_steps:
        for batch_s, batch_a, batch_teacher, batch_features in loader:
            batch_s = batch_s.to(dev)
            batch_a = batch_a.to(dev)
            batch_teacher = batch_teacher.to(dev)
            batch_features = batch_features.to(dev)
            optimizer.zero_grad()
            with session.autocast():
                pred, feats = student(batch_s)
                parts = combine_loss(
                    pred, batch_teacher, batch_a, feats, batch_features, settings
                )
            session.backward(parts.total)
            session.optimizer_step(optimizer)
            if step % log_every == 0 or step == train_steps - 1:
                loss_curve.append(
                    (step, round(float(parts.total.detach().float().item()), 4))
                )
            step += 1
            if step >= train_steps:
                break
    stats = session.end(step)
    out = Path(str(recipe["output"]["dir"])).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), out / "policy.pt")
    final_loss = loss_curve[-1][1] if loss_curve else 0.0
    distillation = {
        "loss": settings.loss,
        "mix_weight": settings.mix_weight,
        "temperature": settings.temperature,
        "feature_matching": settings.feature_matching,
        "feature_weight": settings.feature_weight,
        "teacher_version": teacher.version,
        "dataset_version": data_version,
        "cache_reused": cache.reused,
        "teacher_forward_passes": cache.forward_passes,
        "cache_key": cache.key,
    }
    config = {
        "policy": TINY_STUDENT_POLICY,
        "family": _family_id(recipe, None),
        "mock": False,
        "device": emit_device,
        "precision": plan.precision,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden": hidden,
        "train_steps": train_steps,
        "distillation": distillation,
        "final_loss": final_loss,
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    metrics = {
        "loss_curve": loss_curve,
        "final_loss": final_loss,
        "train_steps": train_steps,
        "precision": plan.precision,
        "loss_scaling": plan.loss_scaling,
        "precision_backend": plan.backend,
        "throughput_steps_per_s": stats["throughput_steps_per_s"],
        "peak_memory_bytes": stats["peak_memory_bytes"],
        "distillation": distillation,
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return TrainResult(
        checkpoint_dir=str(out),
        metrics_path=str(metrics_path),
        device=emit_device,
        base_weights_loaded=False,
    )


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise RecipeConfigError(f"{label} must be a positive number")
    return float(value)


def _train_failed(exc: Exception) -> Exception:
    from ohho.train.recipe_schema import TrainFailed

    return TrainFailed(f"distillation training failed: {exc}")


def _rows(tensor: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in tensor.detach().cpu().tolist()]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        return data
    return None


def _cache_usable(
    record: dict[str, Any] | None,
    teacher_version: str,
    dataset_version: str,
    num_frames: int,
    state_dim: int,
    action_dim: int,
) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("schema") != CACHE_SCHEMA:
        return False
    if record.get("teacher_version") != teacher_version:
        return False
    if record.get("dataset_version") != dataset_version:
        return False
    if record.get("num_frames") != num_frames or record.get("state_dim") != state_dim:
        return False
    if record.get("action_dim") != action_dim:
        return False
    feature_dim = record.get("feature_dim")
    if not isinstance(feature_dim, int) or feature_dim < 1:
        return False
    return _rows_ok(record.get("actions"), num_frames, action_dim) and _rows_ok(
        record.get("features"), num_frames, feature_dim
    )


def _rows_ok(rows: Any, count: int, width: int) -> bool:
    if not isinstance(rows, list) or len(rows) != count:
        return False
    for row in rows:
        if not isinstance(row, list) or len(row) != width:
            return False
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return False
    return True


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _torch() -> Any:
    from ohho.train.builtin_recipes import _import_torch

    try:
        return _import_torch()
    except ImportError as exc:
        raise RecipeConfigError(
            "knowledge distillation needs torch. "
            "Install it with: pip install 'ohho-os[train]'"
        ) from exc


def _regressor_cls() -> type:
    global _REGRESSOR
    if _REGRESSOR is not None:
        return _REGRESSOR
    torch = _torch()

    class TinyRegressor(torch.nn.Module):
        """One hidden layer. The same shape is the tiny teacher and student."""

        def __init__(self, state_dim: int, action_dim: int, hidden: int) -> None:
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden = hidden
            self.encoder = torch.nn.Linear(state_dim, hidden)
            self.head = torch.nn.Linear(hidden, action_dim)
            self.forward_passes = 0

        def forward(self, state: Any) -> tuple[Any, Any]:
            self.forward_passes += 1
            features = torch.tanh(self.encoder(state))
            return self.head(features), features

    _REGRESSOR = TinyRegressor
    return _REGRESSOR


register_lineage_contributor(teacher_lineage)
