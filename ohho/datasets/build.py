"""Build a dataset version from local LeRobot episodes.

Episodes are read with :class:`ohho.data.reader.DatasetReader` and written
with :func:`ohho.data.writer.write_dataset`. Those modules are imported only
while a build is running.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .model import (
    Augmentation,
    CanonicalEpisode,
    CanonicalFrame,
    DatasetBuildError,
    DatasetSpec,
    DatasetVersion,
    DatasetVersionError,
    Filters,
    SourceQuery,
    Split,
    canonical_content_bytes,
    content_hash_of,
    is_version_id,
    json_number,
    render_number,
    stats_from_episodes,
    storage_prefix_for,
    version_id_for,
)
from .store import create_version, load_version

_SPLIT_NAMES = ("all", "train", "val", "test")
_PARTS = ("train", "val", "test")
_WEIGHT_SCALE = 1_000_000


@dataclass(frozen=True)
class _RawFrame:
    observation_state: tuple[float, ...]
    action: tuple[float, ...]
    action_raw: tuple[float, ...] | None
    timestamp: float
    frame_index: int
    next_done: bool


@dataclass(frozen=True)
class _Loaded:
    source: str
    source_index: int
    task: str
    fps: str
    robot_type: str
    frames: tuple[_RawFrame, ...]


def prepare_spec(spec: DatasetSpec) -> DatasetSpec:
    """Normalise a spec so the same request always hashes to the same id."""
    if not spec.source_query.roots:
        raise DatasetBuildError("source query needs at least one local episode root")
    roots = tuple(sorted({_normalize_root(root) for root in spec.source_query.roots}))
    name = spec.dataset_name.strip() if isinstance(spec.dataset_name, str) else ""
    if not name or len(name) > 128 or "/" in name or "\\" in name:
        raise DatasetBuildError(
            "dataset_name must be a short label without path separators"
        )
    if any(ord(char) < 32 for char in name):
        raise DatasetBuildError("dataset_name must not contain control characters")
    if not isinstance(spec.licence, str) or len(spec.licence) > 512:
        raise DatasetBuildError("licence must be a string of at most 512 characters")
    if not isinstance(spec.seed, int) or isinstance(spec.seed, bool):
        raise DatasetBuildError("seed must be an integer")
    filters = Filters(
        task_contains=_optional_text(spec.filters.task_contains),
        robot_type=_optional_text(spec.filters.robot_type),
        min_frames=_optional_count(spec.filters.min_frames, "min_frames"),
        max_frames=_optional_count(spec.filters.max_frames, "max_frames"),
    )
    if (
        filters.min_frames is not None
        and filters.max_frames is not None
        and filters.min_frames > filters.max_frames
    ):
        raise DatasetBuildError("min_frames cannot exceed max_frames")
    if spec.split.name not in _SPLIT_NAMES:
        raise DatasetBuildError("split name must be all, train, val, or test")
    split = Split(
        name=spec.split.name,
        train=_weight(spec.split.train, "train"),
        val=_weight(spec.split.val, "val"),
        test=_weight(spec.split.test, "test"),
    )
    if split.name != "all" and split.train + split.val + split.test <= 0:
        raise DatasetBuildError("split fractions must sum to a positive number")
    augmentation = _prepare_augmentation(spec.augmentation)
    return DatasetSpec(
        source_query=SourceQuery(roots),
        dataset_name=name,
        filters=filters,
        split=split,
        seed=spec.seed,
        augmentation=augmentation,
        licence=spec.licence,
    )


def build_version(
    spec: DatasetSpec,
    *,
    store: str | Path | None = None,
    parent_version_id: str | None = None,
) -> DatasetVersion:
    """Materialise ``spec`` and return its version record.

    Building the same prepared spec again returns the existing record and
    does not rewrite its files. ``parent_version_id`` is stored only when
    this call creates the version.
    """
    prepared = prepare_spec(spec)
    if parent_version_id is not None and not is_version_id(parent_version_id):
        raise DatasetVersionError("parent_version_id is not a dataset version id")
    selected = _select(prepared)
    robot_type = _robot_type(selected)
    canonical = _augment(selected, prepared.augmentation, prepared.seed)
    content_hash = content_hash_of(canonical)
    version_id = version_id_for(prepared, content_hash)
    record = DatasetVersion(
        version_id=version_id,
        content_hash=content_hash,
        source_query=prepared.source_query,
        filters=prepared.filters,
        split=prepared.split,
        seed=prepared.seed,
        augmentation=prepared.augmentation,
        stats=stats_from_episodes(canonical),
        licence=prepared.licence,
        dataset_name=prepared.dataset_name,
        pinned=False,
        parent_version_id=parent_version_id,
        storage_prefix=storage_prefix_for(version_id),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    def populate(dest: Path) -> None:
        _write_lerobot(dest, canonical, prepared.dataset_name, robot_type)

    return create_version(
        record,
        canonical_content_bytes(canonical),
        populate,
        store=store,
    )


def edit_version(
    version_id: str,
    *,
    store: str | Path | None = None,
    licence: str | None = None,
    seed: int | None = None,
    filters: Filters | None = None,
    split: Split | None = None,
    augmentation: Augmentation | None = None,
    source_query: SourceQuery | None = None,
    dataset_name: str | None = None,
) -> DatasetVersion:
    """Create a new version by changing a spec field.

    The source version is left as it is, including when it is pinned. When
    the change does not alter the version id, the existing record is returned.
    """
    current = load_version(version_id, store=store)
    changes: dict[str, object] = {}
    if licence is not None:
        changes["licence"] = licence
    if seed is not None:
        changes["seed"] = seed
    if filters is not None:
        changes["filters"] = filters
    if split is not None:
        changes["split"] = split
    if augmentation is not None:
        changes["augmentation"] = augmentation
    if source_query is not None:
        changes["source_query"] = source_query
    if dataset_name is not None:
        changes["dataset_name"] = dataset_name
    spec = replace(current.to_spec(), **changes) if changes else current.to_spec()
    return build_version(spec, store=store, parent_version_id=current.version_id)


def _normalize_root(root: str) -> str:
    if not isinstance(root, str) or not root.strip():
        raise DatasetBuildError("episode roots must be paths")
    path = Path(root).expanduser()
    if not path.exists():
        raise DatasetBuildError(f"episode source not found: {root}")
    return path.resolve().as_posix()


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetBuildError("filter text must be a string")
    text = value.strip()
    return text or None


def _optional_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DatasetBuildError(f"{name} must be a non-negative integer")
    return value


def _weight(value: float, name: str) -> float:
    number = json_number(value)
    if number < 0:
        raise DatasetBuildError(f"{name} fraction cannot be negative")
    return number


def _prepare_augmentation(augmentation: Augmentation) -> Augmentation:
    copies = augmentation.copies
    if not isinstance(copies, int) or isinstance(copies, bool) or copies < 1:
        raise DatasetBuildError("augmentation copies must be an integer >= 1")
    noise = {
        "action_noise": json_number(augmentation.action_noise),
        "state_noise": json_number(augmentation.state_noise),
    }
    for name, number in noise.items():
        if number < 0:
            raise DatasetBuildError(f"{name} cannot be negative")
    return Augmentation(
        action_scale=json_number(augmentation.action_scale),
        state_scale=json_number(augmentation.state_scale),
        action_offset=json_number(augmentation.action_offset),
        state_offset=json_number(augmentation.state_offset),
        action_noise=noise["action_noise"],
        state_noise=noise["state_noise"],
        copies=copies,
    )


def _select(spec: DatasetSpec) -> list[_Loaded]:
    loaded = [
        item for item in _load(spec.source_query.roots) if _keep(item, spec.filters)
    ]
    if not loaded:
        raise DatasetBuildError("no episodes matched the dataset spec")
    if spec.split.name == "all":
        return loaded
    ordered = _deterministic_shuffle(loaded, spec.seed)
    counts = _allocation(len(ordered), spec.split)
    start = {
        "train": 0,
        "val": counts["train"],
        "test": counts["train"] + counts["val"],
    }
    name = spec.split.name
    chosen = ordered[start[name] : start[name] + counts[name]]
    if not chosen:
        raise DatasetBuildError(f"split {name!r} selected no episodes")
    return chosen


def _load(roots: Sequence[str]) -> list[_Loaded]:
    from ..data.reader import DatasetReader

    loaded: list[_Loaded] = []
    for dataset in _dataset_dirs(roots):
        reader = DatasetReader(str(dataset))
        fps = render_number(float(reader.info.fps))
        robot_type = reader.robot_type or "sim"
        source = dataset.resolve().as_posix()
        for meta, frames in reader.iter_episodes():
            if not isinstance(meta, dict) or "episode_index" not in meta:
                raise DatasetBuildError(
                    f"episode metadata is missing episode_index in {source}"
                )
            if not frames:
                continue
            loaded.append(
                _Loaded(
                    source=source,
                    source_index=int(meta["episode_index"]),
                    task=_task_of(meta, reader, frames),
                    fps=fps,
                    robot_type=robot_type,
                    frames=tuple(_raw_frame(frame) for frame in frames),
                )
            )
    loaded.sort(key=lambda item: (item.source, item.source_index, item.task))
    return loaded


def _dataset_dirs(roots: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        path = Path(root)
        if (path / "meta" / "info.json").is_file():
            found.append(path)
            continue
        if not path.is_dir():
            raise DatasetBuildError(f"episode source not found: {root}")
        children = sorted(
            (
                child
                for child in path.iterdir()
                if child.is_dir() and (child / "meta" / "info.json").is_file()
            ),
            key=lambda child: child.resolve().as_posix(),
        )
        if not children:
            raise DatasetBuildError(f"no LeRobot dataset under {root}")
        found.extend(children)
    return found


def _task_of(meta: dict, reader, frames) -> str:
    tasks = meta.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    if isinstance(tasks, str):
        return tasks
    return reader.task_text(int(frames[0].task_index))


def _raw_frame(frame) -> _RawFrame:
    raw = frame.action_raw
    return _RawFrame(
        observation_state=tuple(float(value) for value in frame.observation_state),
        action=tuple(float(value) for value in frame.action),
        action_raw=None if raw is None else tuple(float(value) for value in raw),
        timestamp=float(frame.timestamp),
        frame_index=int(frame.frame_index),
        next_done=bool(frame.next_done),
    )


def _keep(item: _Loaded, filters: Filters) -> bool:
    if filters.task_contains is not None and filters.task_contains not in item.task:
        return False
    if filters.robot_type is not None and item.robot_type != filters.robot_type:
        return False
    count = len(item.frames)
    if filters.min_frames is not None and count < filters.min_frames:
        return False
    if filters.max_frames is not None and count > filters.max_frames:
        return False
    return True


def _deterministic_shuffle(items: list[_Loaded], seed: int) -> list[_Loaded]:
    """Fisher-Yates driven only by ``Random.random``, which is stable."""
    out = list(items)
    rng = random.Random(seed)
    for index in range(len(out) - 1, 0, -1):
        swap = int(rng.random() * (index + 1))
        out[index], out[swap] = out[swap], out[index]
    return out


def _allocation(count: int, split: Split) -> dict[str, int]:
    """Hamilton apportionment of ``count`` episodes across train/val/test."""
    weights = {"train": split.train, "val": split.val, "test": split.test}
    total = sum(weights.values())
    if total <= 0:
        raise DatasetBuildError("split fractions must sum to a positive number")
    units = {name: int(round(weights[name] / total * _WEIGHT_SCALE)) for name in _PARTS}
    drift = _WEIGHT_SCALE - sum(units.values())
    order = sorted(_PARTS, key=lambda name: (weights[name] / total, name), reverse=True)
    step = 1 if drift > 0 else -1
    index = 0
    while drift != 0 and index < _WEIGHT_SCALE:
        name = order[index % len(order)]
        if step < 0 and units[name] == 0:
            index += 1
            continue
        units[name] += step
        drift -= step
        index += 1
    if drift != 0:
        raise DatasetBuildError("could not allocate the split")
    quotas = {name: count * units[name] / _WEIGHT_SCALE for name in _PARTS}
    counts = {name: int(quotas[name]) for name in _PARTS}
    leftover = count - sum(counts.values())
    if leftover < 0 or leftover > len(_PARTS):
        raise DatasetBuildError("could not allocate the split")
    ranked = sorted(
        _PARTS,
        key=lambda name: (quotas[name] - counts[name], name),
        reverse=True,
    )
    for slot in range(leftover):
        counts[ranked[slot]] += 1
    return counts


def _robot_type(items: Sequence[_Loaded]) -> str:
    fps = items[0].fps
    state_width = len(items[0].frames[0].observation_state)
    action_width = len(items[0].frames[0].action)
    if state_width == 0 or action_width == 0:
        raise DatasetBuildError("selected episodes have empty state or action")
    robots: set[str] = set()
    for item in items:
        if item.fps != fps:
            raise DatasetBuildError("selected episodes do not share one fps")
        robots.add(item.robot_type)
        for frame in item.frames:
            if (
                len(frame.observation_state) != state_width
                or len(frame.action) != action_width
            ):
                raise DatasetBuildError(
                    "selected episodes do not share state and action width"
                )
            if frame.action_raw is not None and len(frame.action_raw) != action_width:
                raise DatasetBuildError("action_raw width does not match action")
    if len(robots) == 1:
        return next(iter(robots))
    return "mixed"


def _augment(
    items: Sequence[_Loaded], augmentation: Augmentation, seed: int
) -> list[CanonicalEpisode]:
    rendered: list[CanonicalEpisode] = []
    for copy_index in range(augmentation.copies):
        rng = random.Random(seed + copy_index)
        for item in items:
            frames = tuple(
                _augment_frame(frame, augmentation, rng) for frame in item.frames
            )
            rendered.append(
                CanonicalEpisode(task=item.task, fps=item.fps, frames=frames)
            )
    if not rendered:
        raise DatasetBuildError("no episodes matched the dataset spec")
    return rendered


def _augment_frame(
    frame: _RawFrame, augmentation: Augmentation, rng: random.Random
) -> CanonicalFrame:
    state = _transform(
        frame.observation_state,
        scale=augmentation.state_scale,
        offset=augmentation.state_offset,
        noise=augmentation.state_noise,
        rng=rng,
    )
    action, jitters = _transform_with_jitter(
        frame.action,
        scale=augmentation.action_scale,
        offset=augmentation.action_offset,
        noise=augmentation.action_noise,
        rng=rng,
    )
    action_raw = None
    if frame.action_raw is not None:
        action_raw = _apply_jitter(
            frame.action_raw,
            scale=augmentation.action_scale,
            offset=augmentation.action_offset,
            jitters=jitters,
        )
    return CanonicalFrame(
        observation_state=state,
        action=action,
        action_raw=action_raw,
        timestamp=render_number(frame.timestamp),
        frame_index=frame.frame_index,
        next_done=frame.next_done,
    )


def _transform(
    values: Sequence[float],
    *,
    scale: float,
    offset: float,
    noise: float,
    rng: random.Random,
) -> tuple[str, ...]:
    rendered, _jitters = _transform_with_jitter(
        values, scale=scale, offset=offset, noise=noise, rng=rng
    )
    return rendered


def _transform_with_jitter(
    values: Sequence[float],
    *,
    scale: float,
    offset: float,
    noise: float,
    rng: random.Random,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    rendered: list[str] = []
    jitters: list[float] = []
    for value in values:
        jitter = 0.0 if noise == 0.0 else (rng.random() * 2.0 - 1.0) * noise
        jitters.append(jitter)
        rendered.append(render_number(float(value) * scale + offset + jitter))
    return tuple(rendered), tuple(jitters)


def _apply_jitter(
    values: Sequence[float],
    *,
    scale: float,
    offset: float,
    jitters: Sequence[float],
) -> tuple[str, ...]:
    if len(values) != len(jitters):
        raise DatasetBuildError("action_raw width does not match action")
    return tuple(
        render_number(float(value) * scale + offset + jitter)
        for value, jitter in zip(values, jitters)
    )


def _write_lerobot(
    dest: Path,
    episodes: Sequence[CanonicalEpisode],
    dataset_name: str,
    robot_type: str,
) -> None:
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
        repo_id=f"local/{dataset_name}",
        fps=fps,
        tasks=tasks or ["demonstration"],
        robot_type=robot_type,
    )
