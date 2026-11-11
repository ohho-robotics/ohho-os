"""Seeded episode splits for a dataset version.

Episodes move as a whole. Frames of one episode are never placed in two
splits. ``strategy="episode"`` shuffles episodes. ``session``, ``task``, and
``scene`` shuffle groups, so a session, task, or scene stays together.
``held_out_scenes`` are appended to test and are left out of the shuffle.

The shuffle is Fisher-Yates driven only by :meth:`random.Random.random`.
Group counts use the same Hamilton apportionment as the episode split.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from .model import DatasetBuildError, Split, json_number

SPLIT_NAMES = ("all", "train", "val", "test")
SPLIT_STRATEGIES = ("episode", "session", "task", "scene")
_PARTS = ("train", "val", "test")
_WEIGHT_SCALE = 1_000_000
_UNIX_TIMESTAMP_MIN = 1_000_000_000


@dataclass(frozen=True)
class SplitEpisode:
    """One episode the splitter assigns as a whole.

    ``frame_ids`` travel with the episode. The splitter does not read them
    to decide the split; callers use them to check that no frame was divided.
    """

    episode_id: str
    task: str = ""
    session: str | None = None
    day: str | None = None
    scene: str | None = None
    timestamp: float | None = None
    frame_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitAssignment:
    """Train, val, and test episodes in materialisation order."""

    train: tuple[SplitEpisode, ...]
    val: tuple[SplitEpisode, ...]
    test: tuple[SplitEpisode, ...]

    def episodes(self, name: str) -> tuple[SplitEpisode, ...]:
        """Return the episodes assigned to ``train``, ``val``, or ``test``."""
        if name == "train":
            return self.train
        if name == "val":
            return self.val
        if name == "test":
            return self.test
        raise DatasetBuildError("split name must be train, val, or test")


@dataclass(frozen=True)
class _Group:
    key: str
    episodes: tuple[SplitEpisode, ...]


def prepare_split(split: Split) -> Split:
    """Normalise a split so the same request always hashes the same way."""
    if split.name not in SPLIT_NAMES:
        raise DatasetBuildError("split name must be all, train, val, or test")
    if split.strategy not in SPLIT_STRATEGIES:
        raise DatasetBuildError(
            "split strategy must be episode, session, task, or scene"
        )
    held = _held_out_names(split.held_out_scenes)
    if held and split.strategy != "scene":
        raise DatasetBuildError("held_out_scenes requires split strategy 'scene'")
    prepared = Split(
        name=split.name,
        train=_weight(split.train, "train"),
        val=_weight(split.val, "val"),
        test=_weight(split.test, "test"),
        strategy=split.strategy,
        held_out_scenes=held,
    )
    if prepared.name != "all" and prepared.train + prepared.val + prepared.test <= 0:
        raise DatasetBuildError("split fractions must sum to a positive number")
    return prepared


def held_out_scenes_present(episodes: Sequence[SplitEpisode], split: Split) -> None:
    """Raise when a held-out scene matches no episode."""
    if split.strategy != "scene" or not split.held_out_scenes:
        return
    present = {_label(episode.scene) for episode in episodes}
    present.discard(None)
    missing = [name for name in split.held_out_scenes if name not in present]
    if missing:
        raise DatasetBuildError(
            "held-out scene not in the episodes: " + ", ".join(missing)
        )


def assign_splits(
    episodes: Sequence[SplitEpisode],
    split: Split,
    seed: int,
) -> SplitAssignment:
    """Assign whole episodes to train, val, and test.

    The same episodes, split, and seed always return the same assignment.
    Every episode lands in exactly one part. ``strategy="episode"`` shuffles
    episodes. ``session`` groups by session, then by ``day``, then by the
    UTC day of a Unix timestamp at or after 1000000000, otherwise one
    undated group. ``task`` groups by the task string. ``scene`` groups by
    scene. Weights are Hamilton shares of those groups. Held-out scenes are
    appended to test in input order.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DatasetBuildError("seed must be an integer")
    if not episodes:
        raise DatasetBuildError("no episodes to split")
    prepared = prepare_split(split)
    _validate_episodes(episodes)
    held_out_scenes_present(episodes, prepared)
    groups = _groups(episodes, prepared.strategy)
    kept, parked = _partition_held(groups, prepared.held_out_scenes)
    if kept:
        shuffled = _deterministic_shuffle(kept, seed)
        sliced = _slice_groups(shuffled, prepared)
    else:
        sliced = {"train": [], "val": [], "test": []}
    return SplitAssignment(
        train=_expand(sliced["train"]),
        val=_expand(sliced["val"]),
        test=_expand([*sliced["test"], *parked]),
    )


def _validate_episodes(episodes: Sequence[SplitEpisode]) -> None:
    seen: set[str] = set()
    for episode in episodes:
        if not isinstance(episode.episode_id, str) or not episode.episode_id.strip():
            raise DatasetBuildError("each episode needs an id")
        if episode.episode_id in seen:
            raise DatasetBuildError(f"duplicate episode id: {episode.episode_id}")
        seen.add(episode.episode_id)
        if not isinstance(episode.task, str):
            raise DatasetBuildError("episode task must be a string")
        for frame_id in episode.frame_ids:
            if not isinstance(frame_id, str):
                raise DatasetBuildError("frame ids must be strings")
        _day_from_timestamp(episode.timestamp)


def _groups(episodes: Sequence[SplitEpisode], strategy: str) -> list[_Group]:
    order: list[str] = []
    found: dict[str, list[SplitEpisode]] = {}
    for episode in episodes:
        key = _group_key(episode, strategy)
        if key not in found:
            order.append(key)
            found[key] = []
        found[key].append(episode)
    return [_Group(key, tuple(found[key])) for key in order]


def _group_key(episode: SplitEpisode, strategy: str) -> str:
    if strategy == "episode":
        return episode.episode_id
    if strategy == "task":
        return episode.task
    if strategy == "scene":
        return _label(episode.scene) or ""
    if strategy == "session":
        session = _label(episode.session)
        if session is not None:
            return "session:" + session
        day = _label(episode.day) or _day_from_timestamp(episode.timestamp)
        if day is not None:
            return "day:" + day
        return "day:"
    raise DatasetBuildError("split strategy must be episode, session, task, or scene")


def _partition_held(
    groups: Sequence[_Group], held: tuple[str, ...]
) -> tuple[list[_Group], list[_Group]]:
    held_keys = set(held)
    kept: list[_Group] = []
    parked: list[_Group] = []
    for group in groups:
        if group.key in held_keys:
            parked.append(group)
        else:
            kept.append(group)
    return kept, parked


def _slice_groups(groups: Sequence[_Group], split: Split) -> dict[str, list[_Group]]:
    counts = _allocation(len(groups), split)
    starts = {
        "train": 0,
        "val": counts["train"],
        "test": counts["train"] + counts["val"],
    }
    return {
        name: list(groups[starts[name] : starts[name] + counts[name]])
        for name in _PARTS
    }


def _expand(groups: Sequence[_Group]) -> tuple[SplitEpisode, ...]:
    return tuple(episode for group in groups for episode in group.episodes)


def _label(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetBuildError("split group labels must be strings")
    text = value.strip()
    return text or None


def _day_from_timestamp(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise DatasetBuildError("episode timestamp must be a number")
    number = float(timestamp)
    if not math.isfinite(number) or number < _UNIX_TIMESTAMP_MIN:
        return None
    return datetime.fromtimestamp(number, timezone.utc).date().isoformat()


def _held_out_names(names: Sequence[str]) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for raw in names:
        if not isinstance(raw, str):
            raise DatasetBuildError("held-out scene names must be strings")
        name = raw.strip()
        if not name:
            raise DatasetBuildError("held-out scene names must be non-empty strings")
        if any(ord(char) < 32 for char in name):
            raise DatasetBuildError(
                "held-out scene names must not contain control characters"
            )
        if name in seen:
            continue
        seen.add(name)
        found.append(name)
    found.sort()
    return tuple(found)


def _weight(value: float, name: str) -> float:
    number = json_number(value)
    if number < 0:
        raise DatasetBuildError(f"{name} fraction cannot be negative")
    return number


def _deterministic_shuffle(items: list[_Group], seed: int) -> list[_Group]:
    """Fisher-Yates driven only by ``Random.random``, which is stable."""
    out = list(items)
    rng = random.Random(seed)
    for index in range(len(out) - 1, 0, -1):
        swap = int(rng.random() * (index + 1))
        out[index], out[swap] = out[swap], out[index]
    return out


def _allocation(count: int, split: Split) -> dict[str, int]:
    """Hamilton apportionment of ``count`` groups across train/val/test.

    Episode strategy uses one group per episode, so the counts match the
    episode split used by dataset versions.
    """
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
