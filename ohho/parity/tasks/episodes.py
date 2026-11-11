"""Held-out episodes for action error.

The built-in set is one episode per fixed seed. Each frame's ground-truth
action is :func:`synthetic_action`. A directory is read as a LeRobot dataset.
A JSON file uses ``{"episodes": [{"seed": 0, "frames": [{"observation": [],
"action": []}]}]}``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from ohho.parity.errors import ParityError
from ohho.parity.tasks.types import Episode, Frame, as_floats

DEFAULT_SEED_COUNT = 20


def fixed_seeds(count: int = DEFAULT_SEED_COUNT) -> tuple[int, ...]:
    """The fixed seed set ``0 .. count-1``. The default count is 20."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ParityError("--task-seeds must be an integer >= 1.")
    return tuple(range(count))


def synthetic_action(observation: Sequence[float]) -> list[float]:
    """Ground-truth action for the built-in held-out set.

    The readout is ``[0.5 * o0, -0.25 * o1, 0.1, 0.2]``. Missing observation
    channels are treated as zero.
    """
    values = as_floats(observation, "observation")
    padded = values + [0.0, 0.0, 0.0, 0.0]
    return [0.5 * padded[0], -0.25 * padded[1], 0.1, 0.2]


def synthetic_episodes(seeds: Sequence[int]) -> tuple[Episode, ...]:
    """Four frames per seed, observations in ``[-1, 1]``, deterministic."""
    if not seeds:
        raise ParityError("the task gate needs at least one seed.")
    episodes: list[Episode] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ParityError("seeds must be integers.")
        rng = random.Random(10_000 + seed)
        frames: list[Frame] = []
        for _ in range(4):
            observation = tuple(rng.uniform(-1.0, 1.0) for _ in range(4))
            action = tuple(synthetic_action(observation))
            frames.append(Frame(observation, action))
        episodes.append(Episode(tuple(frames), seed=seed))
    return tuple(episodes)


def episodes_from_path(path: str) -> tuple[Episode, ...]:
    """Load a LeRobot dataset directory or a JSON episode file."""
    file = Path(path)
    if not file.exists():
        raise ParityError(f"episode set does not exist: {file}")
    if file.is_dir():
        return _from_dataset(file)
    return _from_json(file)


def _from_dataset(path: Path) -> tuple[Episode, ...]:
    from ohho.data.reader import DatasetReader

    reader = DatasetReader(str(path))
    frames = reader.all_frames()
    if not frames:
        raise ParityError(f"{path} has no frames.")
    grouped: dict[int, list[Frame]] = {}
    for frame in frames:
        grouped.setdefault(int(frame.episode_index), []).append(
            Frame(
                tuple(as_floats(frame.observation_state, "observation.state")),
                tuple(as_floats(frame.action, "action")),
            )
        )
    return tuple(
        Episode(tuple(grouped[index]), seed=index) for index in sorted(grouped)
    )


def _from_json(path: Path) -> tuple[Episode, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParityError(f"could not read episodes {path}: {exc}") from exc
    rows = document.get("episodes") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ParityError(f"{path} must contain a non-empty episodes list.")
    episodes: list[Episode] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("frames"), list):
            raise ParityError(f"{path} episodes[{index}] needs a frames list.")
        frames: list[Frame] = []
        for frame in row["frames"]:
            if not isinstance(frame, dict):
                raise ParityError(f"{path} episodes[{index}] has a bad frame.")
            frames.append(
                Frame(
                    tuple(as_floats(frame.get("observation") or [], "observation")),
                    tuple(as_floats(frame.get("action") or [], "action")),
                )
            )
        if not frames:
            raise ParityError(f"{path} episodes[{index}] has no frames.")
        seed = row.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ParityError(f"{path} episodes[{index}].seed must be an integer.")
        episodes.append(Episode(tuple(frames), seed=seed))
    return tuple(episodes)
