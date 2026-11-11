"""LeRobot dataset paths for imitation-learning recipes.

A local directory is used as-is. An Hugging Face Hub id is ``hf://org/name``
or a bare ``org/name`` that is not an existing directory and whose first
segment is not a workspace folder name (so ``datasets/sim-loop`` stays a
local path). Download happens only from :func:`materialize_hub_dataset`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...data.reader import DatasetReader
from ..recipe_schema import DatasetUnavailable

_HUB_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
)
_LOCAL_ROOTS = frozenset(
    {
        "checkpoints",
        "data",
        "datasets",
        "docs",
        "examples",
        "ohho",
        "outputs",
        "plugins",
        "runs",
        "scripts",
        "tests",
        "tmp",
    }
)


@dataclass(frozen=True)
class DatasetRef:
    """A resolved dataset reference. ``path`` is set only for local directories."""

    kind: str
    repo_id: str = ""
    path: Path | None = None


def resolve_dataset(recipe: dict[str, Any]) -> DatasetRef:
    """Return a local directory or a Hub id. Does not download."""
    dataset = recipe.get("dataset")
    if not isinstance(dataset, dict) or "path" not in dataset:
        raise DatasetUnavailable(
            "dataset version ids are not resolved in this build; "
            "set dataset.path to a local dataset directory or hf://org/name"
        )
    return classify_dataset_path(str(dataset["path"]))


def classify_dataset_path(raw: str) -> DatasetRef:
    """Classify one ``dataset.path`` string. Does not download."""
    text = raw.strip()
    if text.startswith("hf://"):
        repo = text[len("hf://") :].strip("/")
        if not _HUB_ID.fullmatch(repo):
            raise DatasetUnavailable(f"invalid HF Hub dataset id: {text}")
        return DatasetRef("hub", repo_id=repo)
    path = Path(text).expanduser()
    if path.is_dir():
        return DatasetRef("local", path=path)
    normalized = text.replace("\\", "/")
    first = normalized.split("/", 1)[0]
    if (
        _HUB_ID.fullmatch(normalized)
        and not path.is_absolute()
        and not normalized.startswith(".")
        and ".." not in normalized.split("/")
        and first not in _LOCAL_ROOTS
    ):
        return DatasetRef("hub", repo_id=normalized)
    raise DatasetUnavailable(f"dataset path does not exist: {path}")


def load_windows(
    dataset: Path, chunk_size: int
) -> tuple[list[list[float]], list[list[list[float]]], int, int]:
    """Load state vectors and action chunks from a local LeRobot v2 dataset."""
    reader = DatasetReader(str(dataset))
    states: list[list[float]] = []
    chunks: list[list[list[float]]] = []
    for _episode, frames in reader.iter_episodes():
        if not frames:
            continue
        for index in range(len(frames)):
            chunk: list[list[float]] = []
            for offset in range(chunk_size):
                source = min(index + offset, len(frames) - 1)
                chunk.append(list(frames[source].action))
            states.append(list(frames[index].observation_state))
            chunks.append(chunk)
    if not states:
        raise ValueError(f"no recorded frames found in dataset at {dataset}")
    state_dim = reader.state_dim or len(states[0])
    action_dim = reader.action_dim or len(chunks[0][0])
    return states, chunks, state_dim, action_dim


def materialize_hub_dataset(repo_id: str) -> Path:
    """Download a Hub dataset to a local LeRobot directory.

    Tests patch this. A missing ``lerobot`` install fails here, before training.
    """
    try:
        factory = _import_lerobot_dataset()
    except ImportError as exc:
        raise DatasetUnavailable(
            f"HF Hub dataset {repo_id!r} needs lerobot to download. "
            "Install with: pip install lerobot"
        ) from exc
    try:
        dataset = _construct_lerobot_dataset(factory, repo_id)
    except DatasetUnavailable:
        raise
    except Exception as exc:
        raise DatasetUnavailable(
            f"could not download HF Hub dataset {repo_id!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    root = _dataset_root(dataset)
    if root is None:
        raise DatasetUnavailable(
            f"HF Hub dataset {repo_id!r} did not produce a local LeRobot directory "
            "(expected meta/info.json)"
        )
    return root


def _import_lerobot_dataset() -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def _construct_lerobot_dataset(factory: Any, repo_id: str) -> Any:
    last: Exception | None = None
    for attempt in (
        lambda: factory(repo_id),
        lambda: factory(repo_id=repo_id),
    ):
        try:
            return attempt()
        except TypeError as exc:
            last = exc
    if last is None:
        raise DatasetUnavailable(f"could not construct a dataset for {repo_id!r}")
    raise last


def _dataset_root(dataset: Any) -> Path | None:
    for attr in ("root", "root_dir", "local_dir"):
        value = getattr(dataset, attr, None)
        if value is None:
            continue
        path = Path(str(value))
        if (path / "meta" / "info.json").is_file():
            return path
    return None
