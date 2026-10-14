"""LeRobot v2.1 dataset writer — Parquet + meta JSON.

Writes episodes to disk in the standard LeRobot v2.1 layout::

    <root>/
      meta/
        info.json           # codebase_version, fps, features, dims
        tasks.jsonl         # {task_index, task}
        episodes.jsonl      # {episode_index, tasks, length}
      data/chunk-000/
        episode_000000.parquet   # or .jsonl without pyarrow

Parquet is used when ``pyarrow`` is available (the ``[data]`` extra); otherwise
JSON Lines keeps the record→inspect loop working with zero deps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .recorder import Episode

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000


def write_dataset(
    episodes: List[Episode],
    root: str,
    repo_id: str,
    fps: float,
    tasks: List[str],
) -> str:
    """Write episodes to ``root`` in LeRobot v2.1 format. Returns the root path."""
    r = Path(root).expanduser()
    meta_dir = r / "meta"
    data_dir = r / "data" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    _write_tasks(meta_dir, tasks)
    _write_episodes(meta_dir, episodes)
    _write_info(meta_dir, episodes, repo_id, fps, tasks)
    _write_episodes_stats(meta_dir, episodes)

    use_parquet = _try_import_pyarrow() is not None
    for ep in episodes:
        fname = (
            f"episode_{ep.episode_index:06d}.{'parquet' if use_parquet else 'jsonl'}"
        )
        fpath = data_dir / fname
        if use_parquet:
            _write_parquet(ep, fpath)
        else:
            _write_jsonl(ep, fpath)

    return str(r)


def _try_import_pyarrow():
    try:
        import pyarrow  # type: ignore
        import pyarrow.parquet  # type: ignore

        return pyarrow
    except ImportError:
        return None


def _write_tasks(meta_dir: Path, tasks: List[str]) -> None:
    path = meta_dir / "tasks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")


def _write_episodes(meta_dir: Path, episodes: List[Episode]) -> None:
    path = meta_dir / "episodes.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(
                json.dumps(
                    {
                        "episode_index": ep.episode_index,
                        "tasks": [ep.task],
                        "length": ep.length,
                    }
                )
                + "\n"
            )


def _write_info(
    meta_dir: Path,
    episodes: List[Episode],
    repo_id: str,
    fps: float,
    tasks: List[str],
) -> None:
    total_frames = sum(ep.length for ep in episodes)
    state_dim = episodes[0].state_dim if episodes else 0
    action_dim = episodes[0].action_dim if episodes else 0
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "omnibot",
        "repo_id": repo_id,
        "fps": fps,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "tasks": tasks,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [
                    "arm_shoulder_pan",
                    "arm_shoulder_lift",
                    "arm_elbow_flex",
                    "arm_wrist_flex",
                    "arm_wrist_roll",
                    "arm_gripper",
                    "base_vx",
                    "base_vy",
                    "base_omega",
                ][:state_dim],
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": [
                    "arm_shoulder_pan",
                    "arm_shoulder_lift",
                    "arm_elbow_flex",
                    "arm_wrist_flex",
                    "arm_wrist_roll",
                    "arm_gripper",
                    "base_vx",
                    "base_vy",
                    "base_omega",
                ][:action_dim],
            },
        },
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def _feature_stats(rows: List[List[float]]) -> dict:
    """Per-feature stats matching LeRobot v2.1 ``episodes_stats`` layout."""
    if not rows:
        return {}
    n = len(rows)
    dim = len(rows[0])
    mins = [float(min(row[i] for row in rows)) for i in range(dim)]
    maxs = [float(max(row[i] for row in rows)) for i in range(dim)]
    means = [float(sum(row[i] for row in rows) / n) for i in range(dim)]
    stds = [
        float(
            (sum((row[i] - means[i]) ** 2 for row in rows) / n) ** 0.5 if n > 0 else 0.0
        )
        for i in range(dim)
    ]
    # keepdims=True on vector features → shape (1, dim) when serialized
    return {
        "min": [mins],
        "max": [maxs],
        "mean": [means],
        "std": [stds],
        "count": [n],
    }


def _write_episodes_stats(meta_dir: Path, episodes: List[Episode]) -> None:
    """Write ``meta/episodes_stats.jsonl`` (LeRobot v2.1)."""
    path = meta_dir / "episodes_stats.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ep in episodes:
            if not ep.frames:
                continue
            state_rows = [f.observation_state for f in ep.frames]
            action_rows = [f.action for f in ep.frames]
            ep_stats = {
                "observation.state": _feature_stats(state_rows),
                "action": _feature_stats(action_rows),
            }
            line = {"episode_index": ep.episode_index, "stats": ep_stats}
            f.write(json.dumps(line) + "\n")


def _write_parquet(ep: Episode, fpath: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {
        "observation.state": [f.observation_state for f in ep.frames],
        "action": [f.action for f in ep.frames],
        "timestamp": [f.timestamp for f in ep.frames],
        "frame_index": [f.frame_index for f in ep.frames],
        "episode_index": [f.episode_index for f in ep.frames],
        "task_index": [f.task_index for f in ep.frames],
        "next.done": [f.next_done for f in ep.frames],
        "index": [f.episode_index * 100000 + f.frame_index for f in ep.frames],
    }
    table = pa.table(cols)
    pq.write_table(table, fpath, compression="snappy")


def _write_jsonl(ep: Episode, fpath: Path) -> None:
    with open(fpath, "w", encoding="utf-8") as f:
        for frame in ep.frames:
            row = {
                "observation.state": frame.observation_state,
                "action": frame.action,
                "timestamp": frame.timestamp,
                "frame_index": frame.frame_index,
                "episode_index": frame.episode_index,
                "task_index": frame.task_index,
                "next.done": frame.next_done,
                "index": frame.episode_index * 100000 + frame.frame_index,
            }
            f.write(json.dumps(row) + "\n")
