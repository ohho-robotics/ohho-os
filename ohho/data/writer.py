"""LeRobot v2.1 dataset writer — Parquet + meta JSON.

Writes episodes to disk in the LeRobot v2.1 layout (the format lerobot 0.3.x
loads locally)::

    <root>/
      meta/
        info.json              # codebase_version v2.1, fps, features, dims
        tasks.jsonl            # {task_index, task}
        episodes.jsonl         # {episode_index, tasks, length}
        episodes_stats.jsonl   # {episode_index, stats} per episode
        stats.json             # global min/max/mean (kept for local readers)
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

# Scalar columns LeRobotDatasetMetadata.create always merges into features.
_SCALAR_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


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
    _write_stats(meta_dir, episodes)
    _write_episode_stats(meta_dir, episodes)

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
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
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
            **_SCALAR_FEATURES,
        },
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def _write_stats(meta_dir: Path, episodes: List[Episode]) -> None:
    all_frames = [f for ep in episodes for f in ep.frames]
    if not all_frames:
        return
    state_dim = episodes[0].state_dim if episodes else 0
    action_dim = episodes[0].action_dim if episodes else 0

    stats = {}
    if state_dim > 0:
        state_cols = [
            [f.observation_state[i] for f in all_frames] for i in range(state_dim)
        ]
        stats["observation.state"] = {
            "min": [float(min(col)) for col in state_cols],
            "max": [float(max(col)) for col in state_cols],
            "mean": [float(sum(col) / len(col)) for col in state_cols],
        }
    if action_dim > 0:
        action_cols = [[f.action[i] for f in all_frames] for i in range(action_dim)]
        stats["action"] = {
            "min": [float(min(col)) for col in action_cols],
            "max": [float(max(col)) for col in action_cols],
            "mean": [float(sum(col) / len(col)) for col in action_cols],
        }
    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def _vector_stats(rows: List[List[float]]) -> dict:
    """Per-dimension min/max/mean/std/count in the shape lerobot 0.3.x aggregates."""
    dim = len(rows[0])
    n = len(rows)
    mins: List[float] = []
    maxs: List[float] = []
    means: List[float] = []
    stds: List[float] = []
    for i in range(dim):
        col = [row[i] for row in rows]
        mean = sum(col) / n
        var = sum((value - mean) ** 2 for value in col) / n
        mins.append(float(min(col)))
        maxs.append(float(max(col)))
        means.append(float(mean))
        stds.append(float(var**0.5))
    return {"min": mins, "max": maxs, "mean": means, "std": stds, "count": [n]}


def _write_episode_stats(meta_dir: Path, episodes: List[Episode]) -> None:
    """Write ``meta/episodes_stats.jsonl`` (one stats object per episode)."""
    path = meta_dir / "episodes_stats.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ep in episodes:
            if not ep.frames:
                continue
            stats: dict = {}
            if ep.state_dim > 0:
                stats["observation.state"] = _vector_stats(
                    [frame.observation_state for frame in ep.frames]
                )
            if ep.action_dim > 0:
                stats["action"] = _vector_stats([frame.action for frame in ep.frames])
            f.write(
                json.dumps({"episode_index": ep.episode_index, "stats": stats}) + "\n"
            )


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
