"""LeRobot v2.1 dataset writer — Parquet + meta JSON.

Writes episodes to disk in the LeRobot v2.1 layout (the format lerobot 0.3.x
loads locally)::

    <root>/
      README.md           # Dataset card (Hugging Face / LeRobot metadata)
      meta/
        info.json         # codebase_version, fps, features, dims, video_path
        tasks.jsonl       # {task_index, task}
        episodes.jsonl    # {episode_index, tasks, length}
        episodes_stats.jsonl  # per-episode min/max/mean/std/count (v2.1)
        stats.json        # min/max/mean per feature
      data/chunk-000/
        episode_000000.parquet   # or .jsonl without pyarrow
      videos/chunk-000/   # (optional) encoded MP4 per episode
        observation.images.front/
          episode_000000.mp4

Parquet is used when ``pyarrow`` is available (the ``[data]`` extra); otherwise
JSON Lines keeps the record→inspect loop working with zero deps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from .recorder import (
    BASE_ACTION_NAMES,
    BASE_STATE_NAMES,
    GO2_ACTION_NAMES,
    GO2_STATE_NAMES,
    OMNIBOT_ACTION_NAMES,
    OMNIBOT_STATE_NAMES,
    Episode,
)

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000
DEFAULT_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"

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
    robot_type: Optional[str] = None,
) -> str:
    """Write episodes to ``root`` in LeRobot v2.1 format. Returns the root path."""
    r = Path(root).expanduser()
    meta_dir = r / "meta"
    data_dir = r / "data" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Inferred or explicit robot_type
    state_dim = episodes[0].state_dim if episodes else 0
    action_dim = episodes[0].action_dim if episodes else 0
    if robot_type is None:
        if state_dim == 31 and action_dim == 4:
            robot_type = "unitree-go2"
        elif state_dim == 9:
            robot_type = "omnibot"
        else:
            robot_type = "sim"

    _write_tasks(meta_dir, tasks)
    _write_episodes(meta_dir, episodes)
    features = _write_info(meta_dir, episodes, repo_id, fps, tasks, robot_type)
    _write_stats(meta_dir, episodes)
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

    # Optional video encoding per episode
    _write_videos(r, episodes, fps)

    # Dataset card
    _write_dataset_card(r, repo_id, robot_type, episodes, fps, tasks, features)

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


def _episode_names(episodes: List[Episode], attr: str) -> Optional[List[str]]:
    """Names stored on the episode by the recorder, when every episode agrees."""
    chosen: Optional[List[str]] = None
    for ep in episodes:
        names = getattr(ep, attr, None)
        if not names:
            continue
        current = [str(name) for name in names]
        if chosen is None:
            chosen = current
        elif current != chosen:
            return None
    return chosen


def _arm_action(episodes: List[Episode]) -> Optional[str]:
    """The arm-action form, when the dataset uses one form."""
    forms = []
    for ep in episodes:
        form = getattr(ep, "arm_action", None)
        if form and form not in forms:
            forms.append(str(form))
    if len(forms) == 1:
        return forms[0]
    return None


def _image_keys(episodes: List[Episode]) -> List[str]:
    keys: List[str] = []
    for ep in episodes:
        for key, frames in ep.images.items():
            if frames and key not in keys:
                keys.append(key)
    return keys


def _first_image(episodes: List[Episode], key: str) -> Any:
    for ep in episodes:
        frames = ep.images.get(key) or []
        if frames:
            return frames[0]
    return None


def _image_hw(sample: Any) -> tuple[int, int]:
    vid_h, vid_w = 480, 640
    if hasattr(sample, "shape") and len(sample.shape) >= 2:
        vid_h, vid_w = int(sample.shape[0]), int(sample.shape[1])
    elif hasattr(sample, "size"):
        vid_w, vid_h = int(sample.size[0]), int(sample.size[1])
    return vid_h, vid_w


def _write_info(
    meta_dir: Path,
    episodes: List[Episode],
    repo_id: str,
    fps: float,
    tasks: List[str],
    robot_type: str,
) -> dict:
    total_frames = sum(ep.length for ep in episodes)
    state_dim = episodes[0].state_dim if episodes else 0
    action_dim = episodes[0].action_dim if episodes else 0

    explicit_state = _episode_names(episodes, "state_names")
    explicit_action = _episode_names(episodes, "action_names")
    if explicit_state is not None and len(explicit_state) == state_dim:
        state_names = explicit_state
    elif robot_type == "unitree-go2" or state_dim == 31:
        state_names = GO2_STATE_NAMES[:state_dim]
    elif state_dim == 9:
        state_names = OMNIBOT_STATE_NAMES[:state_dim]
    else:
        state_names = BASE_STATE_NAMES[:state_dim]
    if explicit_action is not None and len(explicit_action) == action_dim:
        action_names = explicit_action
    elif robot_type == "unitree-go2" or state_dim == 31:
        action_names = GO2_ACTION_NAMES[:action_dim]
    elif state_dim == 9:
        action_names = OMNIBOT_ACTION_NAMES[:action_dim]
    else:
        action_names = BASE_ACTION_NAMES[:action_dim]

    has_raw = any(f.action_raw is not None for ep in episodes for f in ep.frames)
    has_videos = any(ep.has_videos for ep in episodes)

    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": action_names,
        },
    }

    if has_raw:
        features["action_raw"] = {
            "dtype": "float32",
            "shape": [action_dim],
            "names": action_names,
        }

    for image_key in _image_keys(episodes):
        sample = _first_image(episodes, image_key)
        vid_h, vid_w = _image_hw(sample)
        features[image_key] = {
            "dtype": "video",
            "shape": [3, vid_h, vid_w],
            "names": ["channels", "height", "width"],
            "video_info": {
                "video.fps": fps,
                "video.height": vid_h,
                "video.width": vid_w,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    total_videos = len(episodes) if has_videos else 0

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "repo_id": repo_id,
        "fps": fps,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": total_videos,
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH if has_videos else None,
        "tasks": tasks,
        "features": {**features, **_SCALAR_FEATURES},
    }
    arm_action = _arm_action(episodes)
    if arm_action is not None:
        info["arm_action"] = arm_action
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return features


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
    # Vector features (dim > 1): flat length-dim lists, matching keepdims=False.
    # Scalar features (dim == 1): length-1 lists, matching keepdims=True on 1-D data.
    return {
        "min": mins,
        "max": maxs,
        "mean": means,
        "std": stds,
        "count": [n],
    }


def _image_feature_stats(frames: List[Any]) -> dict:
    """Per-channel image stats in the LeRobot v2.1 nested ``(c, 1, 1)`` layout."""
    n = len(frames)
    fallback = {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.0]], [[0.0]], [[0.0]]],
        "count": [n],
    }
    if n == 0:
        return fallback
    try:
        import numpy as np  # type: ignore

        arr = np.array(frames, dtype=np.float32) / 255.0
        c_min = arr.min(axis=(0, 1, 2))
        c_max = arr.max(axis=(0, 1, 2))
        c_mean = arr.mean(axis=(0, 1, 2))
        c_std = arr.std(axis=(0, 1, 2))

        def _nest(vec: Any) -> list:
            return [[[float(vec[0])]], [[float(vec[1])]], [[float(vec[2])]]]

        return {
            "min": _nest(c_min),
            "max": _nest(c_max),
            "mean": _nest(c_mean),
            "std": _nest(c_std),
            "count": [n],
        }
    except Exception:
        return fallback


def _write_episodes_stats(meta_dir: Path, episodes: List[Episode]) -> None:
    """Write ``meta/episodes_stats.jsonl`` (LeRobot v2.1)."""
    path = meta_dir / "episodes_stats.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ep in episodes:
            if not ep.frames:
                continue
            state_rows = [frame.observation_state for frame in ep.frames]
            action_rows = [frame.action for frame in ep.frames]
            ep_stats: dict[str, Any] = {
                "observation.state": _feature_stats(state_rows),
                "action": _feature_stats(action_rows),
            }
            if any(frame.action_raw is not None for frame in ep.frames):
                raw_rows = [
                    frame.action_raw if frame.action_raw is not None else frame.action
                    for frame in ep.frames
                ]
                ep_stats["action_raw"] = _feature_stats(raw_rows)
            for image_key, frames in ep.images.items():
                if frames:
                    ep_stats[image_key] = _image_feature_stats(frames)
            line = {"episode_index": ep.episode_index, "stats": ep_stats}
            f.write(json.dumps(line) + "\n")


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

    has_raw = any(f.action_raw is not None for f in all_frames)
    if has_raw:
        raw_cols = [
            [(f.action_raw or f.action)[i] for f in all_frames]
            for i in range(action_dim)
        ]
        stats["action_raw"] = {
            "min": [float(min(col)) for col in raw_cols],
            "max": [float(max(col)) for col in raw_cols],
            "mean": [float(sum(col) / len(col)) for col in raw_cols],
        }

    for image_key in _image_keys(episodes):
        video_frames = []
        for ep in episodes:
            frames = ep.images.get(image_key, [])
            if frames:
                video_frames.extend(frames[:5])
        if video_frames:
            stats[image_key] = _image_feature_stats(video_frames)

    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def _encode_mp4_video(frames: List[Any], path: Path, fps: float) -> None:
    """Encode video frames into MP4 using PyAV (with fallback to imageio)."""
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import av  # type: ignore

        first = frames[0]
        h, w = first.shape[:2] if hasattr(first, "shape") else (480, 640)
        with av.open(str(path), "w") as output:
            stream = output.add_stream("h264", rate=int(fps) if fps >= 1.0 else 1)
            stream.width = int(w)
            stream.height = int(h)
            stream.pix_fmt = "yuv420p"
            for frame_arr in frames:
                vf = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
                for packet in stream.encode(vf):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        return
    except ImportError:
        pass

    try:
        import imageio  # type: ignore

        imageio.mimsave(str(path), frames, fps=fps, format="mp4")
        return
    except ImportError:
        pass

    raise RuntimeError(
        "Video encoding requires av or imageio: install with `pip install av` or `pip install 'ohho-os[data]'`"
    )


def _write_videos(root: Path, episodes: List[Episode], fps: float) -> None:
    """Write per-episode MP4 video files into videos/chunk-XXX/<key>/episode_XXXXXX.mp4."""
    for ep in episodes:
        if not ep.images:
            continue
        chunk = ep.episode_index // CHUNK_SIZE
        for vid_key, frames in ep.images.items():
            if not frames:
                continue
            vid_dir = root / "videos" / f"chunk-{chunk:03d}" / vid_key
            vid_dir.mkdir(parents=True, exist_ok=True)
            vid_path = vid_dir / f"episode_{ep.episode_index:06d}.mp4"
            _encode_mp4_video(frames, vid_path, fps)


def _write_dataset_card(
    root: Path,
    repo_id: str,
    robot_type: str,
    episodes: List[Episode],
    fps: float,
    tasks: List[str],
    features: dict,
) -> None:
    """Generate a standard LeRobot dataset card (README.md)."""
    total_frames = sum(ep.length for ep in episodes)
    tags = ["lerobot", "robotics", robot_type]
    image_keys = _image_keys(episodes)
    has_videos = bool(image_keys)
    if has_videos:
        tags.append("video")
    tag_lines = "\n".join(f"- {t}" for t in tags)
    task_lines = "\n".join(f"- {t}" for t in tasks) if tasks else "- demonstration"

    state_feat = features.get("observation.state", {})
    action_feat = features.get("action", {})
    state_names = ", ".join(state_feat.get("names", []))
    action_names = ", ".join(action_feat.get("names", []))

    card = f"""---
license: apache-2.0
tags:
{tag_lines}
task_categories:
- robotics
---

# Dataset Card: {repo_id}

This dataset contains teleoperation demonstrations recorded with OhhO OS on {robot_type}.

## Dataset Structure

- **Robot**: {robot_type}
- **Episodes**: {len(episodes)}
- **Total Frames**: {total_frames}
- **FPS**: {fps}
- **Videos**: {"Yes (" + ", ".join(image_keys) + ")" if image_keys else "No"}

### Tasks
{task_lines}

### Features
- `observation.state`: shape {state_feat.get("shape", [])} ({state_names})
- `action`: shape {action_feat.get("shape", [])} ({action_names})
- `action_raw`: shape {action_feat.get("shape", [])} (raw requested action before SafetyGate clamping)
"""
    form = _arm_action(episodes)
    if form is not None:
        card += f"- arm action form: `{form}` (`arm_action` in info.json)\n"
    for image_key in image_keys:
        card += f"- `{image_key}`: shape [3, H, W] encoded as MP4 per episode\n"

    card += f"""
## Usage

```python
from ohho.data.reader import DatasetReader

reader = DatasetReader("{str(root)}")
print(f"Loaded {{reader.episode_count}} episodes with {{reader.frame_count}} frames.")
stats = reader.stats()
print("State stats:", stats.get("observation.state"))
```
"""
    with open(root / "README.md", "w", encoding="utf-8") as f:
        f.write(card)


def _write_parquet(ep: Episode, fpath: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    has_raw = any(f.action_raw is not None for f in ep.frames)
    cols = {
        "observation.state": [f.observation_state for f in ep.frames],
        "action": [f.action for f in ep.frames],
    }
    if has_raw:
        cols["action_raw"] = [
            f.action_raw if f.action_raw is not None else f.action for f in ep.frames
        ]
    cols.update(
        {
            "timestamp": [f.timestamp for f in ep.frames],
            "frame_index": [f.frame_index for f in ep.frames],
            "episode_index": [f.episode_index for f in ep.frames],
            "task_index": [f.task_index for f in ep.frames],
            "next.done": [f.next_done for f in ep.frames],
            "index": [f.episode_index * 100000 + f.frame_index for f in ep.frames],
        }
    )
    table = pa.table(cols)
    pq.write_table(table, fpath, compression="snappy")


def _write_jsonl(ep: Episode, fpath: Path) -> None:
    with open(fpath, "w", encoding="utf-8") as f:
        for frame in ep.frames:
            row = {
                "observation.state": frame.observation_state,
                "action": frame.action,
            }
            if frame.action_raw is not None:
                row["action_raw"] = frame.action_raw
            row.update(
                {
                    "timestamp": frame.timestamp,
                    "frame_index": frame.frame_index,
                    "episode_index": frame.episode_index,
                    "task_index": frame.task_index,
                    "next.done": frame.next_done,
                    "index": frame.episode_index * 100000 + frame.frame_index,
                }
            )
            f.write(json.dumps(row) + "\n")
