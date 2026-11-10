"""Parity inputs: seeded random, a LeRobot dataset, a sim rollout, or a camera log.

Dataset and recorded sources use ``observation.state`` from
:class:`ohho.data.reader.DatasetReader`. The ``recorded`` keyword calls
:func:`ohho.sim_loop.record_sim_episodes` and then reads that dataset.
A camera log is a JSONL file, or a directory with ``frames.jsonl`` or
``log.jsonl``. Each row has ``observation.state`` and may name an ``image``.
Random inputs are drawn with :class:`random.Random` so the seed is stable
without PyTorch.
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ohho.optimize.models import ExportInput

from ohho.parity.errors import ParityError

_PREFIXES = (
    ("dataset:", "dataset"),
    ("recorded:", "recorded"),
    ("episode:", "recorded"),
    ("camera:", "camera"),
)


@dataclass(frozen=True)
class FrameSource:
    """Observation rows selected from a dataset or a recorded episode."""

    source: str
    path: str | None
    seed: int
    n_requested: int
    indices: list[int]
    frames: list[list[float]]
    media: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class InputBundle:
    """Tensors actually fed to both models, keyed by input name."""

    source: str
    path: str | None
    seed: int
    n_requested: int
    indices: list[int] | None
    batches: dict[str, Any]

    @property
    def n(self) -> int:
        if not self.batches:
            return 0
        first = next(iter(self.batches.values()))
        if isinstance(first, list):
            return len(first)
        return 1


def parse_input_source(text: str) -> tuple[str, str | None]:
    """Return ``(kind, path)``.

    ``kind`` is ``random``, ``dataset``, ``recorded``, ``camera``, or ``path``
    (a filesystem path whose kind is decided once the path is opened).
    """
    raw = text.strip()
    if not raw:
        raise ParityError("--inputs is required.")
    lowered = raw.lower()
    if lowered == "random":
        return "random", None
    if lowered == "recorded":
        return "recorded", None
    for prefix, kind in _PREFIXES:
        if lowered.startswith(prefix):
            path = raw[len(prefix) :].strip().strip('"')
            if not path:
                raise ParityError(f"{prefix} needs a path.")
            return kind, path
    return "path", raw


def batches_sha256(batches: dict[str, Any]) -> str:
    """SHA-256 of the inputs as little-endian float32, names in sorted order.

    Each tensor is flattened row-major. The hash is of those bytes concatenated.
    """
    payload = bytearray()
    for name in sorted(batches):
        for value in _flat_numbers(batches[name]):
            payload.extend(struct.pack("<f", float(value)))
    return hashlib.sha256(payload).hexdigest()


def load_observation_frames(
    source: str,
    *,
    n: int,
    seed: int,
    record_dir: str | None = None,
) -> FrameSource:
    """Load real observation rows. ``random`` is not a frame source."""
    if n < 1:
        raise ParityError("--n must be at least 1.")
    kind, path_text = parse_input_source(source)
    if kind == "random":
        raise ParityError("random inputs have no dataset frames.")
    if kind == "recorded" and path_text is None:
        recorded = _record_episode(n, record_dir)
        kind = "recorded"
        path_text = recorded
    if path_text is None:
        raise ParityError("--inputs path is missing.")
    path = Path(path_text)
    if kind == "path":
        kind = _kind_of_path(path)
    media_rows: list[str | None] | None = None
    if kind == "camera":
        frames, media_rows = _read_camera(path)
    else:
        frames = _read_frames(path, kind)
    if not frames:
        raise ParityError(f"{path} has no observation frames.")
    width = len(frames[0])
    if any(len(frame) != width for frame in frames):
        raise ParityError(f"{path} frames do not share one observation width.")
    indices = choose_indices(len(frames), n, seed)
    chosen = [frames[index] for index in indices]
    chosen_media: tuple[str | None, ...] = ()
    if media_rows is not None:
        if len(media_rows) != len(frames):
            raise ParityError(f"{path} camera frames and images do not match.")
        chosen_media = tuple(media_rows[index] for index in indices)
    return FrameSource(
        source=kind,
        path=str(path),
        seed=seed,
        n_requested=n,
        indices=indices,
        frames=chosen,
        media=chosen_media,
    )


def load_batches(
    source: str,
    specs: tuple[ExportInput, ...],
    *,
    n: int,
    seed: int,
    record_dir: str | None = None,
) -> InputBundle:
    """Build the batch each model will see."""
    if n < 1:
        raise ParityError("--n must be at least 1.")
    if not specs:
        raise ParityError("The model has no input shapes.")
    kind, _path = parse_input_source(source)
    if kind == "random":
        return InputBundle(
            source="random",
            path=None,
            seed=seed,
            n_requested=n,
            indices=None,
            batches=_random_batches(specs, n, seed),
        )
    frames = load_observation_frames(source, n=n, seed=seed, record_dir=record_dir)
    return bundle_from_frames(frames, specs)


def bundle_from_frames(
    frames: FrameSource, specs: tuple[ExportInput, ...]
) -> InputBundle:
    """Pack frames that were already loaded into the model's input batches."""
    if not frames.frames:
        raise ParityError("dataset has no frames.")
    return InputBundle(
        source=frames.source,
        path=frames.path,
        seed=frames.seed,
        n_requested=frames.n_requested,
        indices=frames.indices,
        batches=_batches_from_frames(frames.frames, specs, frames.seed),
    )


def choose_indices(count: int, n: int, seed: int) -> list[int]:
    """Frame indexes to score. A short dataset uses every row, in order.

    A longer dataset draws ``n`` indexes with ``random.Random(seed)`` and
    sorts them so the batch stays in time order.
    """
    if count < 1:
        raise ParityError("dataset has no frames.")
    if count <= n:
        return list(range(count))
    rng = random.Random(seed)
    return sorted(rng.sample(range(count), n))


def _record_episode(n: int, record_dir: str | None) -> str:
    import tempfile

    from ohho.sim_loop import record_sim_episodes

    steps = min(max(n, 1), 8)
    if record_dir:
        destination = Path(record_dir)
        destination.mkdir(parents=True, exist_ok=True)
        output = str(destination)
    else:
        output = tempfile.mkdtemp(prefix="ohho-parity-")
    return record_sim_episodes(
        num_episodes=1,
        steps_per_episode=steps,
        output_dir=output,
    )


def _kind_of_path(path: Path) -> str:
    if not path.exists():
        raise ParityError(f"Inputs path does not exist: {path}")
    if path.is_dir():
        if (path / "meta" / "info.json").is_file():
            return "dataset"
        if (path / "frames.jsonl").is_file() or (path / "log.jsonl").is_file():
            return "camera"
        return "dataset"
    if path.is_file() and path.suffix.lower() == ".jsonl":
        return "recorded"
    raise ParityError(
        f"{path} is not a LeRobot dataset directory, an episode .jsonl file, "
        "or a camera log."
    )


def _read_frames(path: Path, kind: str) -> list[list[float]]:
    if path.is_file():
        return _read_episode_file(path)
    if not path.is_dir():
        raise ParityError(f"Inputs path does not exist: {path}")
    if kind == "recorded" and not (path / "meta" / "info.json").is_file():
        if path.suffix.lower() == ".jsonl":
            return _read_episode_file(path)
    return _read_dataset_dir(path)


def _read_dataset_dir(path: Path) -> list[list[float]]:
    from ohho.data.reader import DatasetReader

    reader = DatasetReader(str(path))
    rows: list[list[float]] = []
    for frame in reader.all_frames():
        rows.append(_as_float_row(frame.observation_state, path))
    if rows:
        return rows
    # A directory that only holds episode files still counts.
    for episode in sorted(path.rglob("episode_*.jsonl")):
        rows.extend(_read_episode_file(episode))
    return rows


def _read_camera(path: Path) -> tuple[list[list[float]], list[str | None]]:
    files = _camera_files(path)
    if not files:
        raise ParityError(f"{path} has no camera log.")
    base = path if path.is_dir() else path.parent
    frames: list[list[float]] = []
    media: list[str | None] = []
    for file in files:
        for row in _json_objects(file):
            frames.append(_observation_from_row(row, file))
            media.append(_image_from_row(row, file, base))
    return frames, media


def _camera_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".jsonl":
            raise ParityError(f"{path} is not a camera log (.jsonl).")
        return [path]
    if not path.is_dir():
        raise ParityError(f"Inputs path does not exist: {path}")
    for name in ("frames.jsonl", "log.jsonl"):
        candidate = path / name
        if candidate.is_file():
            return [candidate]
    return sorted(path.glob("*.jsonl"))


def _read_episode_file(path: Path) -> list[list[float]]:
    return [_observation_from_row(row, path) for row in _json_objects(path)]


def _json_objects(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ParityError(f"Could not read {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ParityError(f"{path} is not valid JSON Lines: {exc}") from exc
        if not isinstance(row, dict):
            raise ParityError(f"{path} rows must be objects.")
        rows.append(row)
    return rows


def _observation_from_row(row: dict[str, Any], path: Path) -> list[float]:
    state = row.get("observation.state", row.get("observation_state"))
    if state is None and "observation" in row:
        observed = row["observation"]
        if isinstance(observed, dict):
            state = observed.get("state", observed.get("observation.state"))
        else:
            state = observed
    return _as_float_row(state, path)


def _image_from_row(row: dict[str, Any], path: Path, base: Path) -> str | None:
    image = row.get("image")
    if image is None:
        return None
    if not isinstance(image, str) or not image.strip():
        raise ParityError(f"{path} image must be a path.")
    image_path = Path(image)
    if not image_path.is_absolute():
        image_path = base / image
    return str(image_path)


def _as_float_row(state: object, path: Path) -> list[float]:
    if not isinstance(state, (list, tuple)) or not state:
        raise ParityError(f"{path} observation.state must be a non-empty list.")
    values: list[float] = []
    for item in state:
        if isinstance(item, bool):
            raise ParityError(f"{path} observation.state must be numbers.")
        try:
            values.append(float(item))
        except (TypeError, ValueError) as exc:
            raise ParityError(f"{path} observation.state must be numbers.") from exc
    return values


def _batches_from_frames(
    frames: list[list[float]],
    specs: tuple[ExportInput, ...],
    seed: int,
) -> dict[str, Any]:
    width = len(frames[0])
    matched = [spec for spec in specs if _tail_width(spec) == width]
    if not matched:
        described = ", ".join(f"{spec.name} {list(spec.shape)}" for spec in specs)
        raise ParityError(
            f"Dataset frames are {width}-D and do not match a model input ({described})."
        )
    batches: dict[str, Any] = {}
    for spec in matched:
        tail = spec.shape[1:]
        batches[spec.name] = [_reshape(frame, tail) for frame in frames]
    missing = [spec for spec in specs if spec.name not in batches]
    if missing:
        batches.update(_random_batches(tuple(missing), len(frames), seed))
    return batches


def _random_batches(
    specs: tuple[ExportInput, ...], n: int, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    batches: dict[str, Any] = {}
    for spec in specs:
        shape = (n, *spec.shape[1:]) if spec.shape else (n,)
        count = 1
        for dim in shape:
            if dim < 1:
                raise ParityError(f"Input {spec.name} has an empty shape {spec.shape}.")
            count *= dim
        values = [rng.uniform(-1.0, 1.0) for _ in range(count)]
        batches[spec.name] = _pack(values, shape)
    return batches


def _tail_width(spec: ExportInput) -> int:
    width = 1
    for dim in spec.shape[1:]:
        width *= dim
    return width


def _reshape(flat: list[float], tail: tuple[int, ...]) -> Any:
    if not tail:
        if len(flat) != 1:
            raise ParityError("scalar input expects one value per frame.")
        return flat[0]
    return _pack(flat, tail)


def _pack(values: list[float], shape: tuple[int, ...]) -> Any:
    cursor = 0

    def take(dims: tuple[int, ...]) -> Any:
        nonlocal cursor
        if not dims:
            item = values[cursor]
            cursor += 1
            return item
        size = dims[0]
        if len(dims) == 1:
            start = cursor
            cursor += size
            return list(values[start:cursor])
        return [take(dims[1:]) for _ in range(size)]

    packed = take(shape)
    if cursor != len(values):
        raise ParityError("input shape does not match the number of values.")
    return packed


def _flat_numbers(value: object) -> list[float]:
    if isinstance(value, list):
        flat: list[float] = []
        for item in value:
            flat.extend(_flat_numbers(item))
        return flat
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityError("input value is not a number.")
    return [float(value)]
