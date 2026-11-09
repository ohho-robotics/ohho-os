"""Dataset version record: dataclass, JSON Schema, and content identity.

The schema file ``dataset_version.schema.json`` is the contract. One record
maps onto one future ``dataset_versions`` row. ``pinned``, ``created_at``,
``stats``, ``parent_version_id``, and ``storage_prefix`` are stored with the
version and are not part of the version id.

``split`` and ``augmentation`` are stored on the record and are included in
``content_hash``. ``seed`` is part of the version id. It changes
``content_hash`` when it changes which rows are materialised.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_NAME = "ohho.dataset_version.v1"
CONTENT_ALGORITHM = "ohho.dataset_content.v1"
VERSION_ID_ALGORITHM = "ohho.dataset_version_id.v1"
_VERSION_ID_RE = re.compile(r"^dv_[0-9a-f]{32}$")
_CONTENT_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class DatasetVersionError(ValueError):
    """A dataset version record or spec is invalid."""


class DatasetBuildError(DatasetVersionError):
    """The spec could not be materialised from local episodes."""


class DatasetVersionFrozen(DatasetVersionError):
    """A pinned dataset version cannot be modified."""


class DatasetVersionNotFound(DatasetVersionError):
    """No local dataset version or dataset path matches the reference."""


def schema_path() -> Path:
    """Path to the committed dataset-version JSON Schema."""
    return Path(__file__).resolve().with_name("dataset_version.schema.json")


def load_schema() -> dict[str, Any]:
    """Return the parsed dataset-version JSON Schema."""
    doc = json.loads(schema_path().read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise DatasetVersionError("dataset version schema is not an object")
    return doc


@dataclass(frozen=True)
class SourceQuery:
    """Where local episodes are read from.

    Each root is a LeRobot dataset directory (it contains ``meta/info.json``)
    or a directory whose immediate children are such datasets.
    """

    roots: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"roots": list(self.roots)}


@dataclass(frozen=True)
class Filters:
    """Episode filters applied after the source query is loaded."""

    task_contains: str | None = None
    robot_type: str | None = None
    min_frames: int | None = None
    max_frames: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_contains": self.task_contains,
            "robot_type": self.robot_type,
            "min_frames": self.min_frames,
            "max_frames": self.max_frames,
        }


@dataclass(frozen=True)
class Crop:
    """Height and width in pixels."""

    height: int
    width: int

    def to_dict(self) -> dict[str, int]:
        return {"height": self.height, "width": self.width}


@dataclass(frozen=True)
class ColorJitter:
    """Maximum colour jitter. Zero leaves that channel unchanged."""

    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "hue": self.hue,
        }


@dataclass(frozen=True)
class CameraToggle:
    """Whether one camera receives image augmentation."""

    name: str
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "enabled": self.enabled}


@dataclass(frozen=True)
class Split:
    """Which slice of the filtered episodes to materialise, and how to assign it.

    ``name="all"`` keeps every filtered episode in source order. ``train``,
    ``val``, and ``test`` materialise that part of a seeded partition.
    Fractions are weights, not required to sum to 1. ``strategy`` is
    ``episode``, ``session`` (session, else calendar day), ``task``, or
    ``scene``. ``held_out_scenes`` applies when the strategy is ``scene``.
    See :mod:`ohho.datasets.split`.
    """

    name: str = "all"
    train: float = 1.0
    val: float = 0.0
    test: float = 0.0
    strategy: str = "episode"
    held_out_scenes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "strategy": self.strategy,
            "held_out_scenes": list(self.held_out_scenes),
        }


@dataclass(frozen=True)
class Augmentation:
    """Transform stored on a dataset version.

    ``action_scale``, ``state_scale``, ``action_offset``, ``state_offset``,
    ``action_noise``, ``state_noise``, and ``copies`` are applied when the
    version is materialised. Noise draws use ``random.Random.random`` seeded
    by the version seed plus the copy index. Zero noise leaves that stream
    untouched. ``copies`` repeats the selected episodes.

    ``crop``, ``resize``, ``color_jitter``, ``blur``, ``temporal_stride``,
    and ``cameras`` are stored with the version and applied at train time by
    :func:`ohho.datasets.augment.apply_augmentation`. ``blur`` is a Gaussian
    sigma in pixels. ``temporal_stride`` keeps every Nth frame.
    ``cameras`` lists toggles; a camera omitted from the list is augmented.
    """

    action_scale: float = 1.0
    state_scale: float = 1.0
    action_offset: float = 0.0
    state_offset: float = 0.0
    action_noise: float = 0.0
    state_noise: float = 0.0
    copies: int = 1
    crop: Crop | None = None
    resize: Crop | None = None
    color_jitter: ColorJitter = field(default_factory=ColorJitter)
    blur: float = 0.0
    temporal_stride: int = 1
    cameras: tuple[CameraToggle, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_scale": self.action_scale,
            "state_scale": self.state_scale,
            "action_offset": self.action_offset,
            "state_offset": self.state_offset,
            "action_noise": self.action_noise,
            "state_noise": self.state_noise,
            "copies": self.copies,
            "crop": None if self.crop is None else self.crop.to_dict(),
            "resize": None if self.resize is None else self.resize.to_dict(),
            "color_jitter": self.color_jitter.to_dict(),
            "blur": self.blur,
            "temporal_stride": self.temporal_stride,
            "cameras": [camera.to_dict() for camera in self.cameras],
        }


@dataclass(frozen=True)
class AxisRange:
    """Inclusive min/max of one action or state dimension."""

    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, float]:
        return {"min": self.minimum, "max": self.maximum}


@dataclass(frozen=True)
class DatasetStats:
    """Stats stored on the version for later display."""

    episodes: int
    frames: int
    action_ranges: tuple[AxisRange, ...]
    state_ranges: tuple[AxisRange, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "frames": self.frames,
            "action_ranges": [axis.to_dict() for axis in self.action_ranges],
            "state_ranges": [axis.to_dict() for axis in self.state_ranges],
        }


@dataclass(frozen=True)
class DatasetSpec:
    """Inputs that define a dataset version before it is materialised."""

    source_query: SourceQuery
    dataset_name: str = "dataset"
    filters: Filters = field(default_factory=Filters)
    split: Split = field(default_factory=Split)
    seed: int = 0
    augmentation: Augmentation = field(default_factory=Augmentation)
    licence: str = "unspecified"


@dataclass(frozen=True)
class CanonicalFrame:
    """One frame in the hashed canonical payload. Numbers are decimal strings."""

    observation_state: tuple[str, ...]
    action: tuple[str, ...]
    action_raw: tuple[str, ...] | None
    timestamp: str
    frame_index: int
    next_done: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": list(self.action),
            "action_raw": None if self.action_raw is None else list(self.action_raw),
            "frame_index": self.frame_index,
            "next_done": self.next_done,
            "observation_state": list(self.observation_state),
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class CanonicalEpisode:
    """One materialised episode in source order, before LeRobot indexing."""

    task: str
    fps: str
    frames: tuple[CanonicalFrame, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "frames": [frame.to_dict() for frame in self.frames],
            "task": self.task,
        }


@dataclass(frozen=True)
class DatasetVersion:
    """One immutable dataset version record."""

    version_id: str
    content_hash: str
    source_query: SourceQuery
    filters: Filters
    split: Split
    seed: int
    augmentation: Augmentation
    stats: DatasetStats
    licence: str
    dataset_name: str
    pinned: bool
    parent_version_id: str | None
    storage_prefix: str
    created_at: str
    schema: str = SCHEMA_NAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version_id": self.version_id,
            "dataset_name": self.dataset_name,
            "parent_version_id": self.parent_version_id,
            "pinned": self.pinned,
            "content_hash": self.content_hash,
            "source_query": self.source_query.to_dict(),
            "filters": self.filters.to_dict(),
            "split": self.split.to_dict(),
            "seed": self.seed,
            "augmentation": self.augmentation.to_dict(),
            "stats": self.stats.to_dict(),
            "licence": self.licence,
            "storage_prefix": self.storage_prefix,
            "created_at": self.created_at,
        }

    def to_spec(self) -> DatasetSpec:
        """Spec that rebuilds this version's identity, ignoring pin metadata."""
        return DatasetSpec(
            source_query=self.source_query,
            dataset_name=self.dataset_name,
            filters=self.filters,
            split=self.split,
            seed=self.seed,
            augmentation=self.augmentation,
            licence=self.licence,
        )


def render_number(value: float) -> str:
    """Format a finite number with 8 decimal places for the content hash."""
    number = float(value)
    if not math.isfinite(number):
        raise DatasetBuildError("frame values must be finite")
    return format(number, ".8f")


def json_number(value: float) -> float:
    """Float that survives a JSON dump/load unchanged."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetBuildError("numbers in a dataset spec must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise DatasetBuildError("numbers in a dataset spec must be finite")
    loaded = json.loads(json.dumps(number))
    if isinstance(loaded, bool) or not isinstance(loaded, (int, float)):
        raise DatasetBuildError("numbers in a dataset spec must be finite")
    return float(loaded)


def canonical_content_bytes(
    episodes: Sequence[CanonicalEpisode],
    split: Split,
    augmentation: Augmentation,
) -> bytes:
    """UTF-8 bytes hashed into ``content_hash``, including a trailing newline.

    The payload includes the episode rows and the split and augmentation
    configs. A config change hashes differently when the stored rows stay
    the same, which is how train-time image settings enter the version hash.
    """
    payload = {
        "algorithm": CONTENT_ALGORITHM,
        "augmentation": augmentation.to_dict(),
        "episodes": [episode.to_dict() for episode in episodes],
        "split": split.to_dict(),
    }
    text = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return (text + "\n").encode("utf-8")


def content_hash_of(
    episodes: Sequence[CanonicalEpisode],
    split: Split,
    augmentation: Augmentation,
) -> str:
    """``sha256:`` hash of :func:`canonical_content_bytes`."""
    digest = hashlib.sha256(
        canonical_content_bytes(episodes, split, augmentation)
    ).hexdigest()
    return f"sha256:{digest}"


def identity_document(spec: DatasetSpec, content_hash: str) -> dict[str, Any]:
    """JSON object hashed to produce a version id. Pin metadata is excluded."""
    return {
        "algorithm": VERSION_ID_ALGORITHM,
        "augmentation": spec.augmentation.to_dict(),
        "content_hash": content_hash,
        "dataset_name": spec.dataset_name,
        "filters": spec.filters.to_dict(),
        "licence": spec.licence,
        "schema": SCHEMA_NAME,
        "seed": spec.seed,
        "source_query": spec.source_query.to_dict(),
        "split": spec.split.to_dict(),
    }


def version_id_for(spec: DatasetSpec, content_hash: str) -> str:
    """``dv_`` plus 32 hex chars of the spec identity. Same spec, same id."""
    if _CONTENT_HASH_RE.fullmatch(content_hash) is None:
        raise DatasetVersionError("content hash is not a sha256 digest")
    raw = json.dumps(
        identity_document(spec, content_hash),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "dv_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def is_version_id(value: str) -> bool:
    """True when ``value`` is a ``dv_`` dataset version id."""
    return _VERSION_ID_RE.fullmatch(value) is not None


def storage_prefix_for(version_id: str) -> str:
    """Object-storage prefix corresponding to the local dataset directory."""
    if _VERSION_ID_RE.fullmatch(version_id) is None:
        raise DatasetVersionError(f"invalid dataset version id: {version_id}")
    return f"versions/{version_id}/dataset"


def stats_from_episodes(episodes: Sequence[CanonicalEpisode]) -> DatasetStats:
    """Episode count, frame count, and per-dimension action/state ranges."""
    frames = [frame for episode in episodes for frame in episode.frames]
    if not frames:
        raise DatasetBuildError("no frames selected")

    def ranges(select) -> tuple[AxisRange, ...]:
        width = len(select(frames[0]))
        axes: list[AxisRange] = []
        for index in range(width):
            column = [float(select(frame)[index]) for frame in frames]
            axes.append(AxisRange(minimum=min(column), maximum=max(column)))
        return tuple(axes)

    return DatasetStats(
        episodes=len(episodes),
        frames=len(frames),
        action_ranges=ranges(lambda frame: frame.action),
        state_ranges=ranges(lambda frame: frame.observation_state),
    )


def assert_identity(record: DatasetVersion) -> None:
    """Raise when the stored id or storage prefix does not match the spec."""
    expected = version_id_for(record.to_spec(), record.content_hash)
    if record.version_id != expected:
        raise DatasetVersionError(
            f"dataset version {record.version_id} does not match its spec"
        )
    prefix = storage_prefix_for(record.version_id)
    if record.storage_prefix != prefix:
        raise DatasetVersionError(
            f"storage prefix does not match dataset version {record.version_id}"
        )


def record_to_json(record: DatasetVersion) -> str:
    """Pretty JSON for ``version.json``, validated against the schema."""
    assert_identity(record)
    document = record.to_dict()
    validate_document(document)
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def version_from_dict(document: Mapping[str, Any]) -> DatasetVersion:
    """Parse a version record after checking it against the schema."""
    if not isinstance(document, Mapping):
        raise DatasetVersionError("dataset version record must be an object")
    validate_document(document)
    filters = document["filters"]
    split = document["split"]
    augmentation = document["augmentation"]
    stats = document["stats"]
    record = DatasetVersion(
        version_id=document["version_id"],
        content_hash=document["content_hash"],
        source_query=SourceQuery(tuple(document["source_query"]["roots"])),
        filters=Filters(
            task_contains=filters["task_contains"],
            robot_type=filters["robot_type"],
            min_frames=filters["min_frames"],
            max_frames=filters["max_frames"],
        ),
        split=Split(
            name=split["name"],
            train=float(split["train"]),
            val=float(split["val"]),
            test=float(split["test"]),
            strategy=split["strategy"],
            held_out_scenes=tuple(split["held_out_scenes"]),
        ),
        seed=int(document["seed"]),
        augmentation=Augmentation(
            action_scale=float(augmentation["action_scale"]),
            state_scale=float(augmentation["state_scale"]),
            action_offset=float(augmentation["action_offset"]),
            state_offset=float(augmentation["state_offset"]),
            action_noise=float(augmentation["action_noise"]),
            state_noise=float(augmentation["state_noise"]),
            copies=int(augmentation["copies"]),
            crop=_image_size(augmentation["crop"]),
            resize=_image_size(augmentation["resize"]),
            color_jitter=_color_jitter(augmentation["color_jitter"]),
            blur=float(augmentation["blur"]),
            temporal_stride=int(augmentation["temporal_stride"]),
            cameras=tuple(
                CameraToggle(name=item["name"], enabled=bool(item["enabled"]))
                for item in augmentation["cameras"]
            ),
        ),
        stats=DatasetStats(
            episodes=int(stats["episodes"]),
            frames=int(stats["frames"]),
            action_ranges=tuple(_axis(item) for item in stats["action_ranges"]),
            state_ranges=tuple(_axis(item) for item in stats["state_ranges"]),
        ),
        licence=document["licence"],
        dataset_name=document["dataset_name"],
        pinned=bool(document["pinned"]),
        parent_version_id=document["parent_version_id"],
        storage_prefix=document["storage_prefix"],
        created_at=document["created_at"],
        schema=document["schema"],
    )
    assert_identity(record)
    return record


def validate_document(document: Mapping[str, Any]) -> None:
    """Raise :class:`DatasetVersionError` when ``document`` breaks the schema.

    The checker covers the keywords this schema uses (type, const, enum,
    required, additionalProperties, items, ``$ref``, pattern, minimum,
    minLength, maxLength, minItems). It is intentionally small so validation
    does not need an extra package.
    """
    schema = load_schema()
    _check(document, schema, "$", schema)


def _axis(item: Mapping[str, Any]) -> AxisRange:
    return AxisRange(minimum=float(item["min"]), maximum=float(item["max"]))


def _image_size(value: Any) -> Crop | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DatasetVersionError("image size must be an object")
    return Crop(height=int(value["height"]), width=int(value["width"]))


def _color_jitter(value: Mapping[str, Any]) -> ColorJitter:
    return ColorJitter(
        brightness=float(value["brightness"]),
        contrast=float(value["contrast"]),
        saturation=float(value["saturation"]),
        hue=float(value["hue"]),
    )


def _check(
    instance: Any, schema: Mapping[str, Any], path: str, root: Mapping[str, Any]
) -> None:
    if "$ref" in schema:
        _check(instance, _resolve_ref(root, str(schema["$ref"])), path, root)
        return
    if "const" in schema and instance != schema["const"]:
        raise DatasetVersionError(f"{path} must equal {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise DatasetVersionError(f"{path} is not an allowed value")
    type_spec = schema.get("type")
    if type_spec is not None and not _type_ok(instance, type_spec):
        raise DatasetVersionError(f"{path} has the wrong type")
    if instance is None:
        return
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < int(schema["minLength"]):
            raise DatasetVersionError(f"{path} is too short")
        if "maxLength" in schema and len(instance) > int(schema["maxLength"]):
            raise DatasetVersionError(f"{path} is too long")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.fullmatch(pattern, instance) is None:
            raise DatasetVersionError(f"{path} has the wrong format")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise DatasetVersionError(f"{path} is below the minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            raise DatasetVersionError(f"{path} is above the maximum")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < int(schema["minItems"]):
            raise DatasetVersionError(f"{path} has too few items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(instance):
                _check(item, item_schema, f"{path}[{index}]", root)
    if isinstance(instance, Mapping) and "properties" in schema:
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise DatasetVersionError("schema properties must be an object")
        if schema.get("additionalProperties", True) is False:
            unknown = sorted(set(instance) - set(properties))
            if unknown:
                names = ", ".join(unknown)
                raise DatasetVersionError(f"{path} has unknown fields: {names}")
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in instance:
                    raise DatasetVersionError(f"{path} is missing {key}")
        for key, sub in properties.items():
            if key in instance and isinstance(sub, Mapping):
                _check(instance[key], sub, f"{path}.{key}", root)


def _resolve_ref(root: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    prefix = "#/$defs/"
    if not ref.startswith(prefix):
        raise DatasetVersionError(f"unsupported schema reference {ref}")
    defs = root.get("$defs", {})
    if not isinstance(defs, Mapping) or ref[len(prefix) :] not in defs:
        raise DatasetVersionError(f"unknown schema reference {ref}")
    target = defs[ref[len(prefix) :]]
    if not isinstance(target, Mapping):
        raise DatasetVersionError(f"schema reference {ref} is not an object")
    return target


def _type_ok(instance: Any, type_spec: Any) -> bool:
    names = (type_spec,) if isinstance(type_spec, str) else tuple(type_spec)
    return any(_one_type(instance, name) for name in names)


def _one_type(instance: Any, name: str) -> bool:
    if name == "null":
        return instance is None
    if name == "string":
        return isinstance(instance, str)
    if name == "boolean":
        return isinstance(instance, bool)
    if name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if name == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if name == "array":
        return isinstance(instance, list)
    if name == "object":
        return isinstance(instance, Mapping)
    return False
