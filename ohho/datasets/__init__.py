"""Dataset versions: build, pin, and stats for local episodes.

A version is an immutable, content-hashed LeRobot dataset built from local
episodes. The content hash covers the episode rows and the split and
augmentation config. Training code keeps taking a filesystem path; call
:func:`resolve_dataset_version` to turn a version id into that path, then
:func:`apply_augmentation` inside the data loader.

See ``docs/dataset-versions.md`` for the on-disk layout, the preview payload,
and how ACT and Diffusion Policy loaders call the augmentation.
"""

from __future__ import annotations

from .augment import (
    PREVIEW_SCHEMA,
    AugmentationError,
    apply_augmentation,
    materialised_apply_config,
    preview_augmentation,
)
from .build import build_version, edit_version, prepare_spec
from .model import (
    SCHEMA_NAME,
    Augmentation,
    AxisRange,
    CameraToggle,
    ColorJitter,
    Crop,
    DatasetBuildError,
    DatasetSpec,
    DatasetStats,
    DatasetVersion,
    DatasetVersionError,
    DatasetVersionFrozen,
    DatasetVersionNotFound,
    Filters,
    SourceQuery,
    Split,
    schema_path,
)
from .split import SPLIT_STRATEGIES, SplitAssignment, SplitEpisode, assign_splits
from .store import (
    list_versions,
    load_version,
    overwrite_version,
    pin_version,
    resolve_dataset_version,
    resolve_store,
    version_stats,
)

__all__ = [
    "PREVIEW_SCHEMA",
    "SCHEMA_NAME",
    "SPLIT_STRATEGIES",
    "Augmentation",
    "AugmentationError",
    "AxisRange",
    "CameraToggle",
    "ColorJitter",
    "Crop",
    "DatasetBuildError",
    "DatasetSpec",
    "DatasetStats",
    "DatasetVersion",
    "DatasetVersionError",
    "DatasetVersionFrozen",
    "DatasetVersionNotFound",
    "Filters",
    "SourceQuery",
    "Split",
    "SplitAssignment",
    "SplitEpisode",
    "apply_augmentation",
    "assign_splits",
    "build_version",
    "edit_version",
    "list_versions",
    "load_version",
    "materialised_apply_config",
    "overwrite_version",
    "pin_version",
    "prepare_spec",
    "preview_augmentation",
    "resolve_dataset_version",
    "resolve_store",
    "schema_path",
    "version_stats",
]
