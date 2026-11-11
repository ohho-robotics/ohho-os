"""Hardware target profiles for optimisation and benchmarking.

Import this package by path (``import ohho.targets``). ``ohho.serve`` does
not re-export it. Built-in profiles are JSON package data under
``ohho/targets/profiles``.
"""

from __future__ import annotations

from ohho.targets.detect import (
    DETECTION_SCHEMA,
    Detection,
    MachineProbe,
    detect_target,
    probe_machine,
)
from ohho.targets.errors import TargetCombinationError, TargetProfileError
from ohho.targets.providers import (
    onnx_providers,
    provider_preference,
    workstation_provider_order,
)
from ohho.targets.schema import (
    ACCELERATORS,
    ARCHES,
    BUILTIN_IDS,
    MODEL_CLASSES,
    PRECISIONS,
    SCHEMA_ID,
    TargetProfile,
    TensorRTConstraint,
    catalog_document,
    check_combination,
    get_profile,
    list_profiles,
    load_profile,
    profile_from_dict,
    profile_to_dict,
)

__all__ = [
    "ACCELERATORS",
    "ARCHES",
    "BUILTIN_IDS",
    "DETECTION_SCHEMA",
    "MODEL_CLASSES",
    "PRECISIONS",
    "SCHEMA_ID",
    "Detection",
    "MachineProbe",
    "TargetCombinationError",
    "TargetProfile",
    "TargetProfileError",
    "TensorRTConstraint",
    "catalog_document",
    "check_combination",
    "detect_target",
    "get_profile",
    "list_profiles",
    "load_profile",
    "onnx_providers",
    "probe_machine",
    "profile_from_dict",
    "profile_to_dict",
    "provider_preference",
    "workstation_provider_order",
]
