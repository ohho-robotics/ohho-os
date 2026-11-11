"""TargetProfile documents: JSON, YAML, and the built-in catalog.

Built-in profiles are package data under ``ohho/targets/profiles``. The
Python objects are loaded from those files so the console and the runtime
share one definition. Cross-field rules (FP8 on Orin, TensorRT on a CPU
target, workspace versus memory) are enforced here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ohho.targets.errors import TargetCombinationError, TargetProfileError
from ohho.targets.providers import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    PROVIDER_NAMES,
    TRT_PROVIDER,
    provider_preference,
)

SCHEMA_ID = "ohho.target_profile.v1"
MODEL_CLASSES = ("act", "smolvla", "diffusion", "openvla")
PRECISIONS = ("FP32", "FP16", "INT8", "FP8", "INT4")
ACCELERATORS = ("none", "orin_gpu", "dgpu", "npu")
ARCHES = ("aarch64", "x86_64")
BUILTIN_IDS = (
    "jetson-orin-nano",
    "jetson-orin-nx",
    "jetson-agx-orin",
    "rpi5-cpu",
    "x86-cpu",
    "x86-cuda",
    "cloud-l4",
    "cloud-a10g",
    "cloud-h100",
)

_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_PROFILE_KEYS = frozenset(
    {
        "schema",
        "id",
        "arch",
        "accelerator",
        "precisions",
        "execution_providers",
        "tensorrt",
        "memory_budget_mib",
        "workspace_size_mib",
        "slos_p95_ms",
        "summary",
    }
)
_TENSORRT_KEYS = frozenset({"min_version", "before_version", "excluded"})
_BOUND_KEYS = frozenset({"min_version", "before_version"})
_ALLOWED_PRECISIONS = {
    "orin_gpu": frozenset({"FP32", "FP16", "INT8"}),
    "dgpu": frozenset(PRECISIONS),
    "none": frozenset({"FP32", "INT8"}),
    "npu": frozenset({"FP32", "INT8"}),
}
_MAX_MEMORY_MIB = 1_048_576
_MAX_SLO_MS = 3_600_000.0


def fp8_on_orin_message(profile_id: str) -> str:
    """Rejection text for FP8 on an Orin GPU."""
    return (
        f"FP8 is not valid on {profile_id}. "
        "Orin GPUs support FP32, FP16 and INT8. "
        "FP8 needs an Ada or Hopper GPU."
    )


def int8_tensorrt_message(profile_id: str, providers: Sequence[str]) -> str:
    """Rejection text for INT8 TensorRT on a target without TensorRT."""
    return (
        f"INT8 TensorRT is not valid on {profile_id}. "
        "That target has no TensorRT execution provider "
        f"({', '.join(providers)} only)."
    )


def parse_version(text: str) -> tuple[int, int, int]:
    """Parse ``major.minor.patch``. Raises ``TargetProfileError`` otherwise."""
    match = _VERSION_RE.fullmatch(text)
    if match is None:
        raise TargetProfileError(
            f"Version {text!r} must be a dotted triple such as 10.16.2."
        )
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


@dataclass(frozen=True)
class VersionBound:
    """An inclusive lower version and an exclusive upper version."""

    min_version: str
    before_version: str

    def __post_init__(self) -> None:
        low = parse_version(self.min_version)
        high = parse_version(self.before_version)
        if low >= high:
            raise TargetProfileError(
                "min_version must be lower than before_version "
                f"(got {self.min_version} .. {self.before_version})."
            )

    def contains(self, version: tuple[int, int, int]) -> bool:
        return (
            parse_version(self.min_version)
            <= version
            < parse_version(self.before_version)
        )


@dataclass(frozen=True)
class TensorRTConstraint:
    """TensorRT versions a target can build engines with."""

    min_version: str
    before_version: str
    excluded: tuple[VersionBound, ...] = ()

    def __post_init__(self) -> None:
        low = parse_version(self.min_version)
        high = parse_version(self.before_version)
        if low >= high:
            raise TargetProfileError(
                "tensorrt min_version must be lower than before_version "
                f"(got {self.min_version} .. {self.before_version})."
            )
        object.__setattr__(self, "excluded", tuple(self.excluded))
        for hole in self.excluded:
            if not isinstance(hole, VersionBound):
                raise TargetProfileError(
                    "tensorrt excluded entries must be version ranges."
                )
            if (
                parse_version(hole.min_version) < low
                or parse_version(hole.before_version) > high
            ):
                raise TargetProfileError(
                    "tensorrt excluded ranges must sit inside "
                    f"{self.min_version} .. {self.before_version}."
                )

    def allows(self, version: str) -> bool:
        """Return whether ``version`` may be used on this target."""
        got = parse_version(version)
        if not (
            parse_version(self.min_version) <= got < parse_version(self.before_version)
        ):
            return False
        return not any(hole.contains(got) for hole in self.excluded)

    def describe(self) -> str:
        """Short text of the allowed window, including exclusions."""
        text = f">={self.min_version},<{self.before_version}"
        if not self.excluded:
            return text
        holes = ", ".join(
            f">={item.min_version},<{item.before_version}" for item in self.excluded
        )
        return f"{text} excluding {holes}"


@dataclass(frozen=True)
class TargetProfile:
    """One optimisation and benchmark target.

    ``slos_p95_ms`` values are default budgets, not measured latencies.
    ``memory_budget_mib`` is the memory the optimiser may plan to use.
    """

    id: str
    arch: str
    accelerator: str
    precisions: tuple[str, ...]
    execution_providers: tuple[str, ...]
    tensorrt: TensorRTConstraint | None
    memory_budget_mib: int
    workspace_size_mib: int
    slos_p95_ms: Mapping[str, float]
    summary: str
    schema: str = SCHEMA_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "precisions", tuple(self.precisions))
        object.__setattr__(self, "execution_providers", tuple(self.execution_providers))
        object.__setattr__(
            self, "slos_p95_ms", MappingProxyType(dict(self.slos_p95_ms))
        )
        _validate_profile(self)


def _validate_profile(profile: TargetProfile) -> None:
    if profile.schema != SCHEMA_ID:
        raise TargetProfileError(
            f"schema must be {SCHEMA_ID!r} (got {profile.schema!r})."
        )
    if not isinstance(profile.id, str) or _ID_RE.fullmatch(profile.id) is None:
        raise TargetProfileError(
            f"Target id {profile.id!r} must be a lowercase slug such as rpi5-cpu."
        )
    if profile.arch not in ARCHES:
        raise TargetProfileError(
            f"{profile.id}: arch must be aarch64 or x86_64 (got {profile.arch!r})."
        )
    if profile.accelerator not in ACCELERATORS:
        known = ", ".join(ACCELERATORS)
        raise TargetProfileError(
            f"{profile.id}: accelerator must be one of: {known} "
            f"(got {profile.accelerator!r})."
        )
    if profile.accelerator == "orin_gpu" and profile.arch != "aarch64":
        raise TargetProfileError(f"{profile.id}: an Orin GPU target must be aarch64.")
    _validate_precisions(profile)
    _validate_providers(profile)
    _validate_budget(profile)
    _validate_slos(profile)
    if not isinstance(profile.summary, str) or not profile.summary.strip():
        raise TargetProfileError(f"{profile.id}: summary must be a non-empty string.")
    if "\n" in profile.summary or len(profile.summary) > 400:
        raise TargetProfileError(
            f"{profile.id}: summary must be one line of at most 400 characters."
        )


def _validate_precisions(profile: TargetProfile) -> None:
    if not profile.precisions or not all(
        isinstance(item, str) for item in profile.precisions
    ):
        raise TargetProfileError(f"{profile.id}: precisions must be a list of strings.")
    if len(set(profile.precisions)) != len(profile.precisions):
        raise TargetProfileError(f"{profile.id}: precisions must not repeat.")
    unknown = [item for item in profile.precisions if item not in PRECISIONS]
    if unknown:
        known = ", ".join(PRECISIONS)
        raise TargetProfileError(
            f"{profile.id}: unknown precision {', '.join(unknown)}. Expected {known}."
        )
    if "FP32" not in profile.precisions:
        raise TargetProfileError(f"{profile.id}: FP32 must be a supported precision.")
    if profile.accelerator == "orin_gpu" and "FP8" in profile.precisions:
        raise TargetProfileError(fp8_on_orin_message(profile.id))
    allowed = _ALLOWED_PRECISIONS[profile.accelerator]
    extra = [item for item in profile.precisions if item not in allowed]
    if extra:
        listed = ", ".join(item for item in PRECISIONS if item in allowed)
        raise TargetProfileError(
            f"{profile.id}: {', '.join(extra)} is not valid on accelerator "
            f"{profile.accelerator}. Allowed precisions: {listed}."
        )


def _validate_providers(profile: TargetProfile) -> None:
    providers = profile.execution_providers
    if not providers or not all(isinstance(item, str) for item in providers):
        raise TargetProfileError(
            f"{profile.id}: execution_providers must be a list of strings."
        )
    if len(set(providers)) != len(providers):
        raise TargetProfileError(f"{profile.id}: execution providers must not repeat.")
    unknown = [item for item in providers if item not in PROVIDER_NAMES]
    if unknown:
        known = ", ".join(PROVIDER_NAMES)
        raise TargetProfileError(
            f"{profile.id}: unknown execution provider {', '.join(unknown)}. "
            f"Expected {known}."
        )
    if providers[-1] != CPU_PROVIDER:
        raise TargetProfileError(
            f"{profile.id}: {CPU_PROVIDER} must be the last execution provider."
        )
    if profile.accelerator == "orin_gpu":
        expected = provider_preference("orin_gpu")
        if providers != expected:
            raise TargetProfileError(
                f"{profile.id}: Orin execution providers must be {', '.join(expected)}."
            )
    elif profile.accelerator == "dgpu":
        body = providers[:-1]
        if not body or CPU_PROVIDER in body:
            raise TargetProfileError(
                f"{profile.id}: a dGPU target must list CUDA or TensorRT before CPU."
            )
        if CUDA_PROVIDER not in body and TRT_PROVIDER not in body:
            raise TargetProfileError(
                f"{profile.id}: a dGPU target must list CUDA or TensorRT."
            )
    else:
        expected = provider_preference(profile.accelerator)
        if providers != expected:
            raise TargetProfileError(
                f"{profile.id}: execution providers must be {', '.join(expected)}."
            )
    has_trt = TRT_PROVIDER in providers
    if has_trt and profile.tensorrt is None:
        raise TargetProfileError(
            f"{profile.id}: {TRT_PROVIDER} requires a tensorrt version range."
        )
    if profile.tensorrt is not None and not has_trt:
        raise TargetProfileError(
            f"{profile.id}: a tensorrt version range requires {TRT_PROVIDER}."
        )
    if profile.tensorrt is not None and not isinstance(
        profile.tensorrt, TensorRTConstraint
    ):
        raise TargetProfileError(f"{profile.id}: tensorrt must be a version range.")


def _validate_budget(profile: TargetProfile) -> None:
    memory = profile.memory_budget_mib
    workspace = profile.workspace_size_mib
    if type(memory) is not int or not 1 <= memory <= _MAX_MEMORY_MIB:
        raise TargetProfileError(
            f"{profile.id}: memory_budget_mib must be an integer from 1 to "
            f"{_MAX_MEMORY_MIB}."
        )
    if type(workspace) is not int or workspace < 0 or workspace > memory:
        raise TargetProfileError(
            f"{profile.id}: workspace_size_mib must be an integer from 0 to "
            "memory_budget_mib."
        )
    if profile.tensorrt is None and workspace != 0:
        raise TargetProfileError(
            f"{profile.id}: workspace_size_mib must be 0 when TensorRT is not used."
        )
    if profile.tensorrt is not None and workspace < 1:
        raise TargetProfileError(
            f"{profile.id}: workspace_size_mib must be positive when TensorRT is used."
        )


def _validate_slos(profile: TargetProfile) -> None:
    slos = profile.slos_p95_ms
    missing = [name for name in MODEL_CLASSES if name not in slos]
    if missing:
        raise TargetProfileError(
            f"{profile.id}: slos_p95_ms is missing {', '.join(missing)}."
        )
    extra = [name for name in slos if name not in MODEL_CLASSES]
    if extra:
        known = ", ".join(MODEL_CLASSES)
        raise TargetProfileError(
            f"{profile.id}: unknown model class {', '.join(extra)}. Expected {known}."
        )
    for name in MODEL_CLASSES:
        value = slos[name]
        if type(value) not in (int, float):
            raise TargetProfileError(
                f"{profile.id}: slos_p95_ms.{name} must be a number."
            )
        number = float(value)
        if number <= 0 or number > _MAX_SLO_MS or number != number:
            raise TargetProfileError(
                f"{profile.id}: slos_p95_ms.{name} must be a positive finite "
                f"number up to {_MAX_SLO_MS:g} ms."
            )


def _json_number(value: float) -> int | float:
    if float(value).is_integer():
        return int(value)
    return float(value)


def profile_to_dict(profile: TargetProfile) -> dict[str, Any]:
    """Return the JSON object stored for ``profile``."""
    if profile.tensorrt is None:
        tensorrt: Any = None
    else:
        tensorrt = {
            "min_version": profile.tensorrt.min_version,
            "before_version": profile.tensorrt.before_version,
            "excluded": [
                {
                    "min_version": item.min_version,
                    "before_version": item.before_version,
                }
                for item in profile.tensorrt.excluded
            ],
        }
    return {
        "schema": SCHEMA_ID,
        "id": profile.id,
        "arch": profile.arch,
        "accelerator": profile.accelerator,
        "precisions": list(profile.precisions),
        "execution_providers": list(profile.execution_providers),
        "tensorrt": tensorrt,
        "memory_budget_mib": profile.memory_budget_mib,
        "workspace_size_mib": profile.workspace_size_mib,
        "slos_p95_ms": {
            name: _json_number(float(profile.slos_p95_ms[name]))
            for name in MODEL_CLASSES
        },
        "summary": profile.summary,
    }


def _bound_from_dict(data: object, label: str) -> VersionBound:
    if not isinstance(data, dict):
        raise TargetProfileError(f"{label} must be an object.")
    extra = set(data) - _BOUND_KEYS
    if extra:
        raise TargetProfileError(
            f"{label} has unknown fields: {', '.join(sorted(extra))}."
        )
    for key in ("min_version", "before_version"):
        if not isinstance(data.get(key), str):
            raise TargetProfileError(f"{label}.{key} must be a string.")
    return VersionBound(
        min_version=data["min_version"], before_version=data["before_version"]
    )


def _tensorrt_from_dict(data: object) -> TensorRTConstraint | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise TargetProfileError("tensorrt must be an object or null.")
    extra = set(data) - _TENSORRT_KEYS
    if extra:
        raise TargetProfileError(
            f"tensorrt has unknown fields: {', '.join(sorted(extra))}."
        )
    for key in ("min_version", "before_version"):
        if not isinstance(data.get(key), str):
            raise TargetProfileError(f"tensorrt.{key} must be a string.")
    excluded_raw = data.get("excluded", [])
    if not isinstance(excluded_raw, list):
        raise TargetProfileError("tensorrt.excluded must be a list.")
    excluded = tuple(
        _bound_from_dict(item, "tensorrt.excluded") for item in excluded_raw
    )
    return TensorRTConstraint(
        min_version=data["min_version"],
        before_version=data["before_version"],
        excluded=excluded,
    )


def _string_tuple(data: object, label: str) -> tuple[str, ...]:
    if not isinstance(data, list) or not data:
        raise TargetProfileError(f"{label} must be a non-empty list.")
    if not all(isinstance(item, str) for item in data):
        raise TargetProfileError(f"{label} must be a list of strings.")
    return tuple(data)


def profile_from_dict(data: object) -> TargetProfile:
    """Validate a decoded JSON or YAML object and return a profile."""
    if not isinstance(data, dict):
        raise TargetProfileError("A target profile must be an object.")
    extra = set(data) - _PROFILE_KEYS
    if extra:
        raise TargetProfileError(
            f"Unknown target profile fields: {', '.join(sorted(extra))}."
        )
    missing = sorted(_PROFILE_KEYS - set(data))
    if missing:
        raise TargetProfileError(f"Target profile is missing {', '.join(missing)}.")
    slos = data["slos_p95_ms"]
    if not isinstance(slos, dict):
        raise TargetProfileError("slos_p95_ms must be an object.")
    return TargetProfile(
        schema=data["schema"],
        id=data["id"],
        arch=data["arch"],
        accelerator=data["accelerator"],
        precisions=_string_tuple(data["precisions"], "precisions"),
        execution_providers=_string_tuple(
            data["execution_providers"], "execution_providers"
        ),
        tensorrt=_tensorrt_from_dict(data["tensorrt"]),
        memory_budget_mib=data["memory_budget_mib"],
        workspace_size_mib=data["workspace_size_mib"],
        slos_p95_ms=slos,
        summary=data["summary"],
    )


def profile_from_json(text: str) -> TargetProfile:
    """Parse one JSON target profile."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TargetProfileError(f"Target profile JSON is not valid: {exc}") from exc
    return profile_from_dict(data)


def profile_from_yaml(text: str) -> TargetProfile:
    """Parse one YAML target profile. Requires the ``yaml`` extra."""
    try:
        import yaml
    except ImportError as exc:
        raise TargetProfileError(
            "Reading a YAML target profile needs PyYAML: pip install 'ohho-os[yaml]'."
        ) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise TargetProfileError(f"Target profile YAML is not valid: {exc}") from exc
    return profile_from_dict(data)


def load_profile(path: str | Path) -> TargetProfile:
    """Load a ``.json``, ``.yaml`` or ``.yml`` target profile from disk."""
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise TargetProfileError(
            f"Could not read {file}: {exc.strerror or exc}"
        ) from exc
    suffix = file.suffix.lower()
    if suffix == ".json":
        return profile_from_json(text)
    if suffix in {".yaml", ".yml"}:
        return profile_from_yaml(text)
    raise TargetProfileError(
        f"{file.name} must be a .json, .yaml or .yml target profile."
    )


_CACHE: dict[str, TargetProfile] | None = None


def _read_resource(name: str) -> str:
    path = files("ohho.targets").joinpath("profiles").joinpath(name)
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _builtins() -> dict[str, TargetProfile]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    loaded: dict[str, TargetProfile] = {}
    for profile_id in BUILTIN_IDS:
        filename = f"{profile_id}.json"
        try:
            text = _read_resource(filename)
        except OSError as exc:
            raise TargetProfileError(
                f"Built-in target profile {profile_id} is not in package data."
            ) from exc
        profile = profile_from_json(text)
        if profile.id != profile_id:
            raise TargetProfileError(
                f"{filename} declares id {profile.id!r}, expected {profile_id!r}."
            )
        loaded[profile_id] = profile
    _CACHE = loaded
    return loaded


def list_profiles() -> list[str]:
    """Built-in target ids, in catalog order."""
    _builtins()
    return list(BUILTIN_IDS)


def get_profile(profile_id: str) -> TargetProfile:
    """Return one built-in target profile."""
    loaded = _builtins()
    try:
        return loaded[profile_id]
    except KeyError:
        known = ", ".join(BUILTIN_IDS)
        raise TargetProfileError(
            f"Unknown target profile {profile_id!r}. Available: {known}."
        ) from None


def catalog_document() -> dict[str, Any]:
    """JSON catalog of every built-in profile, for ``ohho targets list --json``."""
    return {
        "schema": SCHEMA_ID,
        "profiles": [
            profile_to_dict(get_profile(profile_id)) for profile_id in BUILTIN_IDS
        ],
    }


def schema_text() -> str:
    """The JSON Schema document shipped beside the profiles."""
    path = files("ohho.targets").joinpath("target_profile.schema.json")
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _canonical_precision(value: str) -> str:
    text = value.strip().upper()
    if text not in PRECISIONS:
        known = ", ".join(PRECISIONS)
        raise TargetCombinationError(
            f"Unknown precision {value!r}. Expected one of: {known}."
        )
    return text


def _canonical_provider(value: str) -> str:
    key = "".join(ch for ch in value.strip().lower() if ch.isalnum())
    aliases = {
        "tensorrt": TRT_PROVIDER,
        "trt": TRT_PROVIDER,
        "tensorrtexecutionprovider": TRT_PROVIDER,
        "cuda": CUDA_PROVIDER,
        "cudaexecutionprovider": CUDA_PROVIDER,
        "cpu": CPU_PROVIDER,
        "cpuexecutionprovider": CPU_PROVIDER,
    }
    try:
        return aliases[key]
    except KeyError:
        known = ", ".join(PROVIDER_NAMES)
        raise TargetCombinationError(
            f"Unknown execution provider {value!r}. Expected {known}."
        ) from None


def check_combination(
    profile: TargetProfile | str,
    precision: str,
    provider: str,
    *,
    tensorrt_version: str = "",
) -> tuple[str, str]:
    """Accept a precision and provider, or raise ``TargetCombinationError``.

    Returns the canonical precision and ONNX Runtime provider name.
    """
    canonical_precision = _canonical_precision(precision)
    canonical_provider = _canonical_provider(provider)
    selected = get_profile(profile) if isinstance(profile, str) else profile
    version = tensorrt_version.strip()
    if version and canonical_provider != TRT_PROVIDER:
        raise TargetCombinationError(
            "A TensorRT version applies only when the execution provider is TensorRT."
        )
    if canonical_precision == "FP8" and selected.accelerator == "orin_gpu":
        raise TargetCombinationError(fp8_on_orin_message(selected.id))
    if (
        canonical_precision == "INT8"
        and canonical_provider == TRT_PROVIDER
        and TRT_PROVIDER not in selected.execution_providers
    ):
        raise TargetCombinationError(
            int8_tensorrt_message(selected.id, selected.execution_providers)
        )
    reasons: list[str] = []
    if canonical_precision not in selected.precisions:
        reasons.append(
            f"{canonical_precision} is not valid on {selected.id}. "
            f"Supported precisions: {', '.join(selected.precisions)}."
        )
    if canonical_provider not in selected.execution_providers:
        reasons.append(
            f"{canonical_provider} is not valid on {selected.id}. "
            f"Execution providers: {', '.join(selected.execution_providers)}."
        )
    if canonical_provider == TRT_PROVIDER and version and selected.tensorrt is not None:
        try:
            allowed = selected.tensorrt.allows(version)
        except TargetProfileError as exc:
            reasons.append(str(exc))
        else:
            if not allowed:
                reasons.append(
                    f"TensorRT {version} is not valid on {selected.id}. "
                    f"This target requires {selected.tensorrt.describe()}."
                )
    if reasons:
        raise TargetCombinationError(" ".join(reasons))
    return canonical_precision, canonical_provider
