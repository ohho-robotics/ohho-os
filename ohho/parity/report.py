"""``ohho.parity/v1`` report.

The same document is what ``ohho parity --json`` prints and what
``ohho optimize`` stores under ``parity`` in ``manifest.json`` and in
``parity.json``. Harness, console, and CI can render ``result``, ``reasons``,
and ``outputs`` without another schema.
"""

from __future__ import annotations

import re
from importlib.resources import files
from typing import Any

from ohho.parity.errors import ParityError

SCHEMA_ID = "ohho.parity/v1"
SCHEMA_URL = "https://ohho-robotics.com/schemas/parity-report.v1.json"

_TOP = (
    "schema",
    "result",
    "reasons",
    "reference",
    "candidate",
    "target",
    "precision",
    "thresholds",
    "inputs",
    "parity_set",
    "outputs",
    "unchecked",
    "allow_parity_fail",
)
_REFERENCE = ("ref", "kind", "family")
_CANDIDATE = ("path", "format")
_FORMATS = frozenset({"onnx", "torchscript", "module"})
_INPUTS = (
    "source",
    "path",
    "seed",
    "n_requested",
    "n",
    "indices",
    "sha256",
    "dtype",
)
_SOURCES = frozenset({"random", "dataset", "recorded", "camera"})
_PARITY_SET = ("id", "version", "sha256", "synthetic", "source")
_SET_SOURCES = frozenset({"synthetic", "dataset", "recorded", "camera"})
_SET_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_THRESHOLDS = (
    "mean_abs",
    "max_abs",
    "rmse",
    "cosine_min",
    "top1_min",
    "kl_max",
    "per_joint",
    "action_chunk",
)
_OUTPUT = (
    "name",
    "shape",
    "mean_abs",
    "max_abs",
    "rmse",
    "cosine",
    "per_joint",
    "action_chunk",
    "top1",
    "kl",
)
_UNCHECKED = ("role", "path", "reason")
_HEX = frozenset("0123456789abcdef")


def schema_text() -> str:
    """The JSON Schema document shipped with the package."""
    path = files("ohho.parity").joinpath("parity_report.schema.json")
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def build_report(
    *,
    result: str,
    reasons: list[str],
    reference: dict[str, Any],
    candidate: dict[str, Any],
    target: str | None,
    precision: str | None,
    thresholds: dict[str, Any],
    inputs: dict[str, Any],
    outputs: list[dict[str, Any]],
    unchecked: list[dict[str, Any]] | None = None,
    allow_parity_fail: bool = False,
    parity_set: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a report and reject a document that does not match the schema.

    ``parity_set`` records the golden set that was scored. Callers that do not
    have one yet receive the unscored synthetic record; ``ohho parity`` and
    ``ohho optimize`` always pass the set they actually used.
    """
    if parity_set is None:
        from ohho.parity.sets import unscored_parity_set

        parity_set = unscored_parity_set()
    document = {
        "schema": SCHEMA_ID,
        "result": result,
        "reasons": reasons,
        "reference": reference,
        "candidate": candidate,
        "target": target,
        "precision": precision,
        "thresholds": thresholds,
        "inputs": inputs,
        "parity_set": parity_set,
        "outputs": outputs,
        "unchecked": list(unchecked or []),
        "allow_parity_fail": allow_parity_fail,
    }
    problems = report_problems(document)
    if problems:
        raise ParityError("parity report is invalid: " + "; ".join(problems))
    return document


def report_problems(document: object) -> list[str]:
    """Return schema problems. An empty list means the document matches."""
    if not isinstance(document, dict):
        return ["report must be an object"]
    problems: list[str] = []
    missing = [key for key in _TOP if key not in document]
    extra = [key for key in document if key not in _TOP]
    if missing:
        problems.append("missing " + ", ".join(missing))
    if extra:
        problems.append("unknown " + ", ".join(sorted(extra)))
    if document.get("schema") != SCHEMA_ID:
        problems.append(f"schema must be {SCHEMA_ID}")
    result = document.get("result")
    if result not in {"pass", "fail"}:
        problems.append("result must be pass or fail")
    reasons = document.get("reasons")
    if not isinstance(reasons, list) or not all(
        isinstance(item, str) and item for item in reasons
    ):
        problems.append("reasons must be a list of strings")
    elif result == "pass" and reasons:
        problems.append("a passing report has no reasons")
    elif result == "fail" and not reasons:
        problems.append("a failing report needs at least one reason")
    _object_keys(document.get("reference"), _REFERENCE, "reference", problems)
    _strings(document.get("reference"), _REFERENCE, "reference", problems)
    _candidate(document.get("candidate"), problems)
    _optional_text(document.get("target"), "target", problems)
    _optional_text(document.get("precision"), "precision", problems)
    _thresholds(document.get("thresholds"), problems)
    _inputs(document.get("inputs"), problems)
    _parity_set(document.get("parity_set"), problems)
    _outputs(document.get("outputs"), result, problems)
    _unchecked(document.get("unchecked"), problems)
    if type(document.get("allow_parity_fail")) is not bool:
        problems.append("allow_parity_fail must be a boolean")
    return problems


def _candidate(value: object, problems: list[str]) -> None:
    if not isinstance(value, dict):
        problems.append("candidate must be an object")
        return
    missing = [key for key in _CANDIDATE if key not in value]
    extra = [key for key in value if key not in _CANDIDATE]
    if missing:
        problems.append("candidate missing " + ", ".join(missing))
    if extra:
        problems.append("candidate unknown " + ", ".join(sorted(extra)))
    path = value.get("path")
    if path is not None and not isinstance(path, str):
        problems.append("candidate.path must be a string or null")
    fmt = value.get("format")
    if fmt is not None and fmt not in _FORMATS:
        problems.append("candidate.format must be onnx, torchscript, module, or null")


def _thresholds(value: object, problems: list[str]) -> None:
    if not _object_keys(value, _THRESHOLDS, "thresholds", problems):
        return
    assert isinstance(value, dict)
    mean = value.get("mean_abs")
    if not _finite(mean) or float(mean) < 0:
        problems.append("thresholds.mean_abs must be >= 0")
    _optional_limit(value.get("max_abs"), "thresholds.max_abs", 0.0, None, problems)
    _optional_limit(value.get("rmse"), "thresholds.rmse", 0.0, None, problems)
    _optional_limit(
        value.get("cosine_min"), "thresholds.cosine_min", -1.0, 1.0, problems
    )
    _optional_limit(value.get("top1_min"), "thresholds.top1_min", 0.0, 1.0, problems)
    _optional_limit(value.get("kl_max"), "thresholds.kl_max", 0.0, None, problems)
    _optional_limit(value.get("per_joint"), "thresholds.per_joint", 0.0, None, problems)
    _optional_limit(
        value.get("action_chunk"), "thresholds.action_chunk", 0.0, None, problems
    )


def _inputs(value: object, problems: list[str]) -> None:
    if not _object_keys(value, _INPUTS, "inputs", problems):
        return
    assert isinstance(value, dict)
    source = value.get("source")
    if source not in _SOURCES:
        problems.append("inputs.source must be random, dataset, recorded, or camera")
    path = value.get("path")
    if path is not None and not isinstance(path, str):
        problems.append("inputs.path must be a string or null")
    if type(value.get("seed")) is not int:
        problems.append("inputs.seed must be an integer")
    n_requested = value.get("n_requested")
    count = value.get("n")
    if type(n_requested) is not int or n_requested < 1:
        problems.append("inputs.n_requested must be an integer >= 1")
    if type(count) is not int or count < 0:
        problems.append("inputs.n must be an integer >= 0")
    elif type(n_requested) is int and count > n_requested:
        problems.append("inputs.n cannot exceed n_requested")
    indices = value.get("indices")
    if source == "random" or count == 0:
        if indices is not None:
            problems.append(
                "inputs.indices must be null for random inputs or when n is 0"
            )
    elif not isinstance(indices, list) or len(indices) != count:
        problems.append("inputs.indices must list the frames that were scored")
    elif any(type(item) is not int or item < 0 for item in indices):
        problems.append("inputs.indices must be non-negative integers")
    digest = value.get("sha256")
    if count == 0:
        if digest is not None:
            problems.append("inputs.sha256 must be null when n is 0")
    elif (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(ch not in _HEX for ch in digest)
    ):
        problems.append("inputs.sha256 must be 64 lowercase hex characters")
    if value.get("dtype") != "float32":
        problems.append("inputs.dtype must be float32")


def _parity_set(value: object, problems: list[str]) -> None:
    if not _object_keys(value, _PARITY_SET, "parity_set", problems):
        return
    assert isinstance(value, dict)
    set_id = value.get("id")
    if not isinstance(set_id, str) or _SET_ID.fullmatch(set_id) is None:
        problems.append("parity_set.id must be a lowercase slug")
        set_id = ""
    version = value.get("version")
    if type(version) is not int or version < 1:
        problems.append("parity_set.version must be an integer >= 1")
    synthetic = value.get("synthetic")
    if type(synthetic) is not bool:
        problems.append("parity_set.synthetic must be a boolean")
        synthetic = None
    source = value.get("source")
    if source not in _SET_SOURCES:
        problems.append(
            "parity_set.source must be synthetic, dataset, recorded, or camera"
        )
    if synthetic is True and (set_id != "synthetic" or source != "synthetic"):
        problems.append("a synthetic parity set has id synthetic and source synthetic")
    if synthetic is False and (set_id == "synthetic" or source == "synthetic"):
        problems.append("a captured parity set is not synthetic")
    digest = value.get("sha256")
    if digest is None:
        if synthetic is not True:
            problems.append("parity_set.sha256 is required unless the set is synthetic")
    elif (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(ch not in _HEX for ch in digest)
    ):
        problems.append("parity_set.sha256 must be 64 lowercase hex characters or null")


def _outputs(value: object, result: object, problems: list[str]) -> None:
    if not isinstance(value, list):
        problems.append("outputs must be a list")
        return
    if result == "pass" and not value:
        problems.append("a passing report needs at least one output")
    for index, item in enumerate(value):
        label = f"outputs[{index}]"
        if not _object_keys(item, _OUTPUT, label, problems):
            continue
        assert isinstance(item, dict)
        if not isinstance(item.get("name"), str) or not item.get("name"):
            problems.append(f"{label}.name must be a string")
        shape = item.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dim) is not int or dim < 1 for dim in shape)
        ):
            problems.append(f"{label}.shape must be positive integers")
            shape = []
        for key in ("mean_abs", "max_abs", "rmse"):
            if not _finite(item.get(key)) or float(item.get(key)) < 0:
                problems.append(f"{label}.{key} must be >= 0")
        cosine = item.get("cosine")
        if not _finite(cosine) or not -1.0 <= float(cosine) <= 1.0:
            problems.append(f"{label}.cosine must be between -1 and 1")
        _metric_list(item.get("per_joint"), shape, f"{label}.per_joint", problems)
        _optional_limit(
            item.get("action_chunk"), f"{label}.action_chunk", 0.0, None, problems
        )
        _optional_limit(item.get("top1"), f"{label}.top1", 0.0, 1.0, problems)
        _optional_limit(item.get("kl"), f"{label}.kl", 0.0, None, problems)


def _unchecked(value: object, problems: list[str]) -> None:
    if not isinstance(value, list):
        problems.append("unchecked must be a list")
        return
    for index, item in enumerate(value):
        label = f"unchecked[{index}]"
        if not _object_keys(item, _UNCHECKED, label, problems):
            continue
        assert isinstance(item, dict)
        for key in _UNCHECKED:
            if not isinstance(item.get(key), str) or not item.get(key):
                problems.append(f"{label}.{key} must be a string")


def _metric_list(
    value: object, shape: list[int], label: str, problems: list[str]
) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        problems.append(f"{label} must be a list or null")
        return
    if shape and len(value) != shape[-1]:
        problems.append(f"{label} must match the last dimension")
    if any(not _finite(item) or float(item) < 0 for item in value):
        problems.append(f"{label} values must be >= 0")


def _optional_limit(
    value: object, label: str, low: float, high: float | None, problems: list[str]
) -> None:
    if value is None:
        return
    if not _finite(value):
        problems.append(f"{label} must be a number or null")
        return
    number = float(value)
    if number < low or (high is not None and number > high):
        problems.append(
            f"{label} is outside {low}..{high if high is not None else 'inf'}"
        )


def _optional_text(value: object, label: str, problems: list[str]) -> None:
    if value is not None and not isinstance(value, str):
        problems.append(f"{label} must be a string or null")


def _strings(
    value: object, keys: tuple[str, ...], label: str, problems: list[str]
) -> None:
    if not isinstance(value, dict):
        return
    for key in keys:
        if key in value and (not isinstance(value[key], str) or not value[key]):
            problems.append(f"{label}.{key} must be a string")


def _object_keys(
    value: object, keys: tuple[str, ...], label: str, problems: list[str]
) -> bool:
    if not isinstance(value, dict):
        problems.append(f"{label} must be an object")
        return False
    missing = [key for key in keys if key not in value]
    extra = [key for key in value if key not in keys]
    if missing:
        problems.append(f"{label} missing " + ", ".join(missing))
    if extra:
        problems.append(f"{label} unknown " + ", ".join(sorted(extra)))
    return not missing and not extra


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and value not in {float("inf"), float("-inf")}
