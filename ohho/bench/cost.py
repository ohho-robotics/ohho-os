"""Estimated cost from a measured bench result and a user-editable price.

Cloud profiles use a configurable USD-per-GPU-hour rate. Edge profiles use
purchase price divided by amortisation hours, plus watts times a configurable
electricity rate. Measured power on the result wins. Otherwise the profile's
typical watts are used and labelled ``profile_default``.

``usd_per_1m_output_tokens`` is ``usd_per_hour * 1_000_000 / (tokens_per_s * 3600)``
from measured output-token throughput.

``usd_per_1k_actions`` is the cost of 1000 actions at ``control_rate_hz`` when
one device is shared across ``actions_per_s / control_rate_hz`` robots. The
control rate cancels, so the figure is ``usd_per_hour * 1000 / (actions_per_s * 3600)``
from measured action throughput. ``meets_control_rate`` and
``robots_per_device`` keep the control rate. ``usd_per_robot_hour`` is
``usd_per_hour * control_rate_hz / actions_per_s``.

``actions_per_watt`` is measured actions per second divided by watts.

Every figure is an estimate. ``estimate`` is true and ``label`` is
``estimate``. :func:`recompute_cost` rewrites those fields from the stored
throughput and does not run the benchmark again.

:func:`rank_by_cost` orders results by one of those estimates, optionally
under a latency p95 ceiling. The console view that renders that order is a
separate change.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from ohho.bench.errors import BenchHarnessError

COST_SCHEMA = "ohho.bench.cost/v1"
PRICE_SCHEMA = "ohho.bench.cost_prices/v1"
DEFAULT_PROFILE = "x86-cpu"
_SECONDS_PER_HOUR = 3600.0
_TOKEN_UNITS = frozenset({"tokens/s", "token/s"})
_ACTION_UNITS = frozenset(
    {"hz", "steps/s", "step/s", "actions/s", "frames/s", "frame/s"}
)
_GENERATE_WORKLOADS = frozenset({"llm", "vlm", "vla"})
_PROFILE_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_PRICE_KEYS = frozenset(
    {
        "schema",
        "estimate",
        "label",
        "note",
        "electricity_usd_per_kwh",
        "amortisation_hours",
        "control_rate_hz",
        "profiles",
    }
)
_PROFILE_KEYS = frozenset({"pricing", "gpu_hour_usd", "hardware_usd", "typical_watts"})
_CLOUD = "cloud_gpu_hour"
_EDGE = "edge_amortised"
_CLOUD_DETAIL = "USD per GPU-hour. Electricity is not added on top of this rate."
_EDGE_DETAIL = "Purchase price divided by amortisation hours, plus watts times the electricity rate."

_BUILTIN: PriceTable | None = None


@dataclass(frozen=True)
class CostOverrides:
    """User edits applied on top of the price table. Unset fields stay put."""

    prices_path: str | None = None
    profile: str | None = None
    gpu_hour_usd: float | None = None
    hardware_usd: float | None = None
    amortisation_hours: float | None = None
    electricity_usd_per_kwh: float | None = None
    watts: float | None = None
    control_rate_hz: float | None = None

    def __post_init__(self) -> None:
        if self.prices_path is not None and not str(self.prices_path).strip():
            object.__setattr__(self, "prices_path", None)
        elif isinstance(self.prices_path, str):
            object.__setattr__(self, "prices_path", self.prices_path.strip())
        if self.profile is not None:
            object.__setattr__(
                self, "profile", _profile_id(self.profile, "target profile")
            )
        _check_range(self.gpu_hour_usd, "--gpu-hour-usd", minimum=0)
        _check_range(self.hardware_usd, "--hardware-usd", minimum=0)
        _check_range(
            self.amortisation_hours, "--amortise-hours", minimum=0, exclusive=True
        )
        _check_range(
            self.electricity_usd_per_kwh,
            "--electricity-usd-per-kwh",
            minimum=0,
        )
        _check_range(self.watts, "--watts", minimum=0, exclusive=True)
        _check_range(
            self.control_rate_hz, "--control-rate-hz", minimum=0, exclusive=True
        )
        if self.gpu_hour_usd is not None and self.hardware_usd is not None:
            raise BenchHarnessError("pass --gpu-hour-usd or --hardware-usd, not both")

    def any_flag(self) -> bool:
        """True when a numeric or profile override is set. ``--prices`` is a file."""
        return any(
            value is not None
            for value in (
                self.profile,
                self.gpu_hour_usd,
                self.hardware_usd,
                self.amortisation_hours,
                self.electricity_usd_per_kwh,
                self.watts,
                self.control_rate_hz,
            )
        )


@dataclass(frozen=True)
class _ProfilePrice:
    pricing: str
    gpu_hour_usd: float | None
    hardware_usd: float | None
    typical_watts: float | None


@dataclass(frozen=True)
class PriceTable:
    """One merged price table. ``origin`` is ``built-in`` or ``file``."""

    note: str
    electricity_usd_per_kwh: float
    amortisation_hours: float
    control_rate_hz: float
    profiles: Mapping[str, _ProfilePrice]
    origin: str


@dataclass(frozen=True)
class _Rates:
    tokens_per_s: float | None
    actions_per_s: float | None
    throughput_value: float
    throughput_unit: str


@dataclass(frozen=True)
class _Resolved:
    target_profile: str
    profile_source: str
    measured_target_profile: str
    pricing: str
    price_source: str
    pricing_detail: str
    usd_per_hour: float
    gpu_hour_usd: float | None
    hardware_usd: float | None
    amortisation_hours: float | None
    hardware_usd_per_hour: float | None
    electricity_usd_per_kwh: float
    electricity_usd_per_hour: float | None
    electricity_applied: bool
    watts: float | None
    watts_source: str
    control_rate_hz: float
    note: str


def require_price(profile_id: str, overrides: CostOverrides | None = None) -> None:
    """Raise when ``profile_id`` has no price and no hourly override.

    An empty id is allowed. The caller then uses the default CPU row.
    """
    if not isinstance(profile_id, str) or not profile_id.strip():
        return
    chosen = _profile_id(profile_id, "target profile")
    edits = overrides or CostOverrides()
    table = load_table(edits)
    if chosen in table.profiles:
        return
    if edits.gpu_hour_usd is not None or edits.hardware_usd is not None:
        return
    raise BenchHarnessError(_missing_profile(chosen, table))


def load_table(overrides: CostOverrides | None = None) -> PriceTable:
    """Built-in table, with ``--prices`` rows and globals merged on top."""
    base = _builtin()
    path = None if overrides is None else overrides.prices_path
    if not path:
        return base
    return _merge(base, _read_price_file(path))


def price_catalog(overrides: CostOverrides | None = None) -> dict[str, Any]:
    """JSON document of the price table. ``estimate`` is true."""
    table = load_table(overrides)
    profiles = {
        profile_id: _profile_dict(table.profiles[profile_id])
        for profile_id in sorted(table.profiles)
    }
    return {
        "schema": PRICE_SCHEMA,
        "estimate": True,
        "label": "estimate",
        "note": table.note,
        "electricity_usd_per_kwh": table.electricity_usd_per_kwh,
        "amortisation_hours": table.amortisation_hours,
        "control_rate_hz": table.control_rate_hz,
        "profiles": profiles,
    }


def attach_cost(
    document: dict[str, Any],
    overrides: CostOverrides | None = None,
    *,
    profile_id: str = "",
) -> dict[str, Any]:
    """Write cost estimates onto ``document`` from its measured throughput.

    Nested batch, point, precision, Pareto, and recommendation rows get the
    same price and their own throughput. The benchmark is not run.
    """
    if not isinstance(document, dict):
        raise BenchHarnessError("bench result must be a JSON object")
    edits = overrides or CostOverrides()
    resolved = _resolve(document, edits, profile_id)
    document["cost"] = _cost_block(resolved, _document_rates(document))
    _attach_batches(document, resolved)
    _attach_llm(document, resolved)
    return document


def recompute_cost(
    document: Mapping[str, Any],
    overrides: CostOverrides | None = None,
) -> dict[str, Any]:
    """Copy ``document`` and replace cost fields. Throughput is unchanged."""
    if not isinstance(document, dict):
        raise BenchHarnessError("bench result must be a JSON object")
    copied = copy.deepcopy(document)
    return attach_cost(copied, overrides)


def rank_by_cost(
    documents: Sequence[Mapping[str, Any]],
    *,
    metric: str = "usd_per_1k_actions",
    max_p95_ms: float | None = None,
) -> list[dict[str, Any]]:
    """Cheapest estimate first, among results at or under a latency p95.

    ``metric`` is ``usd_per_1k_actions``, ``usd_per_1m_output_tokens``, or
    ``usd_per_robot_hour``. A result without that field is skipped. An equal
    p95 meets ``max_p95_ms``. Each row is marked ``estimate``. The console
    comparison view that renders this order is a separate change.
    """
    allowed = {
        "usd_per_1k_actions",
        "usd_per_1m_output_tokens",
        "usd_per_robot_hour",
    }
    if metric not in allowed:
        known = ", ".join(sorted(allowed))
        raise BenchHarnessError(f"cost metric must be one of {known}")
    if max_p95_ms is not None:
        _check_range(max_p95_ms, "max p95", minimum=0)
    rows: list[dict[str, Any]] = []
    for document in documents:
        if not isinstance(document, Mapping):
            continue
        cost = document.get("cost")
        if not isinstance(cost, dict) or metric not in cost:
            continue
        if not _is_number(cost[metric]):
            continue
        latency = document.get("latency")
        p95 = latency.get("p95") if isinstance(latency, Mapping) else None
        p95_value = float(p95) if _is_number(p95) else None
        if max_p95_ms is not None and (
            p95_value is None or p95_value > float(max_p95_ms) + 1e-9
        ):
            continue
        inputs = cost.get("inputs") if isinstance(cost.get("inputs"), Mapping) else {}
        run = document.get("run") if isinstance(document.get("run"), Mapping) else {}
        rows.append(
            {
                "estimate": True,
                "label": "estimate",
                "metric": metric,
                "usd": float(cost[metric]),
                "target_profile": str(inputs.get("target_profile") or ""),
                "run_id": str(run.get("id") or ""),
                "latency_p95_ms": p95_value,
                "inputs": copy.deepcopy(dict(inputs)),
            }
        )
    rows.sort(key=lambda row: (row["usd"], row["target_profile"], row["run_id"]))
    return rows


def format_estimate(cost: Mapping[str, Any]) -> str:
    """One line starting with ``estimate:`` and the inputs that were applied."""
    if not isinstance(cost, Mapping):
        raise BenchHarnessError("cost block must be an object")
    inputs = cost.get("inputs") if isinstance(cost.get("inputs"), Mapping) else {}
    bits: list[str] = []
    if "usd_per_1m_output_tokens" in cost:
        bits.append(f"${_fmt(cost['usd_per_1m_output_tokens'])} per 1M output tokens")
    if "usd_per_1k_actions" in cost:
        bits.append(f"${_fmt(cost['usd_per_1k_actions'])} per 1k actions")
    if "actions_per_watt" in cost:
        bits.append(f"{_fmt(cost['actions_per_watt'])} actions/W")
    if "usd_per_hour" in cost:
        bits.append(f"${_fmt(cost['usd_per_hour'])} per hour")
    shown: list[str] = []
    for key in (
        "target_profile",
        "profile_source",
        "pricing",
        "price_source",
        "gpu_hour_usd",
        "hardware_usd",
        "amortisation_hours",
        "electricity_usd_per_kwh",
        "electricity_applied",
        "watts",
        "watts_source",
        "control_rate_hz",
        "tokens_per_s",
        "actions_per_s",
    ):
        if key in inputs:
            shown.append(f"{key}={inputs[key]}")
    body = ", ".join(bits) if bits else "no derived rate"
    return f"estimate: {body} ({', '.join(shown)})"


def _builtin() -> PriceTable:
    global _BUILTIN
    if _BUILTIN is None:
        _BUILTIN = _parse_table(
            _read_builtin(), origin="built-in", path="cost_prices.json"
        )
    return _BUILTIN


def _read_builtin() -> dict[str, Any]:
    path = files("ohho.targets").joinpath("cost_prices.json")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchHarnessError(
            "built-in price table is not in package data (ohho/targets/cost_prices.json)"
        ) from exc
    return _loads(text, "built-in price table")


def _read_price_file(path: str) -> dict[str, Any]:
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BenchHarnessError(f"price file not found: {path}") from exc
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise BenchHarnessError(f"cannot read price file: {detail}") from exc
    return _loads(text, f"price file {path}")


def _loads(text: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BenchHarnessError(
            f"{label} is not JSON: {exc.msg} (line {exc.lineno})"
        ) from exc
    if not isinstance(parsed, dict):
        raise BenchHarnessError(f"{label} must be a JSON object")
    return parsed


def _parse_table(document: Mapping[str, Any], *, origin: str, path: str) -> PriceTable:
    _check_price_header(document, path)
    note = document.get("note")
    if not isinstance(note, str) or not note.strip():
        raise BenchHarnessError(f"{path}: note must be a non-empty string")
    electricity = _required_number(document, "electricity_usd_per_kwh", path, minimum=0)
    amortisation = _required_number(
        document, "amortisation_hours", path, minimum=0, exclusive=True
    )
    control = _required_number(
        document, "control_rate_hz", path, minimum=0, exclusive=True
    )
    profiles = _parse_profiles(document.get("profiles"), path, base=None)
    if not profiles:
        raise BenchHarnessError(f"{path}: profiles must list at least one target")
    return PriceTable(
        note=note.strip(),
        electricity_usd_per_kwh=electricity,
        amortisation_hours=amortisation,
        control_rate_hz=control,
        profiles=profiles,
        origin=origin,
    )


def _check_price_header(document: Mapping[str, Any], path: str) -> None:
    extra = set(document) - _PRICE_KEYS
    if extra:
        names = ", ".join(sorted(extra))
        raise BenchHarnessError(f"{path}: unknown price field {names}")
    if document.get("schema") != PRICE_SCHEMA:
        raise BenchHarnessError(f"{path}: schema must be {PRICE_SCHEMA}")
    if document.get("estimate") is not True:
        raise BenchHarnessError(f"{path}: estimate must be true")
    if "label" in document and document.get("label") != "estimate":
        raise BenchHarnessError(f"{path}: label must be estimate")


def _merge(base: PriceTable, user: Mapping[str, Any]) -> PriceTable:
    path = "price file"
    _check_price_header(user, path)
    note = user.get("note", base.note)
    if not isinstance(note, str) or not note.strip():
        raise BenchHarnessError(f"{path}: note must be a non-empty string")
    electricity = _optional_number(
        user,
        "electricity_usd_per_kwh",
        base.electricity_usd_per_kwh,
        path,
        minimum=0,
    )
    amortisation = _optional_number(
        user,
        "amortisation_hours",
        base.amortisation_hours,
        path,
        minimum=0,
        exclusive=True,
    )
    control = _optional_number(
        user,
        "control_rate_hz",
        base.control_rate_hz,
        path,
        minimum=0,
        exclusive=True,
    )
    if "profiles" in user:
        profiles = _parse_profiles(user.get("profiles"), path, base=base.profiles)
    else:
        profiles = dict(base.profiles)
    return PriceTable(
        note=note.strip(),
        electricity_usd_per_kwh=float(electricity),
        amortisation_hours=float(amortisation),
        control_rate_hz=float(control),
        profiles=profiles,
        origin="file",
    )


def _parse_profiles(
    raw: object,
    path: str,
    *,
    base: Mapping[str, _ProfilePrice] | None,
) -> dict[str, _ProfilePrice]:
    if not isinstance(raw, dict) or not raw:
        raise BenchHarnessError(f"{path}: profiles must be a non-empty object")
    merged: dict[str, _ProfilePrice] = {} if base is None else dict(base)
    for key, value in raw.items():
        profile_id = _profile_id(key, f"{path} profile")
        if not isinstance(value, dict):
            raise BenchHarnessError(f"{path}: profile {profile_id} must be an object")
        parent = None if base is None else base.get(profile_id)
        merged[profile_id] = _parse_profile(
            value, f"{path} profile {profile_id}", parent
        )
    return merged


def _parse_profile(
    raw: Mapping[str, Any],
    path: str,
    parent: _ProfilePrice | None,
) -> _ProfilePrice:
    extra = set(raw) - _PROFILE_KEYS
    if extra:
        names = ", ".join(sorted(extra))
        raise BenchHarnessError(f"{path}: unknown field {names}")
    if "pricing" in raw:
        pricing = raw["pricing"]
    elif parent is not None:
        pricing = parent.pricing
    else:
        raise BenchHarnessError(f"{path}: pricing is required")
    if pricing not in {_CLOUD, _EDGE}:
        raise BenchHarnessError(f"{path}: pricing must be {_CLOUD} or {_EDGE}")
    gpu = _optional_number(
        raw,
        "gpu_hour_usd",
        None if parent is None else parent.gpu_hour_usd,
        path,
        minimum=0,
    )
    hardware = _optional_number(
        raw,
        "hardware_usd",
        None if parent is None else parent.hardware_usd,
        path,
        minimum=0,
    )
    watts = _optional_number(
        raw,
        "typical_watts",
        None if parent is None else parent.typical_watts,
        path,
        minimum=0,
        exclusive=True,
    )
    if pricing == _CLOUD and gpu is None:
        raise BenchHarnessError(f"{path}: {_CLOUD} needs gpu_hour_usd")
    if pricing == _EDGE and hardware is None:
        raise BenchHarnessError(f"{path}: {_EDGE} needs hardware_usd")
    return _ProfilePrice(
        pricing=str(pricing),
        gpu_hour_usd=None if gpu is None else float(gpu),
        hardware_usd=None if hardware is None else float(hardware),
        typical_watts=None if watts is None else float(watts),
    )


def _profile_dict(row: _ProfilePrice) -> dict[str, Any]:
    payload: dict[str, Any] = {"pricing": row.pricing}
    if row.gpu_hour_usd is not None:
        payload["gpu_hour_usd"] = row.gpu_hour_usd
    if row.hardware_usd is not None:
        payload["hardware_usd"] = row.hardware_usd
    if row.typical_watts is not None:
        payload["typical_watts"] = row.typical_watts
    return payload


def _resolve(document: Mapping[str, Any], edits: CostOverrides, hint: str) -> _Resolved:
    table = load_table(edits)
    profile_id, profile_source = _choose_profile(document, edits, hint)
    measured = _measured_profile(document)
    row = _choose_row(table, profile_id, edits)
    pricing = row.pricing
    gpu = row.gpu_hour_usd
    hardware = row.hardware_usd
    if edits.gpu_hour_usd is not None:
        pricing = _CLOUD
        gpu = edits.gpu_hour_usd
    elif edits.hardware_usd is not None:
        pricing = _EDGE
        hardware = edits.hardware_usd
    amortisation = (
        edits.amortisation_hours
        if edits.amortisation_hours is not None
        else table.amortisation_hours
    )
    electricity = (
        edits.electricity_usd_per_kwh
        if edits.electricity_usd_per_kwh is not None
        else table.electricity_usd_per_kwh
    )
    control = (
        edits.control_rate_hz
        if edits.control_rate_hz is not None
        else table.control_rate_hz
    )
    watts, watts_source = _watts(document, edits, row.typical_watts)
    hourly, hardware_hour, electricity_hour, applied = _hourly(
        pricing,
        gpu,
        hardware,
        amortisation,
        watts,
        electricity,
    )
    note = table.note
    if profile_source == "default":
        note = (
            f"{note} No target profile was recorded on this result. "
            f"The price row for {DEFAULT_PROFILE} was used. "
            "Pass --target to select a row."
        )
    if edits.any_flag():
        price_source = "flag"
    elif table.origin == "file":
        price_source = "file"
    else:
        price_source = "built-in"
    return _Resolved(
        target_profile=profile_id,
        profile_source=profile_source,
        measured_target_profile=measured,
        pricing=pricing,
        price_source=price_source,
        pricing_detail=_CLOUD_DETAIL if pricing == _CLOUD else _EDGE_DETAIL,
        usd_per_hour=_round(hourly),
        gpu_hour_usd=None if pricing != _CLOUD or gpu is None else float(gpu),
        hardware_usd=None if pricing != _EDGE or hardware is None else float(hardware),
        amortisation_hours=None if pricing != _EDGE else float(amortisation),
        hardware_usd_per_hour=None if hardware_hour is None else _round(hardware_hour),
        electricity_usd_per_kwh=float(electricity),
        electricity_usd_per_hour=None
        if electricity_hour is None
        else _round(electricity_hour),
        electricity_applied=applied,
        watts=None if watts is None else float(watts),
        watts_source=watts_source,
        control_rate_hz=float(control),
        note=note,
    )


def _choose_profile(
    document: Mapping[str, Any], edits: CostOverrides, hint: str
) -> tuple[str, str]:
    if edits.profile:
        return edits.profile, "flag"
    cleaned = hint.strip() if isinstance(hint, str) else ""
    if cleaned:
        return _profile_id(cleaned, "target profile"), "run"
    measured = _measured_profile(document)
    if measured:
        return measured, "run"
    return DEFAULT_PROFILE, "default"


def _choose_row(
    table: PriceTable, profile_id: str, edits: CostOverrides
) -> _ProfilePrice:
    row = table.profiles.get(profile_id)
    if row is not None:
        return row
    if edits.gpu_hour_usd is not None:
        return _ProfilePrice(_CLOUD, edits.gpu_hour_usd, None, None)
    if edits.hardware_usd is not None:
        return _ProfilePrice(_EDGE, None, edits.hardware_usd, None)
    raise BenchHarnessError(_missing_profile(profile_id, table))


def _missing_profile(profile_id: str, table: PriceTable) -> str:
    known = ", ".join(sorted(table.profiles))
    return (
        f"no price for target profile {profile_id!r}. "
        "Pass --gpu-hour-usd or --hardware-usd, or add the profile to --prices. "
        f"Known profiles: {known}."
    )


def _measured_profile(document: Mapping[str, Any]) -> str:
    environment = document.get("environment")
    if not isinstance(environment, Mapping):
        return ""
    value = environment.get("target_profile")
    if not isinstance(value, str):
        return ""
    return value.strip()


def _watts(
    document: Mapping[str, Any],
    edits: CostOverrides,
    typical: float | None,
) -> tuple[float | None, str]:
    if edits.watts is not None:
        return float(edits.watts), "user"
    telemetry = document.get("telemetry")
    if isinstance(telemetry, Mapping):
        for key in ("power_w_mean", "gpu_power_w"):
            value = telemetry.get(key)
            if _is_number(value) and float(value) > 0:
                return float(value), "measured"
    if typical is not None and typical > 0:
        return float(typical), "profile_default"
    return None, "unmeasured"


def _hourly(
    pricing: str,
    gpu: float | None,
    hardware: float | None,
    amortisation: float,
    watts: float | None,
    electricity: float,
) -> tuple[float, float | None, float | None, bool]:
    if pricing == _CLOUD:
        if gpu is None:
            raise BenchHarnessError(f"{_CLOUD} needs gpu_hour_usd")
        return float(gpu), None, None, False
    if hardware is None:
        raise BenchHarnessError(f"{_EDGE} needs hardware_usd")
    hardware_hour = float(hardware) / float(amortisation)
    if watts is not None and watts > 0:
        electricity_hour = (float(watts) / 1000.0) * float(electricity)
        return hardware_hour + electricity_hour, hardware_hour, electricity_hour, True
    return hardware_hour, hardware_hour, None, False


def _document_rates(document: Mapping[str, Any]) -> _Rates:
    throughput = document.get("throughput")
    if not isinstance(throughput, dict):
        raise BenchHarnessError("bench result has no throughput")
    return _throughput_rates(throughput, _workload(document), _llm(document))


def _throughput_rates(
    throughput: Mapping[str, Any],
    workload: str,
    llm: Mapping[str, Any] | None,
) -> _Rates:
    if not _is_number(throughput.get("value")) or not isinstance(
        throughput.get("unit"), str
    ):
        raise BenchHarnessError("throughput needs a numeric value and a unit")
    unit = throughput["unit"].strip()
    if not unit:
        raise BenchHarnessError("throughput unit is empty")
    value = float(throughput["value"])
    key = unit.lower()
    tokens = value if key in _TOKEN_UNITS else None
    actions = value if key in _ACTION_UNITS else None
    if llm is not None and _is_number(llm.get("tokens_per_s")):
        tokens = float(llm["tokens_per_s"])
    if workload == "vla" and llm is not None and _is_number(llm.get("requests_per_s")):
        actions = float(llm["requests_per_s"])
    return _Rates(tokens, actions, value, unit)


def _point_rates(point: Mapping[str, Any], workload: str) -> _Rates | None:
    tokens = point.get("tokens_per_s")
    requests = point.get("requests_per_s")
    token_rate = float(tokens) if _is_number(tokens) else None
    request_rate = float(requests) if _is_number(requests) else None
    actions = request_rate if workload == "vla" and request_rate is not None else None
    if token_rate is None and actions is None:
        return None
    if token_rate is not None:
        return _Rates(token_rate, actions, token_rate, "tokens/s")
    assert actions is not None
    return _Rates(None, actions, actions, "actions/s")


def _workload(document: Mapping[str, Any]) -> str:
    llm = _llm(document)
    if llm is not None and llm.get("workload") in _GENERATE_WORKLOADS:
        return str(llm["workload"])
    run = document.get("run")
    if isinstance(run, Mapping) and isinstance(run.get("target"), str):
        target = run["target"].strip()
        if target:
            return target
    return ""


def _llm(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    llm = document.get("llm")
    if isinstance(llm, Mapping):
        return llm
    return None


def _attach_batches(document: dict[str, Any], resolved: _Resolved) -> None:
    batches = document.get("batches")
    if not isinstance(batches, list):
        return
    workload = _workload(document)
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        throughput = batch.get("throughput")
        if not isinstance(throughput, dict):
            batch.pop("cost", None)
            continue
        batch["cost"] = _cost_block(
            resolved, _throughput_rates(throughput, workload, None)
        )


def _attach_llm(document: dict[str, Any], resolved: _Resolved) -> None:
    llm = document.get("llm")
    if not isinstance(llm, dict):
        return
    workload = _workload(document)
    for key in ("points", "precisions", "pareto"):
        rows = llm.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            rates = _point_rates(row, _row_workload(row, workload))
            if rates is None:
                row.pop("cost", None)
            else:
                row["cost"] = _cost_block(resolved, rates)
    recommendation = llm.get("recommendation")
    if isinstance(recommendation, dict):
        _attach_point(recommendation, _row_workload(recommendation, workload), resolved)
    recommendations = llm.get("recommendations")
    if isinstance(recommendations, list):
        for row in recommendations:
            if isinstance(row, dict):
                _attach_point(row, _row_workload(row, workload), resolved)


def _attach_point(row: dict[str, Any], workload: str, resolved: _Resolved) -> None:
    rates = _point_rates(row, workload)
    if rates is None:
        row.pop("cost", None)
    else:
        row["cost"] = _cost_block(resolved, rates)


def _row_workload(row: Mapping[str, Any], fallback: str) -> str:
    name = row.get("workload")
    if name in _GENERATE_WORKLOADS:
        return str(name)
    return fallback


def _cost_block(resolved: _Resolved, rates: _Rates) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "target_profile": resolved.target_profile,
        "profile_source": resolved.profile_source,
        "pricing": resolved.pricing,
        "price_source": resolved.price_source,
        "pricing_detail": resolved.pricing_detail,
        "control_rate_hz": resolved.control_rate_hz,
        "watts_source": resolved.watts_source,
        "electricity_applied": resolved.electricity_applied,
        "electricity_usd_per_kwh": resolved.electricity_usd_per_kwh,
        "throughput_value": rates.throughput_value,
        "throughput_unit": rates.throughput_unit,
        "note": resolved.note,
    }
    if resolved.measured_target_profile:
        inputs["measured_target_profile"] = resolved.measured_target_profile
    if resolved.gpu_hour_usd is not None:
        inputs["gpu_hour_usd"] = resolved.gpu_hour_usd
    if resolved.hardware_usd is not None:
        inputs["hardware_usd"] = resolved.hardware_usd
    if resolved.amortisation_hours is not None:
        inputs["amortisation_hours"] = resolved.amortisation_hours
    if resolved.hardware_usd_per_hour is not None:
        inputs["hardware_usd_per_hour"] = resolved.hardware_usd_per_hour
    if resolved.electricity_usd_per_hour is not None:
        inputs["electricity_usd_per_hour"] = resolved.electricity_usd_per_hour
    if resolved.watts is not None:
        inputs["watts"] = resolved.watts
    if rates.tokens_per_s is not None:
        inputs["tokens_per_s"] = rates.tokens_per_s
    if rates.actions_per_s is not None:
        inputs["actions_per_s"] = rates.actions_per_s
    block: dict[str, Any] = {
        "schema": COST_SCHEMA,
        "estimate": True,
        "label": "estimate",
        "inputs": inputs,
        "usd_per_hour": resolved.usd_per_hour,
    }
    hourly = resolved.usd_per_hour
    if rates.tokens_per_s is not None and rates.tokens_per_s > 0:
        block["usd_per_1m_output_tokens"] = _round(
            hourly * 1_000_000.0 / (rates.tokens_per_s * _SECONDS_PER_HOUR)
        )
    actions = rates.actions_per_s
    if actions is not None and actions > 0:
        block["usd_per_1k_actions"] = _round(
            hourly * 1000.0 / (actions * _SECONDS_PER_HOUR)
        )
        block["usd_per_robot_hour"] = _round(
            hourly * resolved.control_rate_hz / actions
        )
        block["meets_control_rate"] = actions + 1e-9 >= resolved.control_rate_hz
        block["robots_per_device"] = _round(actions / resolved.control_rate_hz)
        if resolved.watts is not None and resolved.watts > 0:
            block["actions_per_watt"] = _round(actions / resolved.watts)
    elif actions == 0:
        block["meets_control_rate"] = False
        block["robots_per_device"] = 0.0
    return block


def _profile_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _PROFILE_ID.fullmatch(value.strip()):
        raise BenchHarnessError(
            f"{label} must be a lowercase id such as x86-cpu (got {value!r})"
        )
    return value.strip()


def _is_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return True


def _check_range(
    value: float | None,
    label: str,
    *,
    minimum: float,
    exclusive: bool = False,
) -> None:
    if value is None:
        return
    if not _is_number(value):
        raise BenchHarnessError(f"{label} must be a finite number")
    number = float(value)
    if exclusive:
        if number <= minimum:
            raise BenchHarnessError(f"{label} must be > {minimum}")
        return
    if number < minimum:
        raise BenchHarnessError(f"{label} must be >= {minimum}")


def _required_number(
    raw: Mapping[str, Any],
    key: str,
    path: str,
    *,
    minimum: float,
    exclusive: bool = False,
) -> float:
    if key not in raw:
        raise BenchHarnessError(f"{path}: {key} is required")
    return float(
        _checked_number(
            raw[key], f"{path}: {key}", minimum=minimum, exclusive=exclusive
        )
    )


def _optional_number(
    raw: Mapping[str, Any],
    key: str,
    fallback: float | None,
    path: str,
    *,
    minimum: float,
    exclusive: bool = False,
) -> float | None:
    if key not in raw:
        return fallback
    return float(
        _checked_number(
            raw[key], f"{path}: {key}", minimum=minimum, exclusive=exclusive
        )
    )


def _checked_number(
    value: object,
    label: str,
    *,
    minimum: float,
    exclusive: bool,
) -> float:
    if not _is_number(value):
        raise BenchHarnessError(f"{label} must be a finite number")
    _check_range(float(value), label, minimum=minimum, exclusive=exclusive)
    return float(value)


def _round(value: float) -> float:
    return round(float(value), 10)


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return ""
