"""``ohho targets`` command implementation."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ohho.targets.detect import detect_target
from ohho.targets.errors import TargetProfileError
from ohho.targets.schema import (
    MODEL_CLASSES,
    catalog_document,
    check_combination,
    get_profile,
    list_profiles,
    profile_to_dict,
)


def register_targets(subparsers: Any) -> None:
    """Add the ``targets`` subcommand to the root parser."""
    parser = subparsers.add_parser(
        "targets",
        help="optimisation targets (precision, providers, memory, SLOs)",
    )
    commands = parser.add_subparsers(dest="action")

    listing = commands.add_parser("list", help="list built-in target profiles")
    listing.add_argument(
        "--json", action="store_true", help="print the catalog as JSON"
    )
    listing.set_defaults(action="list")

    show = commands.add_parser("show", help="print one target profile")
    show.add_argument("profile_id", help="target profile id")
    show.add_argument("--json", action="store_true", help="print the profile as JSON")
    show.set_defaults(action="show")

    detect = commands.add_parser(
        "detect",
        help="probe this machine read-only and name a target",
    )
    detect.add_argument(
        "--json",
        action="store_true",
        help="print the detection report as JSON",
    )
    detect.set_defaults(action="detect")

    check = commands.add_parser(
        "check",
        help="accept or reject a precision and execution provider",
    )
    check.add_argument("--profile", required=True, help="target profile id")
    check.add_argument(
        "--precision",
        required=True,
        help="FP32, FP16, INT8, FP8 or INT4",
    )
    check.add_argument(
        "--provider",
        required=True,
        help="tensorrt, cuda, cpu, or an ONNX Runtime provider name",
    )
    check.add_argument(
        "--tensorrt-version",
        default="",
        help="dotted TensorRT version to check against the profile",
    )
    check.set_defaults(action="check")
    parser.set_defaults(func=run_targets)


def run_targets(args: argparse.Namespace) -> int:
    """Run ``list``, ``show``, ``detect`` or ``check``. Unknown actions return 1."""
    action = getattr(args, "action", None)
    try:
        if action == "list":
            return _list(bool(getattr(args, "json", False)))
        if action == "show":
            return _show(args.profile_id, bool(getattr(args, "json", False)))
        if action == "detect":
            return _detect(bool(getattr(args, "json", False)))
        if action == "check":
            return _check(args)
    except TargetProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("usage: ohho targets [list|show|detect|check]", file=sys.stderr)
    return 1


def _dump(payload: object) -> None:
    print(json.dumps(payload, indent=2, allow_nan=False))


def _list(as_json: bool) -> int:
    if as_json:
        _dump(catalog_document())
        return 0
    for profile_id in list_profiles():
        profile = get_profile(profile_id)
        precisions = ", ".join(profile.precisions)
        print(
            f"{profile.id:<18} {profile.arch:<8} {profile.accelerator:<10} {precisions}"
        )
    return 0


def _show(profile_id: str, as_json: bool) -> int:
    profile = get_profile(profile_id)
    if as_json:
        _dump(profile_to_dict(profile))
        return 0
    print(f"profile: {profile.id}")
    print(f"  {profile.summary}")
    print(f"  arch: {profile.arch}")
    print(f"  accelerator: {profile.accelerator}")
    print(f"  precisions: {', '.join(profile.precisions)}")
    print(f"  execution providers: {', '.join(profile.execution_providers)}")
    if profile.tensorrt is None:
        print("  tensorrt: none")
    else:
        print(f"  tensorrt: {profile.tensorrt.describe()}")
    print(f"  memory budget: {profile.memory_budget_mib} MiB")
    print(f"  workspace: {profile.workspace_size_mib} MiB")
    print("  default p95 ms:")
    for name in MODEL_CLASSES:
        print(f"    {name}: {profile.slos_p95_ms[name]:g}")
    return 0


def _detect(as_json: bool) -> int:
    found = detect_target()
    if as_json:
        _dump(found.to_dict())
    else:
        print(found.text())
    return 0


def _check(args: argparse.Namespace) -> int:
    precision, provider = check_combination(
        args.profile,
        args.precision,
        args.provider,
        tensorrt_version=getattr(args, "tensorrt_version", ""),
    )
    print(f"ok: {precision} on {args.profile} with {provider}")
    return 0
