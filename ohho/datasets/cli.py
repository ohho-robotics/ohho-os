"""``ohho dataset build|pin|stats|import-hf``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .build import build_version
from .hf_import import import_hf_dataset
from .model import (
    Augmentation,
    AxisRange,
    DatasetSpec,
    DatasetStats,
    DatasetVersionError,
    Filters,
    SourceQuery,
    Split,
)
from .store import pin_version, resolve_dataset_version, version_stats


def add_dataset_parser(sub: Any) -> None:
    """Register the ``dataset`` command on the root ``ohho`` parser."""
    dataset = sub.add_parser(
        "dataset",
        help="version local episodes: build, pin, stats, import-hf",
    )
    commands = dataset.add_subparsers(dest="dataset_command", required=True)

    build = commands.add_parser(
        "build",
        help="materialise a LeRobot dataset version from local episodes",
    )
    build.add_argument(
        "--source",
        action="append",
        required=True,
        dest="sources",
        metavar="PATH",
        help="LeRobot episode root, or a directory of them (repeatable)",
    )
    build.add_argument(
        "--name", default="dataset", help="dataset label (default: dataset)"
    )
    build.add_argument(
        "--task", default=None, help="keep episodes whose task contains this text"
    )
    build.add_argument(
        "--robot-type", default=None, help="keep episodes from this robot type"
    )
    build.add_argument(
        "--min-frames", type=int, default=None, help="drop shorter episodes"
    )
    build.add_argument(
        "--max-frames", type=int, default=None, help="drop longer episodes"
    )
    build.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="all keeps every episode; other names use the fractions (default: all)",
    )
    build.add_argument("--train-fraction", type=float, default=1.0)
    build.add_argument("--val-fraction", type=float, default=0.0)
    build.add_argument("--test-fraction", type=float, default=0.0)
    build.add_argument(
        "--seed", type=int, default=0, help="split shuffle and noise seed"
    )
    build.add_argument("--action-scale", type=float, default=1.0)
    build.add_argument("--state-scale", type=float, default=1.0)
    build.add_argument("--action-offset", type=float, default=0.0)
    build.add_argument("--state-offset", type=float, default=0.0)
    build.add_argument("--action-noise", type=float, default=0.0)
    build.add_argument("--state-noise", type=float, default=0.0)
    build.add_argument(
        "--copies", type=int, default=1, help="repeat the selected episodes"
    )
    build.add_argument(
        "--licence", default="unspecified", help="licence string stored on the version"
    )
    _add_common(build)
    build.set_defaults(func=dispatch)

    pin = commands.add_parser("pin", help="freeze a dataset version")
    pin.add_argument("version_id", help="dataset version id (dv_...)")
    _add_common(pin)
    pin.set_defaults(func=dispatch)

    stats = commands.add_parser(
        "stats",
        help="print episodes, frames, and action/state ranges stored on a version",
    )
    stats.add_argument("version_id", help="dataset version id (dv_...)")
    _add_common(stats)
    stats.set_defaults(func=dispatch)

    import_hf = commands.add_parser(
        "import-hf",
        help="import a LeRobot dataset from Hugging Face Hub as a pinned version",
    )
    import_hf.add_argument("repo_id", help="Hugging Face dataset repo id (owner/name)")
    import_hf.add_argument(
        "--revision",
        default=None,
        help="branch, tag, or commit to resolve and pin (default: the Hub default)",
    )
    _add_common(import_hf)
    import_hf.set_defaults(func=dispatch)


def dispatch(args: argparse.Namespace) -> int:
    """Run the selected dataset subcommand. User errors exit 1."""
    try:
        command = args.dataset_command
        if command == "build":
            return _build(args)
        if command == "pin":
            return _pin(args)
        if command == "stats":
            return _stats(args)
        if command == "import-hf":
            return _import_hf(args)
    except DatasetVersionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("error: unknown dataset command", file=sys.stderr)
    return 2


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store",
        default=None,
        help="version store (default: OHHO_DATASETS_ROOT or ~/.ohho/datasets)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON on stdout")


def _build(args: argparse.Namespace) -> int:
    spec = DatasetSpec(
        source_query=SourceQuery(tuple(args.sources)),
        dataset_name=args.name,
        filters=Filters(
            task_contains=args.task,
            robot_type=args.robot_type,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
        ),
        split=Split(
            name=args.split,
            train=args.train_fraction,
            val=args.val_fraction,
            test=args.test_fraction,
        ),
        seed=args.seed,
        augmentation=Augmentation(
            action_scale=args.action_scale,
            state_scale=args.state_scale,
            action_offset=args.action_offset,
            state_offset=args.state_offset,
            action_noise=args.action_noise,
            state_noise=args.state_noise,
            copies=args.copies,
        ),
        licence=args.licence,
    )
    record = build_version(spec, store=args.store)
    dataset = resolve_dataset_version(record.version_id, store=args.store)
    if args.json:
        _emit(
            {
                "version_id": record.version_id,
                "content_hash": record.content_hash,
                "pinned": record.pinned,
                "dataset": str(dataset),
                "stats": record.stats.to_dict(),
            }
        )
        return 0
    print(f"version_id: {record.version_id}")
    print(f"content_hash: {record.content_hash}")
    print(f"pinned: {str(record.pinned).lower()}")
    print(f"dataset: {dataset}")
    _print_ranges(record.stats)
    return 0


def _pin(args: argparse.Namespace) -> int:
    record = pin_version(args.version_id, store=args.store)
    if args.json:
        _emit(
            {
                "version_id": record.version_id,
                "content_hash": record.content_hash,
                "pinned": record.pinned,
            }
        )
        return 0
    print(f"pinned: {record.version_id}")
    print(f"content_hash: {record.content_hash}")
    return 0


def _stats(args: argparse.Namespace) -> int:
    record_stats = version_stats(args.version_id, store=args.store)
    if args.json:
        payload = record_stats.to_dict()
        payload["version_id"] = args.version_id
        _emit(payload)
        return 0
    print(f"version_id: {args.version_id}")
    _print_ranges(record_stats)
    return 0


def _import_hf(args: argparse.Namespace) -> int:
    record = import_hf_dataset(args.repo_id, revision=args.revision, store=args.store)
    dataset = resolve_dataset_version(record.version_id, store=args.store)
    if record.warning:
        print(f"warning: {record.warning}", file=sys.stderr)
    if args.json:
        _emit(
            {
                "version_id": record.version_id,
                "repo_id": record.repo_id,
                "revision": record.revision,
                "requested_revision": record.requested_revision,
                "licence": record.licence,
                "non_commercial": record.non_commercial,
                "revision_pinned": True,
                "licence_pinned": True,
                "pinned": True,
                "dataset": str(dataset),
                "warning": record.warning,
            }
        )
        return 0
    print(f"version_id: {record.version_id}")
    print(f"repo_id: {record.repo_id}")
    print(f"revision: {record.revision}")
    print(f"licence: {record.licence}")
    print(f"non_commercial: {str(record.non_commercial).lower()}")
    print(f"dataset: {dataset}")
    return 0


def _print_ranges(stats: DatasetStats) -> None:
    print(f"episodes: {stats.episodes}")
    print(f"frames: {stats.frames}")
    _print_axis("action", stats.action_ranges)
    _print_axis("state", stats.state_ranges)


def _print_axis(name: str, axes: tuple[AxisRange, ...]) -> None:
    for index, axis in enumerate(axes):
        print(f"{name}[{index}]: min={axis.minimum} max={axis.maximum}")


def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
