"""``ohho bench`` command: run a benchmark, or validate a result file.

``ohho bench validate FILE`` is the schema check from OHH-473. An artifact
run is ``ohho bench --artifact PATH --target PROFILE``. A camera-to-action
run is ``ohho bench --pipeline``. ``ohho bench`` with none of those is
rejected by the root parser so the historical exit code 2 stays.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from ohho.bench.environment import GpuProbe
from ohho.bench.errors import BenchHarnessError
from ohho.bench.harness import BenchConfig, execute
from ohho.bench.pipeline import PipelineConfig, execute_pipeline
from ohho.bench.sensors.base import SensorSource
from ohho.bench.report import (
    dump_result,
    log_wandb,
    push_prometheus,
    render_markdown,
    write_text,
)
from ohho.bench.schema import BenchValidationError, validate_path


def register_bench(subparsers: Any) -> None:
    """Add ``bench`` to the root CLI parser."""
    bench = subparsers.add_parser(
        "bench",
        help="benchmark an inference artifact or validate a result",
        description=(
            "Run an inference benchmark, a camera-to-action pipeline benchmark, "
            "or validate an ohho.bench/v1 result. A bare `ohho bench` is not a run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ohho bench --artifact model.onnx --target x86-cpu "
            "--warmup 10 --iters 50 --batch 1,4,8 --inputs random --json out.json\n"
            "  ohho bench --artifact model.onnx --target x86-cpu --baseline prev.json "
            "--slo p95<=200ms --max-regression p95=10%% --markdown summary.md\n"
            "  ohho bench --artifact model.onnx --target x86-cpu --sustained 30 "
            "--json out.json\n"
            "  ohho bench --pipeline --hz 30 --json out.json\n"
            "  ohho bench --pipeline --fixture stream.json --sink inprocess\n"
            "  ohho bench validate out.json\n"
        ),
    )
    bench.add_argument("--artifact", help="path to an ONNX artifact")
    bench.add_argument(
        "--target",
        help="hardware target profile id, for example x86-cpu",
    )
    bench.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="discarded iterations (artifact default 10, pipeline default 1)",
    )
    bench.add_argument(
        "--iters",
        type=int,
        default=None,
        help="measured iterations (artifact default 50, pipeline default 30)",
    )
    bench.add_argument(
        "--batch",
        default="1",
        help="comma-separated batch sizes, for example 1,4,8",
    )
    bench.add_argument(
        "--inputs",
        default="random",
        help="random synthetic inputs; dataset inputs are not yet supported",
    )
    bench.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="write the ohho.bench/v1 result to this path",
    )
    bench.add_argument(
        "--markdown",
        default=None,
        help="write a GitHub step-summary Markdown report",
    )
    bench.add_argument(
        "--baseline",
        default=None,
        help="previous ohho.bench/v1 result to compare",
    )
    bench.add_argument(
        "--slo",
        action="append",
        default=None,
        help="latency gate such as p95<=200ms (repeatable)",
    )
    bench.add_argument(
        "--max-regression",
        dest="max_regression",
        action="append",
        default=None,
        help="regression gate such as p95=10%% (repeatable)",
    )
    bench.add_argument(
        "--parity",
        default=None,
        help="JSON file embedded on the result as parity",
    )
    bench.add_argument(
        "--prometheus-push",
        default=None,
        help="Pushgateway URL. Off unless set, and not required to run.",
    )
    bench.add_argument(
        "--wandb",
        action="store_true",
        help="log the result to W&B. Off unless set, and not required to run.",
    )
    bench.add_argument(
        "--workload",
        default=None,
        help="workload class stored in run.target (artifact default inference)",
    )
    bench.add_argument(
        "--pipeline",
        action="store_true",
        help="replay a camera stream through decode, preprocess, h2d, "
        "inference, d2h, postprocess, and publish",
    )
    bench.add_argument(
        "--source",
        choices=("sim", "lerobot", "rosbag", "fixture", "hardware"),
        default=None,
        help="pipeline input (default sim, or the path flag that is set)",
    )
    bench.add_argument(
        "--dataset",
        default=None,
        help="LeRobot dataset directory for --source lerobot",
    )
    bench.add_argument(
        "--bag",
        default=None,
        help="ohho.pipeline.rosbag/v1 file or directory for --source rosbag",
    )
    bench.add_argument(
        "--fixture",
        default=None,
        help="ohho.pipeline.stream/v1 JSON file for --source fixture",
    )
    bench.add_argument(
        "--episode",
        type=int,
        default=None,
        help="LeRobot episode index (default 0)",
    )
    bench.add_argument(
        "--hz",
        type=float,
        default=None,
        help="pipeline control rate in Hz (default 30)",
    )
    bench.add_argument(
        "--frames",
        type=int,
        default=None,
        help="cap pipeline frames (default warmup + iters)",
    )
    bench.add_argument(
        "--policy",
        default=None,
        help="pipeline policy (builtin tiny-cpu)",
    )
    bench.add_argument(
        "--sink",
        choices=("inprocess", "sim", "ros2"),
        default=None,
        help="where publish sends the action (sim source defaults to sim)",
    )
    bench.add_argument(
        "--robot",
        default=None,
        help="sim robot id used for the camera and the publish clamp (default sim)",
    )
    bench.add_argument(
        "--allow-hardware",
        action="store_true",
        help="replay a recording labelled hardware; never opens a robot",
    )
    bench.add_argument(
        "--sustained",
        type=float,
        default=None,
        help=(
            "measure for this many seconds instead of --iters, and report "
            "p95 latency drift per time window"
        ),
    )
    commands = bench.add_subparsers(dest="bench_command", required=False)
    validate = commands.add_parser(
        "validate",
        help="validate a result file against ohho.bench/v1",
    )
    validate.add_argument("file", help="path to a JSON result")
    validate.set_defaults(func=run_validate)
    bench.set_defaults(func=run_bench_command, bench_command=None)


def run_validate(args: argparse.Namespace) -> int:
    """``ohho bench validate FILE``. Exit 1 when the file does not match."""
    try:
        validate_path(args.file)
    except BenchValidationError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"valid ohho.bench/v1: {args.file}")
    return 0


def run_bench_command(
    args: argparse.Namespace,
    *,
    runner: Any = None,
    clock: Any = None,
    sources: Sequence[SensorSource] | None = None,
    gpu: GpuProbe | None = None,
) -> int:
    """Run the harness from parsed CLI arguments.

    Tests pass ``runner`` and ``clock`` to avoid a real session. The argparse
    path leaves both unset and loads the ONNX artifact.
    """
    try:
        if getattr(args, "pipeline", False):
            document = execute_pipeline(
                _pipeline_config(args),
                clock=clock,
                baseline_path=getattr(args, "baseline", None),
                slos=tuple(args.slo or ()),
                max_regressions=tuple(args.max_regression or ()),
                parity_path=getattr(args, "parity", None),
            )
        else:
            _reject_pipeline_flags(args)
            config = _config(args)
            document = execute(
                config,
                runner=runner,
                clock=clock,
                sources=sources,
                gpu=gpu,
                baseline_path=getattr(args, "baseline", None),
                slos=tuple(args.slo or ()),
                max_regressions=tuple(args.max_regression or ()),
                parity_path=getattr(args, "parity", None),
            )
        _emit(document, args)
    except BenchHarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    gates = document.get("gates")
    failed = isinstance(gates, dict) and not gates.get("passed", True)
    if failed:
        for check in gates.get("checks", []):
            if not check.get("passed", True):
                print(
                    f"gate failed: {check.get('expr', '')}: {check.get('detail', '')}",
                    file=sys.stderr,
                )
    try:
        _exporters(args, document)
    except BenchHarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1 if failed else 0


def _config(args: argparse.Namespace) -> BenchConfig:
    artifact = getattr(args, "artifact", None)
    target = getattr(args, "target", None)
    if not artifact:
        raise BenchHarnessError("--artifact is required")
    if not target:
        raise BenchHarnessError("--target is required")
    warmup = args.warmup if args.warmup is not None else 10
    iters = args.iters if args.iters is not None else 50
    workload = args.workload if args.workload is not None else "inference"
    return BenchConfig(
        artifact=artifact,
        target=target,
        warmup=warmup,
        iters=iters,
        batches=_batches(args.batch),
        inputs=args.inputs,
        workload=workload,
        sustained=getattr(args, "sustained", None),
    )


def _pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    if getattr(args, "artifact", None):
        raise BenchHarnessError(
            "--pipeline measures a camera-to-action stream. "
            "Omit --artifact. The artifact bench remains "
            "`ohho bench --artifact --target`."
        )
    if args.inputs != "random":
        raise BenchHarnessError(
            "--inputs applies to the artifact bench, not --pipeline"
        )
    if args.batch != "1":
        raise BenchHarnessError("--batch applies to the artifact bench, not --pipeline")
    source = _pipeline_source(args)
    workload = args.workload if args.workload is not None else "pipeline"
    return PipelineConfig(
        source=source,
        target=args.target or "x86-cpu",
        hz=30.0 if args.hz is None else args.hz,
        warmup=1 if args.warmup is None else args.warmup,
        iters=30 if args.iters is None else args.iters,
        frames=args.frames,
        episode=0 if args.episode is None else args.episode,
        robot=args.robot or "sim",
        dataset=args.dataset,
        bag=args.bag,
        fixture=args.fixture,
        sink=args.sink or "",
        policy=args.policy or "tiny-cpu",
        allow_hardware=bool(args.allow_hardware),
        workload=workload,
    )


def _pipeline_source(args: argparse.Namespace) -> str:
    source = args.source
    provided = {
        "fixture": args.fixture,
        "lerobot": args.dataset,
        "rosbag": args.bag,
    }
    named = [name for name, value in provided.items() if value]
    if source is None:
        if len(named) > 1:
            raise BenchHarnessError("pass only one of --fixture, --dataset, or --bag")
        if named:
            return named[0]
        return "sim"
    if source == "sim" and named:
        raise BenchHarnessError("--source sim does not take a recording path")
    if source == "hardware":
        return "hardware"
    if source in provided and not provided[source]:
        flag = {"fixture": "--fixture", "lerobot": "--dataset", "rosbag": "--bag"}
        raise BenchHarnessError(f"--source {source} needs {flag[source]}")
    return source


def _reject_pipeline_flags(args: argparse.Namespace) -> None:
    used: list[str] = []
    if args.source:
        used.append("--source")
    if args.dataset:
        used.append("--dataset")
    if args.bag:
        used.append("--bag")
    if args.fixture:
        used.append("--fixture")
    if args.hz is not None:
        used.append("--hz")
    if args.frames is not None:
        used.append("--frames")
    if args.policy:
        used.append("--policy")
    if args.sink:
        used.append("--sink")
    if args.allow_hardware:
        used.append("--allow-hardware")
    if args.episode is not None:
        used.append("--episode")
    if args.robot:
        used.append("--robot")
    if used:
        raise BenchHarnessError(f"{', '.join(used)} requires --pipeline")


def _batches(text: str) -> tuple[int, ...]:
    if not isinstance(text, str) or not text.strip():
        raise BenchHarnessError("--batch needs at least one positive integer")
    batches: list[int] = []
    for part in text.split(","):
        piece = part.strip()
        if not piece.isdigit():
            raise BenchHarnessError(
                f"--batch values must be positive integers (got {text!r})"
            )
        value = int(piece)
        if value < 1:
            raise BenchHarnessError(
                f"--batch values must be positive integers (got {text!r})"
            )
        batches.append(value)
    return tuple(batches)


def _emit(document: dict[str, Any], args: argparse.Namespace) -> None:
    payload = dump_result(document)
    json_out = getattr(args, "json_out", None)
    if json_out:
        write_text(json_out, payload)
        print(f"wrote {json_out}")
    else:
        sys.stdout.write(payload)
    markdown = getattr(args, "markdown", None)
    if markdown:
        write_text(markdown, render_markdown(document))
        print(f"wrote {markdown}")


def _exporters(args: argparse.Namespace, document: dict[str, Any]) -> None:
    url = getattr(args, "prometheus_push", None)
    if url:
        push_prometheus(url, document)
    if getattr(args, "wandb", False):
        log_wandb(document)
