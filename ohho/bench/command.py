"""``ohho bench`` command: run a benchmark, or validate a result file.

``ohho bench validate FILE`` is the schema check from OHH-473. An artifact
run is ``ohho bench --artifact PATH --target PROFILE``. A camera-to-action
run is ``ohho bench --pipeline``. ``ohho bench`` with none of those is
rejected by the root parser so the historical exit code 2 stays.

OHH-475 adds ``version``, ``launch``, and ``llm`` on the same parser. Those
talk to external OpenAI-compatible engines and do not replace the harness.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
import time
from pathlib import Path
from typing import Any

from ohho.bench.cost import (
    CostOverrides,
    format_estimate,
    price_catalog,
    recompute_cost,
)
from ohho.bench.environment import GpuProbe
from ohho.bench.errors import BenchHarnessError
from ohho.bench.exports import export_result
from ohho.bench.generate import GenerateConfig, execute_generate
from ohho.bench.harness import BenchConfig, execute
from ohho.bench.pipeline import PipelineConfig, execute_pipeline
from ohho.bench.sensors.base import SensorSource
from ohho.bench.sweep import SweepConfig, execute_sweep, render_pareto_svg
from ohho.bench.report import (
    dump_result,
    log_wandb,
    push_prometheus,
    render_markdown,
    write_text,
)
from ohho.bench.schema import BenchValidationError, validate, validate_path


def register_bench(subparsers: Any) -> None:
    """Add ``bench`` to the root CLI parser."""
    bench = subparsers.add_parser(
        "bench",
        help="benchmark an inference artifact, an LLM server, or validate a result",
        description=(
            "Run an inference benchmark, a camera-to-action pipeline benchmark, "
            "measure an external LLM server, or validate an ohho.bench/v1 result. "
            "A bare `ohho bench` is not a run. "
            "`--export` sends the result to an optional exporter plugin."
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
            "  ohho bench --artifact model.onnx --target x86-cpu --export <spec> "
            "--export-ttl 5 --json out.json\n"
            "  ohho bench --pipeline --hz 30 --json out.json\n"
            "  ohho bench --pipeline --fixture stream.json --sink inprocess\n"
            "  ohho bench validate out.json\n"
            "  ohho bench --mode generate --target x86-cpu --workload llm "
            "--slo ttft_p95<=50ms,tpot_p95<=20ms --json out.json\n"
            "  ohho bench launch --backend vllm --model org/model --dry-run\n"
            "  ohho bench llm --endpoint http://127.0.0.1:8000/v1 "
            "--model org/model --output result.json\n"
            "  ohho bench llm --sweep --backend stand-in --target-p95-ttft 40ms "
            "--output result.json --chart pareto.svg\n"
            "  ohho bench cost result.json --gpu-hour-usd 2.5 --output repriced.json\n"
            "  ohho bench prices\n"
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
        "--export",
        action="append",
        default=None,
        help=(
            "repeatable exporter spec (scheme, scheme:target, or scheme=url). "
            "A failure is a warning and does not change the bench exit code."
        ),
    )
    bench.add_argument(
        "--export-ttl",
        dest="export_ttl",
        type=float,
        default=5.0,
        help=(
            "minutes to keep a pushed grouping key before deleting it "
            "(default 5). The process waits, then deletes."
        ),
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
    bench.add_argument(
        "--mode",
        choices=("inference", "generate"),
        default="inference",
        help="inference times an artifact; generate measures TTFT and TPOT",
    )
    bench.add_argument(
        "--backend",
        default=None,
        help="generate backend: stand-in, transformers, vllm, sglang, tensorrt-llm",
    )
    bench.add_argument(
        "--model",
        default=None,
        help="model id for generate mode (default stand-in)",
    )
    bench.add_argument(
        "--precision",
        default=None,
        help="comma-separated generate precisions (default fp16,bnb-int4)",
    )
    bench.add_argument(
        "--concurrency",
        default=None,
        help="comma-separated generate concurrencies (default 1,4,16)",
    )
    bench.add_argument(
        "--prompt-tokens",
        dest="prompt_tokens",
        default=None,
        help="comma-separated prompt lengths (default 16,64)",
    )
    bench.add_argument(
        "--image-sizes",
        dest="image_sizes",
        default=None,
        help="comma-separated square image sides; llm stays at 0",
    )
    bench.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=None,
        help="output tokens per generate request (default 8)",
    )
    bench.add_argument(
        "--requests",
        type=int,
        default=None,
        help="generate requests per cell; raised to the concurrency when lower",
    )
    _add_cost_flags(bench, "")
    commands = bench.add_subparsers(dest="bench_command", required=False)
    validate = commands.add_parser(
        "validate",
        help="validate a result file against ohho.bench/v1",
    )
    validate.add_argument("file", help="path to a JSON result")
    validate.set_defaults(func=run_validate)
    _register_cost_commands(commands)
    _register_llm_commands(commands)
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


def run_version(args: argparse.Namespace) -> int:
    from ohho.serve.backends import ServingBackendError, open_backend

    config: dict[str, object] = {}
    if args.executable:
        config["executable"] = args.executable
    if args.launcher:
        config["launcher"] = args.launcher
    try:
        print(open_backend(args.backend).version(config))
    except ServingBackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _bench_launch_config(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {
        "host": args.host,
        "startup_timeout_s": args.startup_timeout,
    }
    if args.port is not None:
        config["port"] = args.port
    if args.executable:
        config["executable"] = args.executable
    if args.launcher:
        config["launcher"] = args.launcher
    if args.model_repository:
        config["model_repository"] = args.model_repository
    if args.n_gpu_layers is not None:
        config["n_gpu_layers"] = args.n_gpu_layers
    if args.ctx_size is not None:
        config["ctx_size"] = args.ctx_size
    if args.tp_size is not None:
        config["tp_size"] = args.tp_size
    if args.extra_arg:
        config["extra_args"] = list(args.extra_arg)
    return config


def run_launch(args: argparse.Namespace) -> int:
    import json
    import time

    from ohho.serve.backends import ServingBackendError, open_backend

    config = _bench_launch_config(args)
    try:
        backend = open_backend(args.backend)
        if args.dry_run:
            config["dry_run"] = True
            plan = backend.plan(args.model, args.precision, config)
            json.dump(
                {
                    "backend": backend.name,
                    "commands": [list(command) for command in plan.commands],
                    "endpoint_url": plan.endpoint_url,
                    "health_url": plan.health_url,
                    "version_argv": list(plan.version_argv),
                    "precision": plan.precision,
                },
                sys.stdout,
                indent=2,
            )
            sys.stdout.write("\n")
            return 0
        backend.launch(args.model, args.precision, config)
    except ServingBackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(backend.endpoint_url())
    try:
        while backend.running:
            time.sleep(0.5)
        print("error: serving process exited", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        backend.teardown()


def run_llm(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from ohho.bench.schema import BenchValidationError
    from ohho.serve.loadgen import LoadGenError, LoadSpec, run_load, to_bench_result

    spec = LoadSpec(
        endpoint_url=args.endpoint,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        requests=args.requests,
        timeout_s=args.timeout,
    )
    try:
        result = run_load(spec)
        document = to_bench_result(
            result,
            backend=args.backend_name,
            backend_version=args.backend_version,
            precision=args.precision,
        )
    except (LoadGenError, BenchValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    path = Path(args.output)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


def run_bench_command(
    args: argparse.Namespace,
    *,
    runner: Any = None,
    clock: Any = None,
    sources: Sequence[SensorSource] | None = None,
    gpu: GpuProbe | None = None,
    export_context: Any = None,
) -> int:
    """Run the harness from parsed CLI arguments.

    Tests pass ``runner`` and ``clock`` to avoid a real session. The argparse
    path leaves both unset and loads the ONNX artifact. ``--mode generate``
    measures an autoregressive model instead of an ONNX artifact.
    ``export_context`` stands in for an optional exporter client. Export
    failures are warnings.
    """
    if getattr(args, "mode", "inference") == "generate":
        return _run_generate(args)
    try:
        _check_export_ttl(args)
        if getattr(args, "pipeline", False):
            document = execute_pipeline(
                _pipeline_config(args),
                clock=clock,
                baseline_path=getattr(args, "baseline", None),
                slos=_expand_slos(tuple(args.slo or ())),
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
                slos=_expand_slos(tuple(args.slo or ())),
                max_regressions=tuple(args.max_regression or ()),
                parity_path=getattr(args, "parity", None),
                cost=_cost_overrides(args),
            )
        try:
            messages = _auto_export(args, document, export_context)
        except Exception as exc:
            messages = [f"export failed: {exc}"]
        for message in messages:
            print(f"warning: {message}", file=sys.stderr)
        _emit(document, args)
        _print_estimate(document)
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


def _run_generate(args: argparse.Namespace) -> int:
    """``ohho bench --mode generate``."""
    try:
        if getattr(args, "artifact", None):
            raise BenchHarnessError("--mode generate does not take --artifact")
        if getattr(args, "batch", "1") not in (None, "1"):
            raise BenchHarnessError("generate mode sweeps --concurrency, not --batch")
        if getattr(args, "inputs", "random") != "random":
            raise BenchHarnessError(
                "generate mode does not read a dataset; set --prompt-tokens"
            )
        if getattr(args, "baseline", None) or getattr(args, "max_regression", None):
            raise BenchHarnessError(
                "generate mode does not take --baseline or --max-regression"
            )
        if getattr(args, "parity", None):
            raise BenchHarnessError(
                "generate mode scores top-1 action-token agreement and does not "
                "take --parity"
            )
        config = _generate_config(args)
        document = execute_generate(
            config,
            slos=_expand_slos(tuple(args.slo or ())),
            cost=_cost_overrides(args),
        )
        _emit(document, args)
        _print_estimate(document)
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


def _generate_config(args: argparse.Namespace) -> GenerateConfig:
    workload = args.workload
    if not workload or workload == "inference":
        workload = "llm"
    return GenerateConfig(
        target=args.target or "",
        workload=workload,
        backend=args.backend or "stand-in",
        model=args.model or "stand-in",
        precisions=_precisions(args.precision),
        concurrencies=_int_list(
            args.concurrency, "--concurrency", (1, 4, 16), minimum=1
        ),
        prompt_tokens=_int_list(
            args.prompt_tokens, "--prompt-tokens", (16, 64), minimum=1
        ),
        image_sizes=_int_list(args.image_sizes, "--image-sizes", (), minimum=0)
        if args.image_sizes
        else (),
        max_tokens=8 if args.max_tokens is None else args.max_tokens,
        requests=4 if args.requests is None else args.requests,
        warmup=args.warmup,
    )


def _precisions(text: str | None) -> tuple[str, ...]:
    if text is None or not str(text).strip():
        return ("fp16", "bnb-int4")
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if not parts:
        raise BenchHarnessError("--precision needs at least one value")
    return parts


def _int_list(
    text: str | None,
    flag: str,
    default: tuple[int, ...],
    *,
    minimum: int,
) -> tuple[int, ...]:
    if text is None or (isinstance(text, str) and not text.strip()):
        return default
    if not isinstance(text, str):
        raise BenchHarnessError(f"{flag} needs a comma-separated list of integers")
    values: list[int] = []
    for part in text.split(","):
        piece = part.strip()
        if not piece.isdigit():
            raise BenchHarnessError(
                f"{flag} values must be integers >= {minimum} (got {text!r})"
            )
        value = int(piece)
        if value < minimum:
            raise BenchHarnessError(
                f"{flag} values must be integers >= {minimum} (got {text!r})"
            )
        values.append(value)
    if not values:
        raise BenchHarnessError(f"{flag} needs at least one integer")
    return tuple(values)


def _expand_slos(values: tuple[str, ...]) -> tuple[str, ...]:
    expanded: list[str] = []
    for value in values:
        expanded.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(expanded)


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


def _check_export_ttl(args: argparse.Namespace) -> None:
    specs = getattr(args, "export", None) or []
    if not specs:
        return
    ttl = getattr(args, "export_ttl", 5.0)
    if (
        isinstance(ttl, bool)
        or not isinstance(ttl, (int, float))
        or not math.isfinite(float(ttl))
        or float(ttl) < 0
    ):
        raise BenchHarnessError("--export-ttl must be a non-negative number of minutes")


def _job_spec(args: argparse.Namespace) -> dict[str, Any]:
    raw = {
        "command": "bench",
        "artifact": getattr(args, "artifact", None),
        "target": getattr(args, "target", None),
        "warmup": getattr(args, "warmup", None),
        "iters": getattr(args, "iters", None),
        "batch": getattr(args, "batch", None),
        "inputs": getattr(args, "inputs", None),
        "workload": getattr(args, "workload", None),
        "sustained": getattr(args, "sustained", None),
        "baseline": getattr(args, "baseline", None),
        "slo": list(getattr(args, "slo", None) or []),
        "max_regression": list(getattr(args, "max_regression", None) or []),
        "parity": getattr(args, "parity", None),
    }
    return {
        key: value
        for key, value in raw.items()
        if value is not None and value != "" and value != []
    }


def _auto_export(
    args: argparse.Namespace,
    document: dict[str, Any],
    context: Any,
) -> list[str]:
    specs = list(getattr(args, "export", None) or [])
    if not specs:
        return []
    run = document.get("run")
    if not isinstance(run, dict):
        run = {}
    manifest = {
        "manifest_ref": run.get("manifest_ref"),
        "artifact_sha": run.get("artifact_sha"),
        "model_sha": run.get("model_sha"),
        "run_id": run.get("id"),
        "workload": run.get("target"),
    }
    return export_result(
        document,
        specs,
        kind="bench",
        spec=_job_spec(args),
        manifest=manifest,
        ttl_minutes=getattr(args, "export_ttl", 5.0),
        context=context,
    )


def _exporters(args: argparse.Namespace, document: dict[str, Any]) -> None:
    url = getattr(args, "prometheus_push", None)
    if url:
        push_prometheus(url, document)
    if getattr(args, "wandb", False):
        log_wandb(document)


_LLM_BACKENDS = ("vllm", "sglang", "tensorrt-llm", "llama-cpp")


def _register_llm_commands(commands: Any) -> None:
    """Add the OHH-475 external-engine subcommands."""
    version = commands.add_parser(
        "version",
        help="print an external LLM engine version",
    )
    version.add_argument("--backend", required=True, choices=_LLM_BACKENDS)
    version.add_argument(
        "--executable",
        default=None,
        help="engine executable to query instead of the default",
    )
    version.add_argument(
        "--launcher",
        default=None,
        choices=("trtllm-serve", "triton"),
        help="tensorrt-llm only: trtllm-serve (default) or triton",
    )
    version.set_defaults(func=run_bench_version)
    launch = commands.add_parser(
        "launch",
        help="launch an external OpenAI-compatible LLM server",
    )
    launch.add_argument("--backend", required=True, choices=_LLM_BACKENDS)
    launch.add_argument("--model", required=True)
    launch.add_argument("--precision", default="auto")
    launch.add_argument("--host", default="127.0.0.1")
    launch.add_argument("--port", type=int, default=None)
    launch.add_argument("--startup-timeout", type=float, default=30.0)
    launch.add_argument("--executable", default=None)
    launch.add_argument(
        "--launcher",
        default=None,
        choices=("trtllm-serve", "triton"),
    )
    launch.add_argument("--model-repository", default=None)
    launch.add_argument("--n-gpu-layers", default=None, dest="n_gpu_layers")
    launch.add_argument("--ctx-size", type=int, default=None)
    launch.add_argument("--tp-size", type=int, default=None)
    launch.add_argument("--extra-arg", action="append", default=[])
    launch.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command and URLs as JSON and do not start a process",
    )
    launch.set_defaults(func=run_bench_launch)
    llm = commands.add_parser(
        "llm",
        help="measure TTFT, inter-token latency, and throughput",
        description=(
            "Measure one running OpenAI-compatible server, or sweep "
            "concurrency, max batch tokens, prefix caching, chunked prefill, "
            "and speculative decoding. A sweep skips options the backend "
            "cannot change and records the reason. The CPU sweep uses "
            "--backend stand-in and does not launch an engine."
        ),
    )
    llm.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "sweep concurrency, max batch tokens, prefix cache, "
            "chunked prefill, and speculative decoding"
        ),
    )
    llm.add_argument(
        "--endpoint",
        default=None,
        help="OpenAI base URL, .../v1. Required unless --sweep --backend stand-in",
    )
    llm.add_argument(
        "--model",
        default=None,
        help="model id. Defaults to stand-in for a CPU sweep",
    )
    llm.add_argument("--output", required=True, help="ohho.bench/v1 JSON path")
    llm.add_argument(
        "--backend",
        default=None,
        help="sweep backend: stand-in, vllm, sglang, tensorrt-llm, or llama-cpp",
    )
    llm.add_argument(
        "--concurrency",
        default=None,
        help="in-flight requests. A sweep accepts a comma list (default 1,2,4,8)",
    )
    llm.add_argument("--requests", type=int, default=4)
    llm.add_argument("--max-tokens", type=int, default=8)
    llm.add_argument("--prompt", default="ping")
    llm.add_argument("--precision", default=None)
    llm.add_argument("--backend-name", default="openai-compatible")
    llm.add_argument("--backend-version", default=None)
    llm.add_argument("--timeout", type=float, default=30.0)
    llm.add_argument(
        "--max-batch-tokens",
        dest="max_batch_tokens",
        default=None,
        help="comma-separated prefill budgets (sweep default 512,2048)",
    )
    llm.add_argument(
        "--prefix-cache",
        dest="prefix_cache",
        default=None,
        help="on, off, or on,off (sweep default on,off)",
    )
    llm.add_argument(
        "--chunked-prefill",
        dest="chunked_prefill",
        default=None,
        help="on, off, or on,off (sweep default on,off)",
    )
    llm.add_argument(
        "--speculative",
        default=None,
        help="on, off, or on,off. Skipped when the backend has no draft model",
    )
    llm.add_argument(
        "--speculative-model",
        dest="speculative_model",
        default="",
        help="draft model id. Required before speculative decoding is measured",
    )
    llm.add_argument(
        "--workloads",
        default=None,
        help="comma list: short, long-context, shared-prefix",
    )
    llm.add_argument(
        "--target-p95-ttft",
        dest="target_p95_ttft",
        default=None,
        help="recommend the fastest cell at or under this p95 TTFT (ms, or 40ms)",
    )
    llm.add_argument(
        "--chart",
        default=None,
        help="write an SVG of p95 TTFT against tokens/s",
    )
    llm.add_argument(
        "--launcher",
        default=None,
        choices=("trtllm-serve", "triton"),
        help="tensorrt-llm launcher. The triton launcher has no batch sweep flag",
    )
    llm.add_argument(
        "--target",
        dest="llm_target",
        default=None,
        help="hardware target profile id used for the cost estimate",
    )
    _add_cost_flags(llm, "llm_")
    llm.set_defaults(func=run_bench_llm)


def run_bench_version(args: argparse.Namespace) -> int:
    """``ohho bench version --backend NAME``."""
    from ohho.serve.backends import ServingBackendError, open_backend

    config: dict[str, object] = {}
    if args.executable:
        config["executable"] = args.executable
    if args.launcher:
        config["launcher"] = args.launcher
    try:
        print(open_backend(args.backend).version(config))
    except ServingBackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _launch_config(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {
        "host": args.host,
        "startup_timeout_s": args.startup_timeout,
    }
    if args.port is not None:
        config["port"] = args.port
    if args.executable:
        config["executable"] = args.executable
    if args.launcher:
        config["launcher"] = args.launcher
    if args.model_repository:
        config["model_repository"] = args.model_repository
    if args.n_gpu_layers is not None:
        config["n_gpu_layers"] = args.n_gpu_layers
    if args.ctx_size is not None:
        config["ctx_size"] = args.ctx_size
    if args.tp_size is not None:
        config["tp_size"] = args.tp_size
    if args.extra_arg:
        config["extra_args"] = list(args.extra_arg)
    return config


def run_bench_launch(args: argparse.Namespace) -> int:
    """``ohho bench launch`` starts an external engine, or prints a dry run."""
    from ohho.serve.backends import ServingBackendError, open_backend

    config = _launch_config(args)
    try:
        backend = open_backend(args.backend)
        if args.dry_run:
            config["dry_run"] = True
            plan = backend.plan(args.model, args.precision, config)
            json.dump(
                {
                    "backend": backend.name,
                    "commands": [list(command) for command in plan.commands],
                    "endpoint_url": plan.endpoint_url,
                    "health_url": plan.health_url,
                    "version_argv": list(plan.version_argv),
                    "precision": plan.precision,
                },
                sys.stdout,
                indent=2,
            )
            sys.stdout.write("\n")
            return 0
        backend.launch(args.model, args.precision, config)
    except ServingBackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(backend.endpoint_url())
    try:
        while backend.running:
            time.sleep(0.5)
        print("error: serving process exited", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        backend.teardown()


def run_bench_llm(args: argparse.Namespace) -> int:
    """``ohho bench llm`` measures a server, or sweeps serving options."""
    if getattr(args, "sweep", False):
        return _run_llm_sweep(args)
    from ohho.bench.schema import BenchValidationError
    from ohho.serve.loadgen import LoadGenError, LoadSpec, run_load, to_bench_result

    if not getattr(args, "endpoint", None):
        print("error: --endpoint is required", file=sys.stderr)
        return 2
    if not getattr(args, "model", None):
        print("error: --model is required", file=sys.stderr)
        return 2
    try:
        concurrency = _one_int(args.concurrency, "--concurrency", default=1)
    except BenchHarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    spec = LoadSpec(
        endpoint_url=args.endpoint,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        concurrency=concurrency,
        requests=args.requests,
        timeout_s=args.timeout,
    )
    try:
        overrides = _cost_overrides(args, "llm_")
        target = _hardware_target(args)
        result = run_load(spec)
        document = to_bench_result(
            result,
            backend=args.backend_name,
            backend_version=args.backend_version or "unknown",
            precision=args.precision or "auto",
            target_profile=target,
            cost=overrides,
        )
    except (LoadGenError, BenchValidationError, BenchHarnessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return getattr(exc, "exit_code", 1)
    path = Path(args.output)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    _print_estimate(document)
    return 0


def _run_llm_sweep(args: argparse.Namespace) -> int:
    """``ohho bench llm --sweep``."""
    try:
        config = _sweep_config(args)
        document = execute_sweep(config, cost=_cost_overrides(args, "llm_"))
        payload = dump_result(document)
        chart = render_pareto_svg(document) if config.chart else ""
        write_text(args.output, payload)
        print(f"wrote {args.output}")
        if config.chart:
            write_text(config.chart, chart)
            print(f"wrote {config.chart}")
        _print_sweep(document)
        _print_estimate(document)
    except BenchHarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    return 0


def _sweep_config(args: argparse.Namespace) -> SweepConfig:
    backend = args.backend or "stand-in"
    model = args.model or ("stand-in" if backend in {"stand-in", "standin"} else "")
    return SweepConfig(
        backend=backend,
        model=model,
        endpoint=args.endpoint or "",
        precision=args.precision or ("fp16" if args.sweep else "auto"),
        concurrencies=_int_list(
            args.concurrency, "--concurrency", (1, 2, 4, 8), minimum=1
        ),
        max_batch_tokens=_int_list(
            args.max_batch_tokens, "--max-batch-tokens", (512, 2048), minimum=1
        ),
        prefix_cache=_bool_list(args.prefix_cache, "--prefix-cache"),
        chunked_prefill=_bool_list(args.chunked_prefill, "--chunked-prefill"),
        speculative=_bool_list(args.speculative, "--speculative"),
        workloads=_workload_list(args.workloads),
        max_tokens=args.max_tokens,
        requests=args.requests,
        timeout_s=args.timeout,
        target_p95_ttft_ms=_ttft_target(args.target_p95_ttft),
        speculative_model=args.speculative_model or "",
        launcher=args.launcher or "",
        backend_version=args.backend_version or "",
        chart=args.chart or "",
        hardware_target=_hardware_target(args),
    )


def _one_int(text: str | None, flag: str, *, default: int) -> int:
    values = _int_list(text, flag, (default,), minimum=1)
    if len(values) != 1:
        raise BenchHarnessError(f"{flag} must be one integer unless --sweep is set")
    return values[0]


def _bool_list(text: str | None, flag: str) -> tuple[bool, ...]:
    if text is None or not str(text).strip():
        return (False, True)
    values: list[bool] = []
    for part in str(text).split(","):
        key = part.strip().lower()
        if key in {"on", "true", "1", "yes"}:
            values.append(True)
        elif key in {"off", "false", "0", "no"}:
            values.append(False)
        else:
            raise BenchHarnessError(f"{flag} values must be on or off (got {text!r})")
    if not values:
        raise BenchHarnessError(f"{flag} needs at least one of on or off")
    return tuple(values)


def _workload_list(text: str | None) -> tuple[str, ...]:
    if text is None or not str(text).strip():
        return ("short", "long-context", "shared-prefix")
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def _ttft_target(text: str | None) -> float | None:
    if text is None or not str(text).strip():
        return None
    raw = str(text).strip().lower()
    scale = 1.0
    if raw.endswith("ms"):
        raw = raw[:-2].strip()
    elif raw.endswith("s"):
        raw = raw[:-1].strip()
        scale = 1000.0
    try:
        value = float(raw) * scale
    except ValueError as exc:
        raise BenchHarnessError(
            f"--target-p95-ttft must be a number of milliseconds (got {text!r})"
        ) from exc
    if value < 0:
        raise BenchHarnessError("--target-p95-ttft must be non-negative")
    return value


def _register_cost_commands(commands: Any) -> None:
    """``ohho bench cost`` reprices a result. ``ohho bench prices`` prints the table."""
    cost = commands.add_parser(
        "cost",
        help="recompute cost estimates from a saved bench result",
        description=(
            "Read an ohho.bench/v1 result and recompute cost estimates from its "
            "measured throughput. This does not rerun the benchmark. Costs are "
            "estimates: the JSON sets estimate to true and records the inputs used."
        ),
    )
    cost.add_argument("file", help="path to an ohho.bench/v1 JSON result")
    cost.add_argument(
        "--output",
        default=None,
        help="write the repriced result here. Default is stdout.",
    )
    cost.add_argument(
        "--target",
        dest="cost_target",
        default=None,
        help="price row to apply. Default is the profile stored on the result.",
    )
    _add_cost_flags(cost, "cost_")
    cost.set_defaults(func=run_bench_cost)
    prices = commands.add_parser(
        "prices",
        help="print the user-editable price table (estimates)",
        description=(
            "Print the price table used for cost estimates. "
            "Every figure is an estimate, not an invoice. "
            "Edit ohho/targets/cost_prices.json or pass --prices."
        ),
    )
    _add_cost_flags(prices, "prices_")
    prices.add_argument(
        "--json",
        action="store_true",
        dest="prices_json",
        help="print the price table as JSON",
    )
    prices.set_defaults(func=run_bench_prices)


def run_bench_cost(args: argparse.Namespace) -> int:
    """``ohho bench cost FILE`` rewrites cost fields and keeps measured throughput."""
    try:
        document = _load_result(args.file)
        profile = getattr(args, "cost_target", None) or getattr(args, "target", None)
        updated = recompute_cost(
            document, _cost_overrides(args, "cost_", profile=profile)
        )
        validate(updated)
    except BenchValidationError as exc:
        print(exc, file=sys.stderr)
        return 1
    except BenchHarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    payload = dump_result(updated)
    output = getattr(args, "output", None)
    if output:
        try:
            write_text(output, payload)
        except BenchHarnessError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return exc.exit_code
        print(f"wrote {output}")
    else:
        sys.stdout.write(payload)
    _print_estimate(updated)
    return 0


def run_bench_prices(args: argparse.Namespace) -> int:
    """``ohho bench prices`` prints the estimate price table."""
    try:
        document = price_catalog(_cost_overrides(args, "prices_"))
    except BenchHarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    if getattr(args, "prices_json", False):
        sys.stdout.write(dump_result(document))
        return 0
    print(
        "estimate: user-editable prices. Not an invoice, a cloud bill, or a hardware quote."
    )
    print(document["note"])
    print(
        "electricity_usd_per_kwh="
        f"{document['electricity_usd_per_kwh']} "
        f"amortisation_hours={document['amortisation_hours']} "
        f"control_rate_hz={document['control_rate_hz']}"
    )
    print("| profile | pricing | gpu_hour_usd | hardware_usd | typical_watts |")
    print("| --- | --- | ---: | ---: | ---: |")
    for profile_id, row in document["profiles"].items():
        print(
            f"| {profile_id} | {row['pricing']} | "
            f"{row.get('gpu_hour_usd', '')} | {row.get('hardware_usd', '')} | "
            f"{row.get('typical_watts', '')} |"
        )
    return 0


def _load_result(path: str) -> dict[str, Any]:
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BenchHarnessError(f"result not found: {path}") from exc
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise BenchHarnessError(f"cannot read result: {detail}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BenchHarnessError(
            f"result is not JSON: {exc.msg} (line {exc.lineno})"
        ) from exc
    if not isinstance(parsed, dict):
        raise BenchHarnessError("result must be a JSON object")
    try:
        validate(parsed)
    except BenchValidationError as exc:
        raise BenchHarnessError(
            f"result is not ohho.bench/v1:\n{exc}", exit_code=1
        ) from exc
    return parsed


def _add_cost_flags(parser: Any, prefix: str) -> None:
    """Price overrides. ``prefix`` keeps subparser dests from clobbering the parent."""

    def dest(name: str) -> str:
        return f"{prefix}{name}" if prefix else name

    parser.add_argument(
        "--prices",
        dest=dest("prices"),
        default=None,
        help="JSON price table. Merged over the built-in estimates. Does not rerun a benchmark.",
    )
    parser.add_argument(
        "--gpu-hour-usd",
        dest=dest("gpu_hour_usd"),
        type=float,
        default=None,
        help="USD per GPU-hour estimate. Replaces the profile rate.",
    )
    parser.add_argument(
        "--hardware-usd",
        dest=dest("hardware_usd"),
        type=float,
        default=None,
        help="edge purchase-price estimate, amortised over --amortise-hours",
    )
    parser.add_argument(
        "--amortise-hours",
        dest=dest("amortise_hours"),
        type=float,
        default=None,
        help="hours used to amortise --hardware-usd (default 26280)",
    )
    parser.add_argument(
        "--electricity-usd-per-kwh",
        dest=dest("electricity_usd_per_kwh"),
        type=float,
        default=None,
        help="electricity price estimate in USD per kWh for edge profiles",
    )
    parser.add_argument(
        "--watts",
        dest=dest("watts"),
        type=float,
        default=None,
        help="device watts. Overrides measured power and the profile default.",
    )
    parser.add_argument(
        "--control-rate-hz",
        dest=dest("control_rate_hz"),
        type=float,
        default=None,
        help=(
            "control rate in Hz. Sets meets_control_rate, robots per device, "
            "and USD per robot-hour (default 50). USD per 1k actions uses "
            "measured actions per second."
        ),
    )


def _cost_overrides(
    args: argparse.Namespace,
    prefix: str = "",
    *,
    profile: str | None = None,
) -> CostOverrides:
    def pick(name: str) -> Any:
        if prefix:
            chosen = getattr(args, f"{prefix}{name}", None)
            if chosen is not None:
                return chosen
        return getattr(args, name, None)

    return CostOverrides(
        prices_path=pick("prices"),
        profile=profile,
        gpu_hour_usd=pick("gpu_hour_usd"),
        hardware_usd=pick("hardware_usd"),
        amortisation_hours=pick("amortise_hours"),
        electricity_usd_per_kwh=pick("electricity_usd_per_kwh"),
        watts=pick("watts"),
        control_rate_hz=pick("control_rate_hz"),
    )


def _hardware_target(args: argparse.Namespace) -> str:
    chosen = getattr(args, "llm_target", None) or getattr(args, "target", None) or ""
    return str(chosen).strip()


def _print_estimate(document: dict[str, Any]) -> None:
    cost = document.get("cost")
    if isinstance(cost, dict):
        print(format_estimate(cost), file=sys.stderr)


def _print_sweep(document: dict[str, Any]) -> None:
    llm = document["llm"]
    recommendation = llm.get("recommendation")
    if isinstance(recommendation, dict):
        if recommendation.get("met"):
            print(
                "recommend "
                f"workload={recommendation['workload']} "
                f"concurrency={recommendation['concurrency']} "
                f"prefix_cache={recommendation.get('prefix_cache')} "
                f"max_batch_tokens={recommendation.get('max_batch_tokens')} "
                f"chunked_prefill={recommendation.get('chunked_prefill')} "
                f"ttft_p95_ms={recommendation['ttft_p95_ms']:.2f} "
                f"tokens_per_s={recommendation['tokens_per_s']:.2f}"
            )
        else:
            print(recommendation["reason"])
    pareto = llm.get("pareto") or []
    print(f"pareto rows: {len(pareto)}")
    for row in llm.get("skipped") or []:
        print(f"skipped {row['option']}: {row['reason']}")
