"""``ohho bench`` command: run a benchmark, or validate a result file.

``ohho bench validate FILE`` is the schema check from OHH-473. A run is
``ohho bench --artifact PATH --target PROFILE``. ``ohho bench`` with neither
is rejected by the root parser so the historical exit code 2 stays.

OHH-475 adds ``version``, ``launch``, and ``llm`` on the same parser. Those
talk to external OpenAI-compatible engines and do not replace the harness.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from ohho.bench.errors import BenchHarnessError
from ohho.bench.generate import GenerateConfig, execute_generate
from ohho.bench.harness import BenchConfig, execute
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
        help="benchmark an inference artifact, an LLM server, or validate a result",
        description=(
            "Run an inference benchmark, measure an external LLM server, "
            "or validate an ohho.bench/v1 result. "
            "A bare `ohho bench` is not a run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ohho bench --artifact model.onnx --target x86-cpu "
            "--warmup 10 --iters 50 --batch 1,4,8 --inputs random --json out.json\n"
            "  ohho bench --artifact model.onnx --target x86-cpu --baseline prev.json "
            "--slo p95<=200ms --max-regression p95=10%% --markdown summary.md\n"
            "  ohho bench validate out.json\n"
            "  ohho bench --mode generate --target x86-cpu --workload llm "
            "--slo ttft_p95<=50ms,tpot_p95<=20ms --json out.json\n"
            "  ohho bench launch --backend vllm --model org/model --dry-run\n"
            "  ohho bench llm --endpoint http://127.0.0.1:8000/v1 "
            "--model org/model --output result.json\n"
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
        default=10,
        help="discarded iterations after the cold start (default 10)",
    )
    bench.add_argument(
        "--iters",
        type=int,
        default=50,
        help="measured iterations per batch (default 50)",
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
        default="inference",
        help="workload class stored in run.target (default inference)",
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
    commands = bench.add_subparsers(dest="bench_command", required=False)
    validate = commands.add_parser(
        "validate",
        help="validate a result file against ohho.bench/v1",
    )
    validate.add_argument("file", help="path to a JSON result")
    validate.set_defaults(func=run_validate)
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


def run_bench_command(
    args: argparse.Namespace,
    *,
    runner: Any = None,
    clock: Any = None,
) -> int:
    """Run the harness from parsed CLI arguments.

    Tests pass ``runner`` and ``clock`` to avoid a real session. The argparse
    path leaves both unset and loads the ONNX artifact. ``--mode generate``
    measures an autoregressive model instead of an ONNX artifact.
    """
    if getattr(args, "mode", "inference") == "generate":
        return _run_generate(args)
    try:
        config = _config(args)
        document = execute(
            config,
            runner=runner,
            clock=clock,
            baseline_path=getattr(args, "baseline", None),
            slos=_expand_slos(tuple(args.slo or ())),
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
        document = execute_generate(config, slos=_expand_slos(tuple(args.slo or ())))
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


def _generate_config(args: argparse.Namespace) -> GenerateConfig:
    workload = args.workload
    if workload == "inference":
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
    return BenchConfig(
        artifact=artifact,
        target=target,
        warmup=args.warmup,
        iters=args.iters,
        batches=_batches(args.batch),
        inputs=args.inputs,
        workload=args.workload,
    )


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
    )
    llm.add_argument("--endpoint", required=True, help="OpenAI base URL, .../v1")
    llm.add_argument("--model", required=True)
    llm.add_argument("--output", required=True, help="ohho.bench/v1 JSON path")
    llm.add_argument("--concurrency", type=int, default=1)
    llm.add_argument("--requests", type=int, default=4)
    llm.add_argument("--max-tokens", type=int, default=8)
    llm.add_argument("--prompt", default="ping")
    llm.add_argument("--precision", default="auto")
    llm.add_argument("--backend-name", default="openai-compatible")
    llm.add_argument("--backend-version", default="unknown")
    llm.add_argument("--timeout", type=float, default=30.0)
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
    """``ohho bench llm`` measures a running OpenAI-compatible server."""
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
