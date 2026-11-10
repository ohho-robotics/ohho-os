"""JSON, Markdown, Prometheus, and optional W&B exporters.

Prometheus and W&B run only when the caller asks. Tests do not need either
package. The Markdown is a GitHub step summary: one heading, then pipe tables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ohho.bench.errors import BenchHarnessError

_LATENCY_ROWS = (
    "mean",
    "median",
    "p50",
    "p90",
    "p95",
    "p99",
    "min",
    "max",
    "std",
    "cold_start_ms",
)


def dump_result(document: dict[str, Any]) -> str:
    """Pretty JSON with a trailing newline. Non-finite numbers are rejected."""
    return json.dumps(document, indent=2, allow_nan=False) + "\n"


def write_text(path: str | Path, text: str) -> None:
    """Write ``text`` as UTF-8. A missing parent directory is an error."""
    file = Path(path)
    parent = file.parent
    if str(parent) not in {"", "."} and not parent.exists():
        raise BenchHarnessError(f"directory does not exist: {parent}")
    file.write_text(text, encoding="utf-8", newline="\n")


def render_markdown(document: dict[str, Any]) -> str:
    """GitHub-flavoured Markdown summary of one bench result."""
    run = document["run"]
    env = document["environment"]
    latency = document["latency"]
    unit = str(latency.get("unit") or "ms")
    lines = ["## Ohho bench", ""]
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Artifact | {_cell(run.get('manifest_ref', ''))} |")
    lines.append(f"| Workload | {_cell(run.get('target', ''))} |")
    lines.append(f"| Target profile | {_cell(env.get('target_profile', ''))} |")
    lines.append(f"| Execution provider | {_cell(env.get('execution_provider', ''))} |")
    lines.append(f"| Timing | {_cell(env.get('timing_source', ''))} |")
    lines.append(f"| Git | {_cell(env.get('git_sha', ''))} |")
    if env.get("driver"):
        lines.append(f"| Driver | {_cell(env['driver'])} |")
    lines.append("")
    lines.append("### Latency")
    lines.append("")
    lines.append("| Stat | Value |")
    lines.append("| --- | ---: |")
    for stat in _LATENCY_ROWS:
        if stat in latency:
            lines.append(f"| {stat} | {_num(latency[stat])} {unit} |")
    if "achievable_hz" in latency:
        lines.append(f"| achievable_hz | {_num(latency['achievable_hz'])} Hz |")
    lines.append("")
    lines.append("### Stages")
    lines.append("")
    lines.append("| Stage | mean | median | p95 | min | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for stage in latency.get("stages", []):
        lines.append(
            "| {name} | {mean} | {median} | {p95} | {min} | {max} |".format(
                name=stage.get("name", ""),
                mean=_num(stage.get("mean", 0)),
                median=_num(stage.get("median", 0)),
                p95=_num(stage.get("p95", 0)),
                min=_num(stage.get("min", 0)),
                max=_num(stage.get("max", 0)),
            )
        )
    pipeline = document.get("pipeline")
    if isinstance(pipeline, dict):
        lines.append("")
        lines.append("### Pipeline")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| World | {_cell(pipeline.get('world', ''))} |")
        lines.append(f"| Source | {_cell(pipeline.get('source', ''))} |")
        lines.append(f"| Device | {_cell(pipeline.get('device', ''))} |")
        lines.append(f"| Target Hz | {_num(pipeline.get('target_hz', 0))} |")
        lines.append(f"| Achieved Hz | {_num(pipeline.get('achieved_hz', 0))} |")
        lines.append(f"| Dropped frames | {pipeline.get('dropped_frames', '')} |")
        sustained = "yes" if pipeline.get("rate_sustained") else "no"
        lines.append(f"| Rate sustained | {sustained} |")
        lines.append(f"| Jitter ms | {_num(pipeline.get('jitter_ms', 0))} |")
        lines.append(
            f"| Interval jitter ms | {_num(pipeline.get('interval_jitter_ms', 0))} |"
        )
        lines.append(
            f"| Live robot | {'yes' if pipeline.get('live_robot') else 'no'} |"
        )
    batches = document.get("batches")
    if isinstance(batches, list) and batches:
        lines.append("")
        lines.append("### Batches")
        lines.append("")
        lines.append("| Batch | mean | p95 | Hz |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for item in batches:
            item_latency = item.get("latency", {})
            lines.append(
                f"| {item.get('batch_size', '')} | {_num(item_latency.get('mean', 0))} "
                f"| {_num(item_latency.get('p95', 0))} "
                f"| {_num(item_latency.get('achievable_hz', 0))} |"
            )
    telemetry = document.get("telemetry", {})
    lines.append("")
    lines.append("### Resources")
    lines.append("")
    lines.append("| Gauge | Value |")
    lines.append("| --- | --- |")
    if "memory_mb_max" in telemetry:
        lines.append(f"| peak memory MiB | {_num(telemetry['memory_mb_max'])} |")
    lines.append(f"| GPU util % | {_optional(telemetry.get('gpu_util_percent'))} |")
    lines.append(f"| GPU power W | {_optional(telemetry.get('gpu_power_w'))} |")
    gates = document.get("gates")
    if isinstance(gates, dict):
        lines.append("")
        lines.append("### Gates")
        lines.append("")
        lines.append("| Gate | Passed | Detail |")
        lines.append("| --- | --- | --- |")
        for check in gates.get("checks", []):
            flag = "yes" if check.get("passed") else "no"
            lines.append(
                f"| {_cell(check.get('expr', ''))} | {flag} | {_cell(check.get('detail', ''))} |"
            )
        lines.append("")
        lines.append(f"Gates passed: {'yes' if gates.get('passed') else 'no'}")
    comparison = document.get("comparison")
    if isinstance(comparison, dict):
        lines.append("")
        lines.append("### Baseline")
        lines.append("")
        lines.append(f"Baseline: `{_cell(comparison.get('baseline_ref', ''))}`")
        lines.append("")
        lines.append("| Metric | Baseline | Current | Delta | Change % | Regression |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for row in comparison.get("metrics", []):
            change = row.get("pct_change")
            change_text = "" if change is None else _num(change)
            lines.append(
                "| {name} | {baseline} | {current} | {delta} | {change} | {flag} |".format(
                    name=_cell(row.get("name", "")),
                    baseline=_num(row.get("baseline", 0)),
                    current=_num(row.get("current", 0)),
                    delta=_num(row.get("delta", 0)),
                    change=change_text,
                    flag="yes" if row.get("regression") else "no",
                )
            )
    if "parity" in document:
        lines.append("")
        lines.append("### Parity")
        lines.append("")
        lines.append("A parity document is embedded on this result.")
    lines.append("")
    return "\n".join(lines)


def render_prometheus(document: dict[str, Any]) -> str:
    """Prometheus text exposition for the primary batch. No network I/O."""
    profile = _label(str(document.get("environment", {}).get("target_profile", "")))
    workload = _label(str(document.get("run", {}).get("target", "")))
    latency = document["latency"]
    lines = []
    for stat in ("mean", "median", "p90", "p95", "p99", "min", "max"):
        if stat not in latency:
            continue
        lines.append(
            "ohho_bench_latency_ms"
            f'{{stat="{stat}",target="{profile}",workload="{workload}"}} '
            f"{float(latency[stat])}"
        )
    if "achievable_hz" in latency:
        lines.append(
            "ohho_bench_achievable_hz"
            f'{{target="{profile}",workload="{workload}"}} '
            f"{float(latency['achievable_hz'])}"
        )
    if "memory_mb_max" in document.get("telemetry", {}):
        lines.append(
            "ohho_bench_memory_mb_max"
            f'{{target="{profile}",workload="{workload}"}} '
            f"{float(document['telemetry']['memory_mb_max'])}"
        )
    gates = document.get("gates")
    if isinstance(gates, dict):
        lines.append(
            "ohho_bench_gate_passed"
            f'{{target="{profile}",workload="{workload}"}} '
            f"{1 if gates.get('passed') else 0}"
        )
    pipeline = document.get("pipeline")
    if isinstance(pipeline, dict):
        lines.append(
            "ohho_bench_pipeline_dropped_frames"
            f'{{target="{profile}",workload="{workload}"}} '
            f"{int(pipeline.get('dropped_frames', 0))}"
        )
        lines.append(
            "ohho_bench_pipeline_rate_sustained"
            f'{{target="{profile}",workload="{workload}"}} '
            f"{1 if pipeline.get('rate_sustained') else 0}"
        )
        lines.append(
            "ohho_bench_pipeline_jitter_ms"
            f'{{target="{profile}",workload="{workload}"}} '
            f"{float(pipeline.get('jitter_ms', 0))}"
        )
    lines.append("")
    return "\n".join(lines)


def push_prometheus(url: str, document: dict[str, Any], *, opener: Any = None) -> None:
    """POST the exposition to a Pushgateway URL. Not called unless requested."""
    import urllib.request

    body = render_prometheus(document).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "text/plain; version=0.0.4"},
    )
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=5) as response:
            response.read()
    except BenchHarnessError:
        raise
    except Exception as exc:
        raise BenchHarnessError(
            f"Prometheus push failed: {exc}. The bench result was still written."
        ) from exc


def log_wandb(
    document: dict[str, Any],
    *,
    module: Any = None,
    loader: Any = None,
) -> None:
    """Log the primary latency to Weights & Biases. Not called unless requested."""
    wandb = module
    if wandb is None:
        load = loader or _load_wandb
        try:
            wandb = load()
        except ImportError as exc:
            raise BenchHarnessError(
                "W&B export needs the wandb package, which is not installed. "
                "The bench result was still written."
            ) from exc
    latency = document["latency"]
    payload = {
        f"latency/{key}": latency[key]
        for key in ("mean", "median", "p95", "min", "max")
        if key in latency
    }
    if "achievable_hz" in latency:
        payload["throughput/hz"] = latency["achievable_hz"]
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "ohho-bench"),
            reinit=True,
        )
        wandb.log(payload)
        finish = getattr(run, "finish", None)
        if finish is not None:
            finish()
    except BenchHarnessError:
        raise
    except Exception as exc:
        raise BenchHarnessError(
            f"W&B export failed: {exc}. The bench result was still written."
        ) from exc


def _load_wandb() -> Any:
    import wandb

    return wandb


def _num(value: object) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return ""


def _optional(value: object) -> str:
    if value is None:
        return "null"
    return _num(value)


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "").replace('"', '\\"')
