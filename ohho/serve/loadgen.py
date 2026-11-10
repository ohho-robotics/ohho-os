"""Load generator for an OpenAI-compatible streaming endpoint.

Measures time to first token, inter-token latency, output tokens per second,
and completed requests per second at a fixed concurrency. The result is an
``ohho.bench/v1`` document with an ``ohho.bench.llm/v1`` block.

Token counts are streamed completion chunks (one SSE ``delta.content`` or
``text`` piece per token). This module does not score answer quality.
``accuracy.primary`` is the fraction of requests that returned a token.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..bench.cost import CostOverrides

LLM_BLOCK_ID = "ohho.bench.llm/v1"
_MAX_HASH_BYTES = 8 * 1024 * 1024
_STATS = ("mean", "median", "p50", "p90", "p95", "p99", "min", "max", "std")


class LoadGenError(RuntimeError):
    """The load generator could not complete a measurement."""


@dataclass(frozen=True)
class LoadSpec:
    """One streaming benchmark against an OpenAI-compatible ``/v1`` base URL."""

    endpoint_url: str
    model: str
    prompt: str = "ping"
    max_tokens: int = 8
    concurrency: int = 1
    requests: int = 1
    timeout_s: float = 30.0
    system_prompt: str = ""
    image_url: str = ""
    prompts: tuple[str, ...] = ()


@dataclass(frozen=True)
class TokenSample:
    """Timing for one streaming request."""

    ok: bool
    ttft_s: float
    inter_token_s: tuple[float, ...]
    e2e_s: float
    output_tokens: int
    text: str
    error: str = ""


@dataclass(frozen=True)
class LoadResult:
    """Samples from one :func:`run_load` call."""

    samples: tuple[TokenSample, ...]
    wall_s: float
    concurrency: int
    endpoint_url: str
    model: str
    prompt: str


def latency_stats(values: Sequence[float], unit: str) -> dict[str, Any]:
    """Return an ohho.bench/v1 latency distribution.

    Percentiles use linear interpolation. ``median`` and ``p50`` match.
    ``std`` is the population standard deviation. An empty sample is all
    zeros with ``samples`` 0.
    """
    if not unit:
        raise LoadGenError("latency unit is required")
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count == 0:
        stats = {name: 0.0 for name in _STATS}
        stats["unit"] = unit
        stats["samples"] = 0
        return stats
    mean = sum(ordered) / count
    variance = sum((value - mean) ** 2 for value in ordered) / count
    stats = {
        "mean": mean,
        "median": _percentile(ordered, 50),
        "p50": _percentile(ordered, 50),
        "p90": _percentile(ordered, 90),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "min": ordered[0],
        "max": ordered[-1],
        "std": math.sqrt(variance),
        "unit": unit,
        "samples": count,
    }
    return stats


def run_load(spec: LoadSpec) -> LoadResult:
    """Send ``spec.requests`` streaming chat completions.

    At most ``spec.concurrency`` requests are in flight. Raises
    :class:`LoadGenError` when every request fails.
    """
    _check_spec(spec)
    workers = min(spec.concurrency, spec.requests)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, spec, index) for index in range(spec.requests)]
        samples = tuple(future.result() for future in futures)
    wall = time.perf_counter() - started
    if not any(sample.ok for sample in samples):
        detail = next((sample.error for sample in samples if sample.error), "no token")
        raise LoadGenError(
            f"every request failed ({len(samples)} attempts) against "
            f"{spec.endpoint_url}. {detail}"
        )
    return LoadResult(
        samples=samples,
        wall_s=wall if wall > 0 else 1e-9,
        concurrency=spec.concurrency,
        endpoint_url=spec.endpoint_url,
        model=spec.model,
        prompt=spec.prompt,
    )


def to_bench_result(
    result: LoadResult,
    *,
    backend: str,
    backend_version: str,
    precision: str,
    manifest_ref: str | None = None,
    accelerator: str = "none",
    target_profile: str = "",
    cost: CostOverrides | None = None,
) -> dict[str, Any]:
    """Build a validated ``ohho.bench/v1`` document with an LLM block.

    ``accelerator`` defaults to ``none``. This function does not probe GPUs
    and does not invent device measurements. Cost is an estimate from the
    measured token throughput and ``target_profile``'s price row.
    """
    from ..bench.cost import attach_cost
    from ..bench.schema import SCHEMA_ID, validate

    if not backend.strip():
        raise LoadGenError("backend is required")
    ok = [sample for sample in result.samples if sample.ok]
    if not ok:
        raise LoadGenError("bench result requires at least one successful request")
    wall = result.wall_s if result.wall_s > 0 else 1e-9
    output_tokens = sum(sample.output_tokens for sample in ok)
    tokens_per_s = output_tokens / wall
    requests_per_s = len(ok) / wall
    e2e_ms = [sample.e2e_s * 1000.0 for sample in ok]
    ttft_ms = [sample.ttft_s * 1000.0 for sample in ok]
    gaps_ms = [gap * 1000.0 for sample in ok for gap in sample.inter_token_s]
    model_sha, model_sha_of = _model_sha(result.model)
    artifact = "\n".join([backend, backend_version, precision, result.model])
    artifact_sha = "sha256:" + hashlib.sha256(artifact.encode("utf-8")).hexdigest()
    finished = datetime.now(timezone.utc)
    started = finished - timedelta(seconds=wall)
    attempted = len(result.samples)
    completion = len(ok) / attempted
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "ohho": _ohho_version(),
        backend: backend_version or "unknown",
    }
    document: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "run": {
            "id": f"llm-{backend}-{uuid.uuid4().hex[:8]}",
            "model_sha": model_sha,
            "artifact_sha": artifact_sha,
            "artifact_sha_of": "serving-setup",
            "target": "llm",
            "manifest_ref": manifest_ref or f"llm://{backend}",
            "host": _host(accelerator),
            "started_at": _stamp(started),
            "finished_at": _stamp(finished),
        },
        "latency": {
            **latency_stats(e2e_ms, "ms"),
            "stages": [
                {"name": "ttft", **latency_stats(ttft_ms, "ms")},
                {"name": "inter_token", **latency_stats(gaps_ms, "ms")},
            ],
        },
        "throughput": {
            "value": tokens_per_s,
            "unit": "tokens/s",
            "samples": output_tokens,
            "batch_size": result.concurrency,
            "duration_s": wall,
        },
        "accuracy": {
            "primary": {
                "name": "request_completion_rate",
                "value": completion,
                "unit": "fraction",
            },
            "metrics": [
                {
                    "name": "requests_per_s",
                    "value": requests_per_s,
                    "unit": "requests/s",
                },
                {
                    "name": "tokens_per_s",
                    "value": tokens_per_s,
                    "unit": "tokens/s",
                },
            ],
            "samples": attempted,
        },
        "telemetry": {"duration_s": wall, "samples": attempted},
        "environment": {
            "versions": versions,
            **(
                {"target_profile": target_profile.strip()}
                if isinstance(target_profile, str) and target_profile.strip()
                else {}
            ),
        },
        "llm": {
            "schema": LLM_BLOCK_ID,
            "backend": backend,
            "backend_version": backend_version or "unknown",
            "precision": precision,
            "model": result.model,
            "model_sha_of": model_sha_of,
            "prompt": result.prompt,
            "concurrency": result.concurrency,
            "ttft": latency_stats(ttft_ms, "ms"),
            "inter_token": latency_stats(gaps_ms, "ms"),
            "tokens_per_s": tokens_per_s,
            "requests_per_s": requests_per_s,
            "output_tokens": output_tokens,
            "requests": attempted,
            "requests_ok": len(ok),
            "endpoint_url": result.endpoint_url,
        },
    }
    attach_cost(
        document,
        cost,
        profile_id=target_profile.strip() if isinstance(target_profile, str) else "",
    )
    validate(document)
    return document


def _check_spec(spec: LoadSpec) -> None:
    if not str(spec.endpoint_url).strip():
        raise LoadGenError("endpoint_url is required")
    if not str(spec.model).strip():
        raise LoadGenError("model is required")
    if spec.concurrency < 1:
        raise LoadGenError("concurrency must be >= 1")
    if spec.requests < 1:
        raise LoadGenError("requests must be >= 1")
    if spec.max_tokens < 1:
        raise LoadGenError("max_tokens must be >= 1")
    if spec.timeout_s <= 0:
        raise LoadGenError("timeout_s must be positive")
    for prompt in spec.prompts:
        if not isinstance(prompt, str) or not prompt:
            raise LoadGenError("prompts entries must be non-empty strings")
    if spec.system_prompt and not isinstance(spec.system_prompt, str):
        raise LoadGenError("system_prompt must be a string")
    if spec.image_url and not isinstance(spec.image_url, str):
        raise LoadGenError("image_url must be a string")


def _one(spec: LoadSpec, index: int = 0) -> TokenSample:
    started = time.perf_counter()
    try:
        return _request(spec, started, index)
    except Exception as exc:
        return TokenSample(
            ok=False,
            ttft_s=0.0,
            inter_token_s=(),
            e2e_s=time.perf_counter() - started,
            output_tokens=0,
            text="",
            error=str(exc),
        )


def _request(spec: LoadSpec, started: float, index: int = 0) -> TokenSample:
    url = _chat_url(spec.endpoint_url)
    payload = {
        "model": spec.model,
        "messages": _messages(spec, index),
        "stream": True,
        "max_tokens": spec.max_tokens,
        "temperature": 0,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=spec.timeout_s)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise LoadGenError(f"HTTP {exc.code} from {url}. {detail}") from exc
    except urllib.error.URLError as exc:
        raise LoadGenError(f"could not reach {url}: {exc.reason}") from exc
    ttft: float | None = None
    gaps: list[float] = []
    chunks: list[str] = []
    last = started
    with response:
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            if body == "[DONE]":
                break
            try:
                event = json.loads(body)
            except json.JSONDecodeError as exc:
                raise LoadGenError(f"invalid streaming chunk: {exc}") from exc
            text = _chunk_text(event)
            if not text:
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - started
            else:
                gaps.append(now - last)
            last = now
            chunks.append(text)
    if ttft is None:
        raise LoadGenError(f"stream from {url} ended without a token")
    return TokenSample(
        ok=True,
        ttft_s=ttft,
        inter_token_s=tuple(gaps),
        e2e_s=time.perf_counter() - started,
        output_tokens=len(chunks),
        text="".join(chunks),
    )


def _prompt_at(spec: LoadSpec, index: int) -> str:
    if spec.prompts:
        return spec.prompts[index % len(spec.prompts)]
    return spec.prompt


def _messages(spec: LoadSpec, index: int) -> list[dict[str, Any]]:
    """Chat messages for one request.

    A system prompt is shared across the sweep's prefix-cache workload.
    An image URL is sent as an OpenAI image part for the long-context
    vision workload. The server decides whether it can read that part.
    """
    prompt = _prompt_at(spec, index)
    messages: list[dict[str, Any]] = []
    if spec.system_prompt:
        messages.append({"role": "system", "content": spec.system_prompt})
    if spec.image_url:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": spec.image_url}},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def _chat_url(endpoint: str) -> str:
    base = endpoint.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _chunk_text(event: object) -> str:
    if not isinstance(event, Mapping):
        return ""
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return ""
    delta = choice.get("delta")
    if isinstance(delta, Mapping):
        content = delta.get("content")
        if isinstance(content, str) and content:
            return content
    text = choice.get("text")
    if isinstance(text, str) and text:
        return text
    return ""


def _percentile(ordered: Sequence[float], pct: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _model_sha(model: str) -> tuple[str, str]:
    path = Path(model)
    try:
        if path.is_file() and path.stat().st_size <= _MAX_HASH_BYTES:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return f"sha256:{digest}", "file"
    except OSError:
        pass
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()
    return f"sha256:{digest}", "identifier"


def _host(accelerator: str) -> dict[str, str]:
    cpu = platform.processor() or platform.machine() or "unknown"
    return {
        "name": platform.node() or "localhost",
        "platform": platform.system().lower() or "unknown",
        "arch": platform.machine() or "unknown",
        "cpu": cpu,
        "accelerator": accelerator or "none",
    }


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ohho_version() -> str:
    from .. import __version__

    return __version__
