"""Generate-mode benchmark for autoregressive models.

``ohho bench --mode generate`` measures time to first token (TTFT), time per
output token (TPOT), end-to-end latency, and tokens per second. A run sweeps
concurrency, prompt length, and, for ``vlm`` and ``vla``, image size. It also
compares FP16 / BF16 with bitsandbytes 4-bit, AWQ, and GPTQ.

The default backend is a deterministic stand-in that runs on CPU with no
optional packages. ``transformers`` and the external engines from
``ohho.serve.backends`` are used only when the caller selects them and the
target can run them. Those imports stay inside the functions that need them.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ohho.bench.agreement import top1_action_agreement
from ohho.bench.clock import VirtualClock
from ohho.bench.environment import (
    MemoryTrace,
    collect_environment,
    gpu_telemetry,
    host_block,
)
from ohho.bench.errors import BenchHarnessError
from ohho.bench.gates import check_gate_expressions, evaluate_gates
from ohho.bench.schema import SCHEMA_ID, BenchValidationError, validate
from ohho.bench.stats import summarize
from ohho.targets.errors import TargetProfileError
from ohho.targets.schema import TargetProfile, get_profile

LLM_BLOCK_ID = "ohho.bench.llm/v1"
WORKLOADS = ("llm", "vlm", "vla")
REFERENCE_PRECISIONS = frozenset({"fp16", "bf16", "fp32"})
PRECISIONS = ("fp16", "bf16", "fp32", "bnb-int4", "awq", "gptq")
_FOUR_BIT = frozenset({"bnb-int4", "awq", "gptq"})
_GPU_BACKENDS = ("vllm", "sglang", "tensorrt-llm")
_LOCAL_BACKENDS = ("stand-in", "transformers")
_PRECISION_ALIASES = {
    "float16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "int4": "bnb-int4",
    "bitsandbytes": "bnb-int4",
    "bitsandbytes-int4": "bnb-int4",
    "int4-bnb": "bnb-int4",
    "int4-awq": "awq",
    "int4-gptq": "gptq",
}
_BACKEND_ALIASES = {
    "standin": "stand-in",
    "hf": "transformers",
    "huggingface": "transformers",
    "tensorrt": "tensorrt-llm",
    "tensorrt_llm": "tensorrt-llm",
    "trtllm": "tensorrt-llm",
}
_TTFT_FACTOR = {
    "fp32": 1.15,
    "fp16": 1.0,
    "bf16": 1.0,
    "bnb-int4": 1.08,
    "awq": 1.04,
    "gptq": 1.05,
}
_TPOT_FACTOR = {
    "fp32": 1.40,
    "fp16": 1.0,
    "bf16": 1.02,
    "bnb-int4": 0.70,
    "awq": 0.62,
    "gptq": 0.66,
}
_BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "bnb-int4": 0.5,
    "awq": 0.5,
    "gptq": 0.5,
}
_QUANT_SALT = {"bnb-int4": 1, "awq": 3, "gptq": 5}
_PARAMS = 1_048_576
_VOCAB = 128
_STAT_KEYS = ("mean", "median", "p50", "p90", "p95", "p99", "min", "max", "std")


@dataclass(frozen=True)
class GenerateConfig:
    """One generate-mode sweep. Gates are applied by :func:`execute_generate`."""

    target: str
    workload: str = "llm"
    backend: str = "stand-in"
    model: str = "stand-in"
    precisions: tuple[str, ...] = ("fp16", "bnb-int4")
    concurrencies: tuple[int, ...] = (1, 4, 16)
    prompt_tokens: tuple[int, ...] = (16, 64)
    image_sizes: tuple[int, ...] = ()
    max_tokens: int = 8
    requests: int = 4
    warmup: int = 0

    def __post_init__(self) -> None:
        target = _text(self.target, "--target")
        workload = _text(self.workload, "--workload").lower()
        if workload not in WORKLOADS:
            known = ", ".join(WORKLOADS)
            raise BenchHarnessError(
                f"--workload must be one of {known} for --mode generate"
            )
        backend = canonical_backend(self.backend)
        model = self.model.strip() if isinstance(self.model, str) else ""
        if not model:
            raise BenchHarnessError("--model must be a non-empty string")
        if backend != "stand-in" and model == "stand-in":
            raise BenchHarnessError(f"--model is required for backend {backend}")
        precisions = tuple(canonical_precision(item) for item in self.precisions)
        if not precisions:
            raise BenchHarnessError("--precision needs at least one value")
        concurrencies = _positive_ints(self.concurrencies, "--concurrency")
        prompts = _positive_ints(self.prompt_tokens, "--prompt-tokens")
        if self.image_sizes:
            images = _non_negative_ints(self.image_sizes, "--image-sizes")
        elif workload == "llm":
            images = (0,)
        else:
            images = (224,)
        if workload == "llm" and any(size > 0 for size in images):
            raise BenchHarnessError(
                "llm generate mode has no image; use --workload vlm or vla "
                "for an image-size sweep"
            )
        _check_count(self.max_tokens, "--max-tokens", minimum=1)
        _check_count(self.requests, "--requests", minimum=1)
        _check_count(self.warmup, "--warmup", minimum=0)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "workload", workload)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "precisions", precisions)
        object.__setattr__(self, "concurrencies", concurrencies)
        object.__setattr__(self, "prompt_tokens", prompts)
        object.__setattr__(self, "image_sizes", images)
        object.__setattr__(self, "max_tokens", int(self.max_tokens))
        object.__setattr__(self, "requests", int(self.requests))
        object.__setattr__(self, "warmup", int(self.warmup))


@dataclass(frozen=True)
class MeasureRequest:
    """One sweep cell sent to a generator."""

    precision: str
    prompt_tokens: int
    image_tokens: int
    image_size: int
    concurrency: int
    requests: int
    max_tokens: int


@dataclass(frozen=True)
class RequestSample:
    """Timing and tokens for one generated request."""

    ttft_s: float
    tpot_s: tuple[float, ...]
    e2e_s: float
    tokens: tuple[int, ...]
    output_tokens: int | None = None


@dataclass(frozen=True)
class PointMeasurement:
    """Samples for one sweep cell, plus the wall time that produced them."""

    samples: tuple[RequestSample, ...]
    wall_s: float
    peak_vram_mb: float | None
    vram_source: str
    requests_total: int | None = None


def canonical_precision(value: str) -> str:
    """Return the bench id for a precision name."""
    if not isinstance(value, str) or not value.strip():
        raise BenchHarnessError("--precision values must be non-empty strings")
    key = value.strip().lower().replace("_", "-")
    name = _PRECISION_ALIASES.get(key, key)
    if name not in PRECISIONS:
        known = ", ".join(PRECISIONS)
        raise BenchHarnessError(
            f"unknown precision {value!r}. Expected one of: {known}."
        )
    return name


def canonical_backend(value: str) -> str:
    """Return the bench id for a generate backend name."""
    if not isinstance(value, str) or not value.strip():
        raise BenchHarnessError("--backend must be a non-empty string")
    key = value.strip().lower().replace("_", "-")
    name = _BACKEND_ALIASES.get(key, key)
    known = (*_LOCAL_BACKENDS, *_GPU_BACKENDS)
    if name not in known:
        listed = ", ".join(known)
        raise BenchHarnessError(
            f"unknown generate backend {value!r}. Expected one of: {listed}."
        )
    return name


def supported_backends(profile: TargetProfile) -> tuple[str, ...]:
    """Backends this hardware profile can run.

    The stand-in and the transformers path are available on every profile.
    vLLM, SGLang, and TensorRT-LLM need an accelerator of ``dgpu`` or
    ``orin_gpu``. Whether the program is installed is checked when that
    backend is opened.
    """
    names = list(_LOCAL_BACKENDS)
    if profile.accelerator in {"dgpu", "orin_gpu"}:
        names.extend(_GPU_BACKENDS)
    return tuple(names)


def image_token_count(image_size: int) -> int:
    """Patch count for a square image. A side of 0 is a text-only request."""
    if image_size <= 0:
        return 0
    patches = max(1, int(image_size) // 16)
    return patches * patches


def weight_nbytes(precision: str) -> int:
    """Byte size of the stand-in weight buffer for ``precision``."""
    return max(1, int(_PARAMS * _BYTES_PER_PARAM[precision]))


def execute_generate(
    config: GenerateConfig,
    *,
    slos: Sequence[str] = (),
    generator: Any = None,
) -> dict[str, Any]:
    """Run ``config`` and return a validated ``ohho.bench/v1`` document.

    ``generator`` replaces the backend selected by ``config``. Tests use it
    to avoid launching an external engine. Gate expressions are checked
    before the first request.
    """
    check_gate_expressions(tuple(slos), (), has_baseline=False)
    try:
        profile = get_profile(config.target)
    except TargetProfileError as exc:
        raise BenchHarnessError(str(exc)) from exc
    if config.backend not in supported_backends(profile):
        allowed = ", ".join(supported_backends(profile))
        raise BenchHarnessError(
            f"backend {config.backend} is not supported on target {profile.id} "
            f"(accelerator {profile.accelerator}). Supported backends: {allowed}."
        )
    owns_generator = generator is None
    if generator is None:
        generator = open_generator(config)
    trace = MemoryTrace()
    trace.start()
    try:
        points, skipped_precisions = _sweep(config, generator)
    finally:
        if owns_generator:
            close = getattr(generator, "close", None)
            if callable(close):
                close()
        memory = trace.snapshot()
        trace.close()
    document = _document(
        config,
        profile,
        points,
        memory.as_telemetry(),
        skipped_precisions=skipped_precisions,
    )
    if slos:
        document["gates"] = evaluate_gates(document, tuple(slos), ())
    _require_valid(document)
    return document


def open_generator(config: GenerateConfig) -> Any:
    """Build the generator named by ``config.backend``."""
    if config.backend == "stand-in":
        return StandInGenerator()
    if config.backend == "transformers":
        return TransformersGenerator(config.model)
    return ServingGenerator(config.backend, config.model)


class StandInGenerator:
    """Deterministic autoregressive stand-in.

    Service time is a virtual clock. TTFT grows with prompt tokens, image
    tokens, and concurrency. TPOT grows with concurrency and is lower for
    4-bit precisions. Aggregate tokens/s still rises with concurrency because
    each point runs at least as many requests as its concurrency.

    FP16, BF16, and FP32 share greedy action tokens. 4-bit precisions
    disagree on every fourth token. Peak memory is the size of the weight
    buffer this object allocates, not a GPU probe.
    """

    name = "stand-in"

    def __init__(self) -> None:
        self._buffers: dict[str, bytearray] = {}

    def measure(self, request: MeasureRequest) -> PointMeasurement:
        count = max(request.requests, request.concurrency)
        samples = [
            _one_standin(request, index, VirtualClock()) for index in range(count)
        ]
        durations = [sample.e2e_s for sample in samples]
        nbytes = self._allocate(request.precision)
        return PointMeasurement(
            samples=tuple(samples),
            wall_s=_pool_wall(durations, request.concurrency),
            peak_vram_mb=nbytes / (1024 * 1024),
            vram_source="stand-in-weights",
        )

    def close(self) -> None:
        self._buffers.clear()

    def _allocate(self, precision: str) -> int:
        buffer = self._buffers.get(precision)
        if buffer is None:
            buffer = bytearray(weight_nbytes(precision))
            buffer[0] = 1
            buffer[-1] = 1
            self._buffers[precision] = buffer
        return len(buffer)


class TransformersGenerator:
    """Hugging Face causal LM. Imported only when this backend is opened.

    4-bit bitsandbytes, AWQ, and GPTQ need a CUDA device. Image sizes add
    prefill positions; this path does not decode pixels.
    """

    name = "transformers"

    def __init__(self, model: str) -> None:
        torch, transformers = _load_transformers()
        self._torch = torch
        self._model_id = model
        self._loaded_precision: str | None = None
        self._loaded_model: Any = None
        self._tokenizer: Any = None
        self._transformers = transformers
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def _cache(self) -> dict[str, tuple[Any, Any]]:
        """Active precision model for callers inspecting cache state."""
        if self._loaded_precision and self._loaded_model is not None:
            return {self._loaded_precision: (self._loaded_model, self._tokenizer)}
        return {}

    def measure(self, request: MeasureRequest) -> PointMeasurement:
        model, tokenizer = self._load(request.precision)
        prompt = _prompt_ids(
            tokenizer,
            self._torch,
            request.prompt_tokens,
            request.image_tokens,
            self._device,
        )
        if self._device == "cuda":
            self._torch.cuda.reset_peak_memory_stats()
        count = max(request.requests, request.concurrency)
        batch_size = max(1, request.concurrency)
        started = time.perf_counter()
        samples: list[RequestSample] = []
        remaining = count
        while remaining > 0:
            current_batch = min(remaining, batch_size)
            batch_samples = _transformers_batch_sample(
                model,
                self._torch,
                prompt,
                request.max_tokens,
                self._device,
                current_batch,
            )
            samples.extend(batch_samples)
            remaining -= current_batch
        wall = time.perf_counter() - started
        peak: float | None = None
        source = "unavailable"
        if self._device == "cuda":
            peak = float(self._torch.cuda.max_memory_allocated()) / (1024 * 1024)
            source = "cuda"
        return PointMeasurement(
            samples=tuple(samples),
            wall_s=wall if wall > 0 else 1e-9,
            peak_vram_mb=peak,
            vram_source=source,
            requests_total=count,
        )

    def close(self) -> None:
        self._loaded_model = None
        self._loaded_precision = None
        self._tokenizer = None
        if self._device == "cuda":
            self._torch.cuda.empty_cache()

    def _load(self, precision: str) -> tuple[Any, Any]:
        if precision in _FOUR_BIT and self._device != "cuda":
            raise BenchHarnessError(
                f"{precision} needs a CUDA device, and this process has none."
            )
        if self._tokenizer is None:
            self._tokenizer = self._transformers.AutoTokenizer.from_pretrained(
                self._model_id
            )
        if self._loaded_precision == precision and self._loaded_model is not None:
            return self._loaded_model, self._tokenizer

        # Precision is the outer loop: evict previous model before loading next
        self._loaded_model = None
        self._loaded_precision = None
        if self._device == "cuda":
            self._torch.cuda.empty_cache()

        kwargs: dict[str, Any] = {}
        torch = self._torch
        transformers = self._transformers
        if precision == "bnb-int4":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise BenchHarnessError(
                    "bitsandbytes 4-bit needs the bitsandbytes package, "
                    "which is not installed."
                ) from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        elif precision == "awq":
            import importlib.util

            if (
                importlib.util.find_spec("autoawq") is None
                and importlib.util.find_spec("awq") is None
            ):
                raise BenchHarnessError(
                    "AWQ needs the autoawq package, which is not installed."
                )
            try:
                from transformers import AwqConfig

                kwargs["quantization_config"] = AwqConfig(bits=4)
            except (ImportError, AttributeError, ValueError) as exc:
                raise BenchHarnessError(
                    f"AWQ quantization configuration failed for {self._model_id}: {exc}"
                ) from exc
            kwargs["device_map"] = "auto"
        elif precision == "gptq":
            import importlib.util

            if (
                importlib.util.find_spec("auto_gptq") is None
                and importlib.util.find_spec("gptqmodel") is None
            ):
                raise BenchHarnessError(
                    "GPTQ needs the auto-gptq package, which is not installed."
                )
            try:
                from transformers import GPTQConfig

                kwargs["quantization_config"] = GPTQConfig(bits=4)
            except (ImportError, AttributeError, ValueError) as exc:
                raise BenchHarnessError(
                    f"GPTQ quantization configuration failed for {self._model_id}: {exc}"
                ) from exc
            kwargs["device_map"] = "auto"
        else:
            dtype = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }[precision]
            kwargs["torch_dtype"] = dtype
            if self._device == "cuda":
                kwargs["device_map"] = "auto"
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self._model_id, **kwargs
            )
        except Exception as exc:
            raise BenchHarnessError(
                f"could not load {self._model_id} at {precision}: {exc}"
            ) from exc
        if "device_map" not in kwargs:
            model = model.to(self._device)
        model.eval()
        self._loaded_model = model
        self._loaded_precision = precision
        return model, self._tokenizer


class ServingGenerator:
    """Measure a vLLM, SGLang, or TensorRT-LLM server through the load generator.

    The engine is an external program. Launch and teardown use the adapters
    in ``ohho.serve.backends``. Image tokens lengthen the text prompt; the
    server is not sent pixels.
    """

    SUPPORTED_PRECISIONS = frozenset({"fp16", "bf16", "fp32"})

    def __init__(self, backend: str, model: str) -> None:
        self.name = backend
        self._model = model
        self._backend: Any = None
        self._precision: str | None = None

    def supports_precision(self, precision: str) -> bool:
        if precision in _FOUR_BIT:
            return False
        from ohho.serve.backends import ServingBackendError, open_backend

        try:
            backend = self._backend or open_backend(self.name)
            backend.plan(self._model, precision, {})
            return True
        except ServingBackendError:
            return False

    def measure(self, request: MeasureRequest) -> PointMeasurement:
        if not self.supports_precision(request.precision):
            raise BenchHarnessError(
                f"backend {self.name!r} does not support precision {request.precision!r}; "
                "serving backends do not support 4-bit precisions (bnb-int4, awq, gptq)"
            )
        endpoint = self._ensure(request.precision)
        words = request.prompt_tokens + request.image_tokens
        prompt = " ".join(["token"] * max(1, words))
        from ohho.serve.loadgen import LoadGenError, LoadSpec, run_load

        spec = LoadSpec(
            endpoint_url=endpoint,
            model=self._model,
            prompt=prompt,
            max_tokens=request.max_tokens,
            concurrency=request.concurrency,
            requests=max(request.requests, request.concurrency),
            timeout_s=120.0,
        )
        try:
            result = run_load(spec)
        except LoadGenError as exc:
            raise BenchHarnessError(str(exc)) from exc
        return PointMeasurement(
            samples=samples_from_load(result),
            wall_s=result.wall_s,
            peak_vram_mb=None,
            vram_source="unavailable",
            requests_total=len(result.samples),
        )

    def close(self) -> None:
        if self._backend is not None:
            self._backend.teardown()
            self._backend = None
            self._precision = None

    def _ensure(self, precision: str) -> str:
        from ohho.serve.backends import ServingBackendError, open_backend

        if self._backend is None:
            self._backend = open_backend(self.name)
        if self._precision != precision:
            if getattr(self._backend, "running", False):
                self._backend.teardown()
            try:
                self._backend.launch(self._model, precision, {})
            except ServingBackendError as exc:
                raise BenchHarnessError(str(exc)) from exc
            self._precision = precision
        return str(self._backend.endpoint_url())


def samples_from_load(result: Any) -> tuple[RequestSample, ...]:
    """Map a load-generator result onto generate-mode samples."""
    samples: list[RequestSample] = []
    for sample in result.samples:
        if not getattr(sample, "ok", False):
            continue
        chunks = getattr(sample, "chunks", None)
        if chunks:
            tokens = tuple(
                int.from_bytes(hashlib.sha256(c.encode("utf-8")).digest()[:2], "big") % _VOCAB
                for c in chunks
            )
        else:
            words = _ids_from_text(str(sample.text))
            count_val = int(getattr(sample, "output_tokens", 0) or len(words))
            if len(words) == count_val:
                tokens = words
            elif len(words) > 0:
                tokens = tuple(words[i % len(words)] for i in range(count_val))
            else:
                tokens = tuple((i % _VOCAB) for i in range(count_val))
        count = int(getattr(sample, "output_tokens", 0) or len(tokens))
        samples.append(
            RequestSample(
                ttft_s=float(sample.ttft_s),
                tpot_s=tuple(float(gap) for gap in sample.inter_token_s),
                e2e_s=float(sample.e2e_s),
                tokens=tokens,
                output_tokens=count,
            )
        )
    if not samples:
        raise BenchHarnessError("serving backend returned no successful request")
    return tuple(samples)


def run_decode(
    step: Callable[[], int],
    max_tokens: int,
    clock: Any,
) -> RequestSample:
    """Time ``max_tokens`` greedy steps.

    The first ``step`` call is the prefill. Its duration is TTFT. Each later
    call is one output token. ``step`` may advance ``clock`` itself.
    """
    if max_tokens < 1:
        raise BenchHarnessError("--max-tokens must be a positive integer")
    started = float(clock.now())
    tokens = [int(step())]
    ttft = float(clock.now()) - started
    gaps: list[float] = []
    last = float(clock.now())
    for _ in range(max_tokens - 1):
        tokens.append(int(step()))
        now = float(clock.now())
        gaps.append(max(0.0, now - last))
        last = now
    e2e = float(clock.now()) - started
    return RequestSample(
        ttft_s=max(0.0, ttft),
        tpot_s=tuple(gaps),
        e2e_s=max(0.0, e2e),
        tokens=tuple(tokens),
    )


def reference_tokens(
    prompt_tokens: int,
    image_tokens: int,
    count: int,
) -> tuple[int, ...]:
    """FP16 greedy tokens for one prompt. 4-bit agreement is scored against these."""
    return tuple(
        _action_token("fp16", prompt_tokens, image_tokens, step)
        for step in range(count)
    )


def _sweep(
    config: GenerateConfig, generator: Any
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    points: list[dict[str, Any]] = []
    skipped_precisions: list[dict[str, str]] = []

    supported_precisions: list[str] = []
    for precision in config.precisions:
        if hasattr(generator, "supports_precision") and not generator.supports_precision(precision):
            skipped_precisions.append(
                {
                    "precision": precision,
                    "reason": (
                        f"backend {getattr(generator, 'name', config.backend)!r} "
                        f"does not support precision {precision!r}"
                    ),
                }
            )
        else:
            supported_precisions.append(precision)

    if not supported_precisions:
        reasons = "; ".join(f"{s['precision']}: {s['reason']}" for s in skipped_precisions)
        raise BenchHarnessError(
            f"backend {config.backend!r} cannot run any requested precision. {reasons}"
        )

    ref_precision = _primary_precision(supported_precisions)
    precisions_order = list(supported_precisions)
    if ref_precision in precisions_order:
        precisions_order.remove(ref_precision)
        precisions_order.insert(0, ref_precision)

    reference_map: dict[tuple[int, int], tuple[int, ...]] = {}

    for precision in precisions_order:
        for image_size in config.image_sizes:
            image_tokens = image_token_count(image_size)
            for prompt in config.prompt_tokens:
                ref_key = (prompt, image_tokens)
                for concurrency in config.concurrencies:
                    request = MeasureRequest(
                        precision=precision,
                        prompt_tokens=prompt,
                        image_tokens=image_tokens,
                        image_size=image_size,
                        concurrency=concurrency,
                        requests=config.requests,
                        max_tokens=config.max_tokens,
                    )
                    if config.warmup:
                        generator.measure(
                            MeasureRequest(
                                precision=precision,
                                prompt_tokens=prompt,
                                image_tokens=image_tokens,
                                image_size=image_size,
                                concurrency=1,
                                requests=config.warmup,
                                max_tokens=config.max_tokens,
                            )
                        )
                    measured = generator.measure(request)
                    if (
                        precision == ref_precision
                        and ref_key not in reference_map
                        and measured.samples
                    ):
                        reference_map[ref_key] = measured.samples[0].tokens

                    ref_tokens = reference_map.get(ref_key)
                    points.append(
                        _point_row(config, request, measured, reference=ref_tokens)
                    )
    if not points:
        raise BenchHarnessError("generate mode produced no measurements")
    return points, skipped_precisions


def _point_row(
    config: GenerateConfig,
    request: MeasureRequest,
    measured: PointMeasurement,
    reference: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    samples = measured.samples
    if not samples:
        raise BenchHarnessError("generate backend returned no samples")
    ttft = distribution([sample.ttft_s for sample in samples])
    gaps = [gap for sample in samples for gap in sample.tpot_s]
    tpot = distribution(gaps)
    e2e = distribution([sample.e2e_s for sample in samples])
    output_tokens = sum(
        sample.output_tokens if sample.output_tokens is not None else len(sample.tokens)
        for sample in samples
    )
    wall = measured.wall_s if measured.wall_s > 0 else 1e-9
    per_request = [
        (sample.output_tokens if sample.output_tokens is not None else len(sample.tokens))
        / sample.e2e_s
        for sample in samples
        if sample.e2e_s > 0
    ]
    tokens_per_request_s = sum(per_request) / len(per_request) if per_request else 0.0
    if reference is None:
        reference = reference_tokens(
            request.prompt_tokens, request.image_tokens, config.max_tokens
        )
    scores = [top1_action_agreement(reference, sample.tokens) for sample in samples]
    agreement = sum(scores) / len(scores)
    requests_total = (
        measured.requests_total
        if measured.requests_total is not None
        else len(samples)
    )
    requests_ok = len(samples)
    row: dict[str, Any] = {
        "concurrency": request.concurrency,
        "prompt_tokens": request.prompt_tokens,
        "image_tokens": request.image_tokens,
        "image_size": request.image_size,
        "precision": request.precision,
        "ttft": ttft,
        "tpot": tpot,
        "e2e": e2e,
        "tokens_per_s": output_tokens / wall,
        "tokens_per_request_s": tokens_per_request_s,
        "requests_per_s": requests_ok / wall,
        "requests": requests_total,
        "requests_ok": requests_ok,
        "output_tokens": output_tokens,
        "agreement": _agreement_metric(agreement, config.max_tokens),
    }
    if measured.peak_vram_mb is not None:
        row["peak_vram_mb"] = measured.peak_vram_mb
        row["vram_source"] = measured.vram_source
    elif measured.vram_source:
        row["vram_source"] = measured.vram_source
    return row


def _document(
    config: GenerateConfig,
    profile: TargetProfile,
    points: list[dict[str, Any]],
    memory: dict[str, Any],
    skipped_precisions: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    primary = _primary_point(config, points)
    precision_rows = _precision_rows(config, points)
    virtual_s = _virtual_seconds(points)
    finished = datetime.now(timezone.utc)
    started = finished - timedelta(seconds=max(virtual_s, 0.0))
    versions = _version_note(config)
    timing_source, timing_note, provider = _timing(config)
    environment, gpu = collect_environment(
        profile,
        timing_source=timing_source,
        execution_provider=provider,
        timing_note=timing_note,
    )
    environment["versions"].update(versions)
    identity = _identity(config)
    digest = "sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
    mean_s = float(primary["e2e"]["mean"]) / 1000.0
    latency = {
        **dict(primary["e2e"]),
        "stages": [
            {"name": "ttft", **dict(primary["ttft"])},
            {"name": "tpot", **dict(primary["tpot"])},
        ],
        "achievable_hz": 0.0 if mean_s == 0 else 1.0 / mean_s,
    }
    telemetry: dict[str, Any] = {
        "duration_s": virtual_s,
        "samples": int(primary["requests"]),
    }
    telemetry.update(memory)
    telemetry.update(gpu_telemetry(gpu))
    if skipped_precisions:
        telemetry["skipped_precisions"] = [
            item["precision"] for item in skipped_precisions
        ]
    llm: dict[str, Any] = {
        "schema": LLM_BLOCK_ID,
        "backend": config.backend,
        "backend_version": versions.get(config.backend, "1"),
        "precision": primary["precision"],
        "model": config.model,
        "model_sha_of": "identifier",
        "prompt": f"prompt_tokens={primary['prompt_tokens']}",
        "workload": config.workload,
        "concurrency": primary["concurrency"],
        "ttft": primary["ttft"],
        "inter_token": primary["tpot"],
        "tpot": primary["tpot"],
        "e2e": primary["e2e"],
        "tokens_per_s": primary["tokens_per_s"],
        "tokens_per_request_s": primary["tokens_per_request_s"],
        "requests_per_s": primary["requests_per_s"],
        "prompt_tokens": primary["prompt_tokens"],
        "image_tokens": primary["image_tokens"],
        "output_tokens": primary["output_tokens"],
        "requests": primary["requests"],
        "requests_ok": primary.get("requests_ok", primary["requests"]),
        "agreement": primary["agreement"],
        "points": points,
        "precisions": precision_rows,
    }
    if skipped_precisions:
        llm["skipped_precisions"] = skipped_precisions
    if "peak_vram_mb" in primary:
        llm["peak_vram_mb"] = primary["peak_vram_mb"]
        llm["vram_source"] = primary.get("vram_source", "")
    document: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "run": {
            "id": f"gen-{uuid.uuid4().hex[:12]}",
            "model_sha": digest,
            "artifact_sha": digest,
            "target": config.workload,
            "manifest_ref": f"generate://{config.backend}/{config.workload}",
            "host": host_block(profile),
            "started_at": _stamp(started),
            "finished_at": _stamp(finished),
        },
        "latency": latency,
        "throughput": {
            "value": primary["tokens_per_s"],
            "unit": "tokens/s",
            "samples": primary["output_tokens"],
            "batch_size": primary["concurrency"],
            "duration_s": primary["output_tokens"] / primary["tokens_per_s"]
            if primary["tokens_per_s"]
            else 0.0,
        },
        "accuracy": {
            "primary": primary["agreement"],
            "metrics": [
                row["agreement"] for row in precision_rows if "agreement" in row
            ],
            "samples": config.max_tokens,
        },
        "telemetry": telemetry,
        "environment": environment,
        "llm": llm,
    }
    return document


def _primary_point(
    config: GenerateConfig, points: list[dict[str, Any]]
) -> dict[str, Any]:
    concurrency = 1 if 1 in config.concurrencies else min(config.concurrencies)
    prompt = min(config.prompt_tokens)
    image = min(config.image_sizes)
    precision = _primary_precision(config.precisions)
    for point in points:
        if (
            point["concurrency"] == concurrency
            and point["prompt_tokens"] == prompt
            and point["image_size"] == image
            and point["precision"] == precision
        ):
            return point
    return points[0]


def _primary_precision(precisions: Sequence[str]) -> str:
    for name in ("fp16", "bf16", "fp32"):
        if name in precisions:
            return name
    return precisions[0]


def _precision_rows(
    config: GenerateConfig, points: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    primary = _primary_point(config, points)
    rows: list[dict[str, Any]] = []
    for precision in config.precisions:
        match = next(
            (
                point
                for point in points
                if point["precision"] == precision
                and point["concurrency"] == primary["concurrency"]
                and point["prompt_tokens"] == primary["prompt_tokens"]
                and point["image_size"] == primary["image_size"]
            ),
            None,
        )
        if match is None:
            continue
        row = {
            "precision": precision,
            "ttft": match["ttft"],
            "tpot": match["tpot"],
            "e2e": match["e2e"],
            "tokens_per_s": match["tokens_per_s"],
            "tokens_per_request_s": match["tokens_per_request_s"],
            "concurrency": match["concurrency"],
            "prompt_tokens": match["prompt_tokens"],
            "image_tokens": match["image_tokens"],
            "agreement": match["agreement"],
        }
        if "peak_vram_mb" in match:
            row["peak_vram_mb"] = match["peak_vram_mb"]
            row["vram_source"] = match.get("vram_source", "")
        rows.append(row)
    return rows


def _one_standin(
    request: MeasureRequest, index: int, clock: VirtualClock
) -> RequestSample:
    state = {"step": 0}

    def step() -> int:
        step_index = state["step"]
        if step_index == 0:
            cost = _prefill_s(request, index)
        else:
            cost = _tpot_s(request, index, step_index)
        clock.advance(cost)
        token = _action_token(
            request.precision,
            request.prompt_tokens,
            request.image_tokens,
            step_index,
        )
        state["step"] = step_index + 1
        return token

    return run_decode(step, request.max_tokens, clock)


def _prefill_s(request: MeasureRequest, index: int) -> float:
    tokens = max(1, request.prompt_tokens + request.image_tokens)
    return (
        tokens
        * 2.0e-5
        * _TTFT_FACTOR[request.precision]
        * _contention(request.concurrency)
        * _jitter(index)
    )


def _tpot_s(request: MeasureRequest, index: int, step: int) -> float:
    return (
        5.0e-4
        * _TPOT_FACTOR[request.precision]
        * _contention(request.concurrency)
        * _jitter(index + step)
    )


def _contention(concurrency: int) -> float:
    return 1.0 + 0.25 * (concurrency - 1)


def _jitter(index: int) -> float:
    return 1.0 + 0.01 * (index % 7)


def _action_token(
    precision: str, prompt_tokens: int, image_tokens: int, step: int
) -> int:
    base = (prompt_tokens * 3 + image_tokens * 5 + step * 7 + 11) % _VOCAB
    if precision in REFERENCE_PRECISIONS or step % 4 != 0:
        return base
    return (base + _QUANT_SALT[precision]) % _VOCAB


def _pool_wall(durations: Sequence[float], concurrency: int) -> float:
    workers = [0.0] * max(1, concurrency)
    for duration in durations:
        slot = min(range(len(workers)), key=lambda index: workers[index])
        workers[slot] += duration
    return max(workers) if workers else 0.0


def distribution(samples_s: Sequence[float]) -> dict[str, Any]:
    """Latency distribution in milliseconds. An empty sample is all zeros."""
    if not samples_s:
        stats = {key: 0.0 for key in _STAT_KEYS}
    else:
        stats = summarize([max(0.0, float(value) * 1000.0) for value in samples_s])
    stats["unit"] = "ms"
    stats["samples"] = len(samples_s)
    return stats


def _agreement_metric(value: float, samples: int) -> dict[str, Any]:
    return {
        "name": "top1_action_token",
        "value": float(value),
        "unit": "fraction",
        "samples": samples,
    }


def _ids_from_text(text: str) -> tuple[int, ...]:
    parts = [part for part in text.split() if part] or ([text] if text else [])
    ids = []
    for part in parts:
        digest = hashlib.sha256(part.encode("utf-8")).digest()
        ids.append(int.from_bytes(digest[:2], "big") % _VOCAB)
    return tuple(ids)


def _transformers_batch_sample(
    model: Any,
    torch: Any,
    prompt: Any,
    max_tokens: int,
    device: str,
    batch_size: int = 1,
) -> tuple[RequestSample, ...]:
    if max_tokens < 1:
        raise BenchHarnessError("--max-tokens must be a positive integer")
    batched_prompt = prompt.repeat(batch_size, 1) if batch_size > 1 else prompt
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(batched_prompt, use_cache=True)
    past_value = output.past_key_values
    ttft = max(0.0, time.perf_counter() - started)
    next_tokens = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens_per_item: list[list[int]] = [
        [int(next_tokens[i, 0].item())] for i in range(batch_size)
    ]
    gaps: list[float] = []
    last = time.perf_counter()
    for _ in range(max_tokens - 1):
        with torch.inference_mode():
            output = model(next_tokens, use_cache=True, past_key_values=past_value)
        past_value = output.past_key_values
        next_tokens = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        now = time.perf_counter()
        gaps.append(max(0.0, now - last))
        last = now
        for i in range(batch_size):
            tokens_per_item[i].append(int(next_tokens[i, 0].item()))
    e2e = max(0.0, time.perf_counter() - started)
    return tuple(
        RequestSample(
            ttft_s=ttft,
            tpot_s=tuple(gaps),
            e2e_s=e2e,
            tokens=tuple(tokens_per_item[i]),
            output_tokens=max_tokens,
        )
        for i in range(batch_size)
    )


def _transformers_sample(
    model: Any,
    torch: Any,
    prompt: Any,
    max_tokens: int,
    device: str,
) -> RequestSample:
    return _transformers_batch_sample(model, torch, prompt, max_tokens, device, 1)[0]


def _prompt_ids(
    tokenizer: Any,
    torch: Any,
    prompt_tokens: int,
    image_tokens: int,
    device: str,
) -> Any:
    text = " ".join(["token"] * max(1, prompt_tokens))
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded["input_ids"]
    width = int(ids.shape[-1])
    if width > prompt_tokens:
        ids = ids[:, :prompt_tokens]
    elif width < prompt_tokens:
        pad = int(getattr(tokenizer, "eos_token_id", None) or 0)
        extra = torch.full((1, prompt_tokens - width), pad, dtype=ids.dtype)
        ids = torch.cat([ids, extra], dim=1)
    if image_tokens > 0:
        pad = int(getattr(tokenizer, "eos_token_id", None) or 0)
        extra = torch.full((1, image_tokens), pad, dtype=ids.dtype)
        ids = torch.cat([ids, extra], dim=1)
    return ids.to(device)


def _load_transformers() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise BenchHarnessError(
            "the transformers backend needs torch and transformers, "
            "which are not installed. They are optional and are not "
            "base dependencies."
        ) from exc
    return torch, transformers


def _version_note(config: GenerateConfig) -> dict[str, str]:
    if config.backend == "stand-in":
        return {"stand-in": "1"}
    if config.backend == "transformers":
        try:
            import transformers
        except ImportError:
            return {}
        version = getattr(transformers, "__version__", "")
        return {"transformers": str(version or "unknown")}
    return {config.backend: "external"}


def _timing(config: GenerateConfig) -> tuple[str, str, str]:
    if config.backend == "stand-in":
        return (
            "virtual",
            "Stand-in service time advances a virtual clock. "
            "TTFT scales with prompt and image tokens. "
            "Concurrency adds queueing and still raises aggregate tokens/s. "
            "Peak memory is the stand-in weight buffer, not a GPU probe.",
            "stand-in",
        )
    if config.backend == "transformers":
        return (
            "perf_counter",
            "Transformers prefill and decode are timed with perf_counter. "
            "Image tokens are extra prefill positions, not pixels. "
            "4-bit paths need a CUDA device.",
            "transformers",
        )
    return (
        "perf_counter",
        "External server measured with the OpenAI streaming load generator. "
        "Image tokens lengthen the text prompt; the server is not sent pixels.",
        config.backend,
    )


def _identity(config: GenerateConfig) -> str:
    precisions = ",".join(config.precisions)
    return "\n".join(
        [
            "generate-v1",
            config.backend,
            config.model,
            config.workload,
            precisions,
        ]
    )


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(value: object, flag: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchHarnessError(f"{flag} must be a non-empty string")
    return value.strip()


def _positive_ints(values: Sequence[int], flag: str) -> tuple[int, ...]:
    items = _int_items(values, flag)
    if not items:
        raise BenchHarnessError(f"{flag} needs at least one positive integer")
    return tuple(_check_count(item, flag, minimum=1) for item in items)


def _non_negative_ints(values: Sequence[int], flag: str) -> tuple[int, ...]:
    items = _int_items(values, flag)
    if not items:
        raise BenchHarnessError(f"{flag} needs at least one integer")
    return tuple(_check_count(item, flag, minimum=0) for item in items)


def _int_items(values: Sequence[int], flag: str) -> Sequence[object]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise BenchHarnessError(f"{flag} needs a list of integers")
    return values


def _virtual_seconds(points: Sequence[dict[str, Any]]) -> float:
    total = 0.0
    for point in points:
        rate = float(point["tokens_per_s"])
        if rate > 0:
            total += float(point["output_tokens"]) / rate
    return total


def _check_count(value: object, flag: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchHarnessError(f"{flag} values must be integers >= {minimum}")
    return value


def _require_valid(document: dict[str, Any]) -> None:
    try:
        validate(document)
    except BenchValidationError as exc:
        raise BenchHarnessError(
            f"bench result does not match ohho.bench/v1:\n{exc}",
            exit_code=1,
        ) from exc
