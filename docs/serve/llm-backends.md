# LLM serving backends

`ServingBackend` launches an external OpenAI-compatible server, reports its
version, and stops the process tree when you are done. The engines are not
dependencies of ohho-os. A base install does not download them.

```python
from ohho.serve.backends import open_backend

backend = open_backend("vllm")
backend.launch("org/model", "fp16", {"port": 8000})
print(backend.endpoint_url())  # http://127.0.0.1:8000/v1
print(backend.version())
backend.teardown()
```

`launch(model, precision, config)` returns when the health URL answers.
`health()` is that probe. `endpoint_url()` is the OpenAI base URL and
includes `/v1`. `teardown()` is idempotent and kills the engine's child
processes. `version()` runs the engine's version command.

If the process exits first, or the health URL stays down until
`startup_timeout_s` (default 30), `launch` raises `ServingBackendError`
with the command and the captured output, and no child process is left
behind.

## Backends

| Id | Program | Health | OpenAI URL | Precision |
|---|---|---|---|---|
| `vllm` | `vllm serve MODEL --host --port --dtype` | `GET /health` | `http://host:port/v1` | `auto`, `fp16`, `bf16`, `fp32`, `half`. `fp8` selects `--quantization fp8`. |
| `sglang` | `python -m sglang.launch_server --model-path` | `GET /health` | `http://host:port/v1` | Same dtype names as vLLM. `fp8` selects `--quantization fp8`. Default port 30000. |
| `tensorrt-llm` | `trtllm-serve serve MODEL --host --port` | `GET /health` | `http://host:port/v1` | `fp8` and `nvfp4` select `--kv_cache_dtype`. `fp16`, `bf16`, and `fp32` are properties of the checkpoint. `trtllm-serve` has no weight-dtype flag. |
| `tensorrt-llm` with `launcher=triton` | `tritonserver --model-repository --http-port --grpc-port --metrics-port` | `GET /v2/health/ready` | `http://host:openai_port/v1` | Precision for this launcher lives in the model repository. The command does not rewrite it. |
| `llama-cpp` | `llama-server -m FILE --host --port --n-gpu-layers` | `GET /health` | `http://host:port/v1` | KV cache via `--cache-type-k` and `--cache-type-v` (`fp16` → `f16`, and the llama.cpp cache names). `auto` omits the cache flags. |

`n_gpu_layers` defaults to `0` so a CPU machine does not offload. On Jetson
set `n_gpu_layers` to `auto` or a layer count.

Aliases: `trtllm` and `tensorrt` select `tensorrt-llm`. `llama.cpp` selects
`llama-cpp`.

Config keys shared by the adapters:

| Key | Meaning |
|---|---|
| `host` | Bind address. Default `127.0.0.1`. |
| `port` | Listen port. Omitted on a real launch, a free loopback port is chosen. |
| `startup_timeout_s` | How long `launch` waits for health. |
| `extra_args` | List of strings appended to the command. Not a shell line. |
| `executable` | Replace the engine binary (or, for SGLang, the Python interpreter). |
| `env` | Environment updates for the engine process. |
| `tp_size` / `tensor_parallel_size` | Passed through when the engine has that flag. |

`ohho bench launch --backend vllm --model org/model --precision fp16 --dry-run`
prints the command and URLs as JSON and does not start a process.
`ohho bench version --backend llama-cpp` prints the version, or says the
engine is not installed.

## Load generator

`ohho bench llm` posts streaming chat completions to an OpenAI-compatible
`/v1` URL. It keeps at most `--concurrency` requests in flight and records:

- time to first streamed chunk (TTFT)
- gaps between later chunks (inter-token latency)
- output chunks per wall-clock second (`tokens/s`)
- completed requests per wall-clock second

A chunk is one SSE `delta.content` or `text` piece. The generator does not
score answer quality. `accuracy.primary` is `request_completion_rate`.

```bash
ohho bench llm \
  --endpoint http://127.0.0.1:8000/v1 \
  --model org/model \
  --concurrency 4 \
  --requests 32 \
  --output result.json
```

The file is `ohho.bench/v1`. The `llm` object is `ohho.bench.llm/v1`.
`run.host.accelerator` is `none` unless the caller passes another value.
This package does not probe a GPU and does not fill accelerator memory or
utilization.

`run.model_sha` hashes a local model file of at most 8 MiB. Otherwise it
hashes the model identifier and `llm.model_sha_of` is `identifier`.

```python
from ohho.serve.loadgen import LoadSpec, run_load, to_bench_result

result = run_load(LoadSpec(
    endpoint_url="http://127.0.0.1:8000/v1",
    model="org/model",
    concurrency=4,
    requests=32,
))
document = to_bench_result(
    result,
    backend="vllm",
    backend_version="0.0.0",
    precision="fp16",
)
```

## What this tree does not measure

CPU tests use a local fake streaming server. They do not start vLLM,
SGLang, TensorRT-LLM, or llama.cpp, and they do not report GPU latency or
throughput. A real run needs the engine installed on a machine with the
matching device:

- vLLM, SGLang, and TensorRT-LLM (`trtllm-serve` or Triton) on a GPU host
- llama.cpp `llama-server` on a CPU host or on Jetson with `n_gpu_layers`
  set above 0

A console UI for choosing a backend and charting the `llm` block is not
part of this package.
