# Safe model files and the worker sandbox (OHH-485)

Managed workers load untrusted checkpoints. The loader prefers safetensors
and ONNX. A `.pt` or `.bin` pickle is scanned with `pickletools` before
`torch.load(..., weights_only=True)`. The scan does not execute the pickle.
Globals such as `os`, `nt`, `posix`, `subprocess`, and `builtins.exec` /
`builtins.eval` are refused. Any other `GLOBAL` or `REDUCE` outside the
weights allowlist is refused, with the global name in the error.

Set `OHHO_MANAGED_WORKER=1` on a cloud worker. In that mode a `.pt` / `.bin`
file is refused unless the scan passes and `weights_only=True` succeeds.
There is no second load without `weights_only`. Outside that mode the same
scanner still runs before a weights load. Bytes that are not a pickle are
left for `weights_only` so an existing caller can keep its current error.

## Loading

```python
from ohho.model_safety import load_weights, prepare_model_load

prepare_model_load(path)          # scan only; hub ids that are not local files are ignored
weights = load_weights(path)      # safetensors, ONNX, or weights-only torch
```

`ohho.serve` calls `prepare_model_load` before `/load_model` and before an
automatic startup load. A refusal is HTTP 400 and includes the reason.
`ohho.train.finetune` calls it when `checkpoint` is a local path.
`load_act_checkpoint` scans `policy.pt` before `torch.load`.

`ohho.optimize` is not in this tree. Call `prepare_model_load` before that
package reads a user checkpoint. Wiring it in is a follow-up.

Safetensors are parsed in the standard library (header JSON plus raw tensor
bytes). ONNX is parsed with the optional `onnx` package and is not run.
A pickle whose name ends in `.safetensors` or `.onnx` is refused.

A requested CUDA or MPS device is used only when that runtime is available.
Otherwise the file is loaded on CPU and `LoadedWeights.device_note` says so.
This loader does not allocate a GPU.

TorchScript locomotion files (`.pt` from `torch.jit.save`) are not
weights-only checkpoints. `LocomotionPolicy` still uses `torch.jit.load`
outside managed mode. Managed workers refuse those files. Use `.onnx` for a
managed locomotion worker.

## Worker job sandbox

`ohho.worker_sandbox.container_run_spec` returns a JSON-ready description of
one job container. It does not start a container, pull an image, or open a
socket.

| Field | Value |
|---|---|
| Root filesystem | read-only |
| Host mounts | none |
| User | non-root (`uid`/`gid` 10001, name `ohho`) |
| CPU, memory, pids | positive limits (defaults: 2 CPU, 4096 MB, 256 pids) |
| GPU | `count` is recorded; `attached` is false. This module does not attach a device. |
| Scratch | in-container path `/scratch`, wiped when the job ends |
| Network | allowlist only, not host networking |
| Egress | `api.ohho-robotics.com:443` (console API) and `registry.ohho-robotics.com:443` (registry storage). The list can be narrowed. It cannot be widened. |
| Credentials | `OHHO_JOB_TOKEN` is named and injected at runtime. The spec does not contain the token. User tokens (`OHHO_TOKEN`, `HF_TOKEN`, cloud and model-provider keys) are refused. |

`JobScratch` is the local helper for the scratch directory. After `wipe()`,
or after the context manager exits, the directory exists and
`verify_empty()` succeeds. A symlink inside the directory is removed; the
target outside it is not deleted.

`check_egress(host, port, resolve=...)` decides whether a destination is on
the allowlist. `resolve` is supplied by the caller. A name that is not on
the allowlist is rejected before `resolve` is called. A resolved address
that is not a public unicast address (loopback, private, link-local,
metadata) is rejected. `open_allowed_connection` calls `socket_factory`
only after the check allows the host. Neither function is a packet filter.
The orchestrator still has to enforce the allowlist on the network.

Confirm the two hostnames against the deployed console API and registry
before a worker fleet uses the spec.

## Security review

Sign-off below is a human step. Cloud workers must not accept external
users until it is recorded.

- [ ] Review the pickle opcode allowlist against the torch build the workers run.
- [ ] Confirm every worker sets `OHHO_MANAGED_WORKER=1` and that a failed `weights_only` load is not retried.
- [ ] Confirm the orchestrator applies this spec: read-only root, no host mounts, non-root uid, CPU and memory limits, scratch wipe.
- [ ] Confirm egress is enforced in the network, not only by `check_egress`.
- [ ] Confirm the only injected credential is a short-lived `OHHO_JOB_TOKEN`, and that user platform tokens are absent.
- [ ] Confirm `api.ohho-robotics.com` and `registry.ohho-robotics.com` are the deployed console API and registry storage hosts.
- [ ] GPU attachment, if added later, keeps the same root, user, mount, and egress rules.
- [ ] Record security-review sign-off before cloud workers accept external users.

## Follow-ups

- Call `prepare_model_load` from `ohho.optimize` (OHH-429 owns that module; it is not edited here).
- Console UI for a refusal or for the sandbox spec.
- Attach a GPU in the orchestrator. The spec records the count and leaves `attached` false.
- Serve managed locomotion from ONNX (or safetensors) instead of TorchScript.
