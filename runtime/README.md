# ohho-runtime

Onboard service lifecycle for OhhO OS. This is the `ohho-runtime` distribution
(import package `ohho_runtime`). It is not the executor port at `ohho/runtime/`.

`ohho_runtime` depends on public `ohho` APIs. `ohho` does not import
`ohho_runtime`.

## Service API

`ohho_runtime.service` is a lifecycle state machine:

`created → configured → starting → ready → stopping → stopped`, with `failed`
when configure, start, readiness, liveness, or stop fails.

- `configure` / `start` / `ready` / `stop` move the machine.
- `heartbeat` refreshes liveness. `ready` reports readiness. They are separate.
- `health` returns `ok`, `degraded`, or `failed`, plus a reason and both probes.
- `poll` commits a missed readiness timeout or a missed heartbeat to `failed`.
- SIGTERM calls `stop`. A subprocess that does not exit within the grace period
  is killed, and that kill is recorded.

Adapters:

- `InProcessService` — a target callable on a thread.
- `SubprocessService` — a console script (`ohho-runtime-service`,
  `python -m ohho_runtime.worker`) with heartbeats on a Unix domain socket.
  Platforms without `AF_UNIX` raise `UnixSocketUnavailable`; tests skip.
- `TransportService` — an `ohho` transport. `connect` is start. When the
  transport is a `SafetyGate`, liveness uses that gate's public state-freshness
  window and only the samples the gate would count. It does not command the
  robot and does not change the gate.

## Tests

From the repository root, after `pip install -e .` and
`pip install -e ./runtime --no-deps`:

```bash
python -m pytest runtime/tests -q
```

Licence: Apache-2.0.
