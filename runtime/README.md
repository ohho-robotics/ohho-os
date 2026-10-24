# ohho-runtime

Onboard service lifecycle for OhhO OS. This is the `ohho-runtime` distribution
(import package `ohho_runtime`). It is not the executor port at `ohho/runtime/`.

`ohho_runtime` depends on public `ohho` APIs. `ohho` does not import
`ohho_runtime`.

## Runtime manifest

`ohho_runtime.manifest.RuntimeManifest` is the service set for one robot.
Each service has a role, an entry (Python callable, console script, or
ROS 2 launch/node), dependencies, criticality (`critical`, `degradable`,
`optional`), readiness and liveness timeouts, a restart policy (`never` or
`on-failure` with `max_restarts`, `window_s`, and backoff), a loop rate,
named QoS profiles, and optional scheduling (`nice`, `policy`,
`priority`, `cpu_affinity`, `mlockall`). Defaults are policy `other`
with those fields unset, which changes nothing.

Unknown QoS profile names and dependency cycles are rejected when the
manifest is loaded. The named profiles are `command`, `state`, `telemetry`,
and `sensor`.

`class_default_manifest(robot_class)` covers every robot class: transport,
the safety watchdog, telemetry, and the teleop gateway. Aerial adds
`autopilot_heartbeat`. Humanoid adds `mode_monitor`.

`resolve_manifest(spec)` uses a plugin `services(spec)` hook when that
method exists and returns a manifest. `None`, or a plugin with no such
method, keeps the class default. The hook is optional. It is discovered
with `getattr`, so existing plugins stay valid. A returned manifest may
add services. It may not drop transport, the safety watchdog, or telemetry.

The manifest does not start processes, connect to a robot, or arm. Call
`resolve_manifest` for a spec: an aerial spec that is experimental, or not
flight-ready, comes back with motion services marked do-not-operate.

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

## Supervisor

`ohho_runtime.supervisor.Supervisor` loads a manifest and builds its
dependency graph. Dev mode (`mode="dev"`, the default) runs each service
in-process on the caller thread: no subprocess, socket, or signal handler.
The same unit tests run on Linux, macOS, and Windows.

`start` launches a service only when every dependency is `ready`. `poll`
is how the clock moves. There is no sleep. Pass a `ManualClock` from
`ohho_runtime.service` in tests.

Restart uses the manifest policy. Failures inside `window_s` count toward
`max_restarts`. The failure that reaches the cap is not restarted. Backoff
doubles from `backoff_s` and is capped at 10 seconds, the same schedule as
the ohho-os link helper from PR #15. That helper is private and lives in
an adapter package, which `ohho_runtime` is not allowed to import, so
`ohho_runtime.backoff.ExponentialBackoff` keeps the schedule here.

A critical service that reaches the cap latches `do_not_operate`. The
supervisor holds the class safe action and `allows_motion()` stays false
until `clear_do_not_operate()`. That call is the human clear. It does not
arm the robot and does not restart the failed service. The
`on_do_not_operate` hook is the extension point for a later policy. This
layer does not change the safety gate.

A degradable service that reaches the cap stays failed. The other
services keep running.

`shutdown` stops launched services in reverse start order and records
whether that finished within `grace_s`. Events are appended to
`supervisor.events`.

## Scheduling

`SchedulingHint` on a service is optional.

| Field | Default | Meaning |
|---|---|---|
| `nice` | unset | Absolute niceness, -20..19. Unset leaves it. |
| `policy` | `other` | `other`, `fifo`, or `rr`. |
| `priority` | unset | Realtime priority 1..99. Required for `fifo` and `rr`. Rejected for `other`. |
| `cpu_affinity` | empty | CPU indexes. Empty leaves the mask. |
| `mlockall` | false | Lock the service process's current and future pages. |
| `period_s` | unset | Loop-period hint for LoopTimer. Not an OS scheduler call. |

`mode="dev"` (the default) runs services in-process and ignores OS
scheduling. `mode="process"` uses the same lifecycle. On Linux it applies
the declared values when the service starts. It probes `CAP_SYS_NICE`,
`RLIMIT_RTPRIO`, and `RLIMIT_MEMLOCK`. If the kernel will not accept the
declared value, the service still starts. The supervisor logs a warning
and increments `scheduling_refused_total`. It does not clamp a realtime
priority or swap in another policy. Niceness, policy, and affinity are
applied to the service pid. `mlockall` runs in the service process.

On macOS and Windows, and in dev mode, a declared OS scheduling config
produces one warning (`scheduling_ignored_total`) and no OS call.

Recommended order, highest first:

1. Safety watchdog and transport.
2. Control loops (teleop gateway, telemetry, mode monitor, autopilot heartbeat).
3. Perception, the policy or agent loop, and the recorder.

Do not raise a service's policy, realtime priority, or niceness until
`ohho_runtime.timing.loop_report` for that service shows the deadline
misses the change is meant to address. The class defaults do not raise
priority. Scheduling is not a hard real-time promise on a stock kernel.

OS calls live in `ohho_runtime.os_sched` and are injected, so the unit
tests do not need Linux. A privileged Linux host can check the result in
`/proc/<pid>/sched` and with `taskset -p`.

## Logging

`ohho_runtime.log.RuntimeLog` writes one record per event for the supervisor
and for every supervised service. Each record has these fields:

| Field | Meaning |
|---|---|
| `timestamp` | UTC time, `YYYY-MM-DDTHH:MM:SS.ffffffZ` |
| `level` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |
| `service` | Service role, or `supervisor` for supervisor events |
| `event` | Stable event name (`service_start`, `heartbeat_miss`, ...) |
| `robot_class` | Robot class value, for example `mobile_base` |
| `run_id` | Id of this runtime run |

The default format is JSON Lines (one JSON object per line). Development
mode is plain text (`service=... event=... robot_class=... run_id=...`).

The level and format come from the manifest, and the environment overrides
them when the variable is set:

| Source | Level | Format | Run id |
|---|---|---|---|
| Manifest | `log_level` on `load_manifest` / `RuntimeManifest` | `log_format` (`json` or `text`; `human` is text) | |
| Environment | `OHHO_RUNTIME_LOG_LEVEL` | `OHHO_RUNTIME_LOG_FORMAT` (`json` or `text`) | `OHHO_RUNTIME_RUN_ID` |
| Default | `INFO` | `json` | generated |

```python
from ohho_runtime.manifest import load_manifest

manifest = load_manifest(services, log_level="DEBUG", log_format="text")
```

A caught exception in runtime code is re-raised, or passed to
`note_exception`, which writes the record (including `exc_type`) and
increments `ohho_runtime_exceptions_total`. `runtime/tests/test_no_swallow.py`
scans `runtime/src` and fails on a bare `except:` or an
`except Exception: pass`.

## Metrics

`ohho_runtime.metrics` renders Prometheus text. No extra dependency is
required. `Registry.render()` is the text a scrape would return; tests use
it directly.

`GET /metrics` is served by `MetricsEndpoint`. It binds `127.0.0.1` and
port **9470** unless configured otherwise. It stays off until enabled, so
importing the runtime does not open a socket.

| Control | Effect |
|---|---|
| `OHHO_RUNTIME_METRICS=1` | enable (`0`, `false`, `off` disable) |
| `OHHO_RUNTIME_METRICS_HOST` | bind address (default `127.0.0.1`) |
| `OHHO_RUNTIME_METRICS_PORT` | port (default `9470`; `0` asks the OS) |
| `Supervisor(..., metrics_enabled=True)` | enable for that supervisor |

An explicit `metrics_enabled` / host / port argument wins over the
environment. The supervisor starts the endpoint in `start()` and stops it
in `shutdown()`.

Scrape the loopback endpoint:

```bash
export OHHO_RUNTIME_METRICS=1
export OHHO_RUNTIME_METRICS_PORT=9470
curl -s http://127.0.0.1:9470/metrics
```

Prometheus job:

```yaml
scrape_configs:
  - job_name: ohho-runtime
    static_configs:
      - targets: ["127.0.0.1:9470"]
```

`Supervisor.observe_loop(role, timer)` publishes the `LoopTimer` that role
owns. Cycle time is `ohho_runtime_loop_cycle_seconds` with a `quantile`
label of `0.5`, `0.95`, or `0.99`.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `ohho_runtime_service_state` | gauge | `service`, `state`, `robot_class`, `run_id` | `1` for the current lifecycle state |
| `ohho_runtime_service_restarts_total` | counter | `service`, `robot_class`, `run_id` | restarts so far |
| `ohho_runtime_heartbeat_age_seconds` | gauge | `service`, `robot_class`, `run_id` | seconds since the last heartbeat |
| `ohho_runtime_heartbeat_misses_total` | counter | `service`, `robot_class`, `run_id` | liveness deadlines committed to failed |
| `ohho_runtime_heartbeat_beats_total` | counter | `service`, `robot_class`, `run_id` | beats a subprocess heartbeat server accepted |
| `ohho_runtime_start_duration_seconds` | gauge | `service`, `robot_class`, `run_id` | last start duration |
| `ohho_runtime_stop_duration_seconds` | gauge | `service`, `robot_class`, `run_id` | last stop duration |
| `ohho_runtime_loop_cycle_seconds` | gauge | `service`, `quantile`, `robot_class`, `run_id` | LoopTimer cycle time at p50, p95, p99 |
| `ohho_runtime_loop_jitter_seconds` | gauge | `service`, `robot_class`, `run_id` | LoopTimer inter-start jitter |
| `ohho_runtime_loop_deadline_misses_total` | counter | `service`, `robot_class`, `run_id` | LoopTimer deadline misses |
| `ohho_runtime_loop_samples` | gauge | `service`, `robot_class`, `run_id` | LoopTimer samples in the report |
| `ohho_runtime_loop_lateness_seconds` | gauge | `service`, `robot_class`, `run_id` | latest LoopTimer lateness |
| `ohho_runtime_exceptions_total` | counter | `service`, `event`, `exc_type`, `robot_class`, `run_id` | caught exceptions that were logged |
| `ohho_runtime_do_not_operate` | gauge | `robot_class`, `run_id` | `1` while the latch is set |
| `ohho_runtime_allows_motion` | gauge | `robot_class`, `run_id` | `1` when the supervisor allows motion |
| `ohho_runtime_supervisor_state` | gauge | `state`, `robot_class`, `run_id` | `1` for the current supervisor state |

The content type is `text/plain; version=0.0.4; charset=utf-8`.

## Tests

From the repository root, after `pip install -e .` and
`pip install -e ./runtime --no-deps`:

```bash
python -m pytest runtime/tests -q
```

Licence: Apache-2.0.
