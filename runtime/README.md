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
named QoS profiles, and an optional scheduling hint.

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

## Operator CLI

`ohho-runtime` is the operator command. `up` runs in the foreground and
supervises a manifest in dev mode: in-process services, no hardware, and
no arming. The other commands talk to that process over a loopback TCP
socket (`127.0.0.1`, or `::1` when `--host ::1` is set). The socket is
never bound to any other address.

```bash
ohho-runtime up --class mobile_base
ohho-runtime status
ohho-runtime status --json
ohho-runtime health
ohho-runtime health --json
ohho-runtime logs --service transport --level info
ohho-runtime restart transport
ohho-runtime down
```

`up --class` uses the class default manifest. `up --manifest FILE` loads
the JSON manifest below. `up` writes a state file so the other commands
can find the socket. The default path is `~/.ohho/runtime/control.json`.
`--state` selects another file, which is how tests avoid sharing one
supervisor. Ctrl+C on `up` stops the supervisor and removes the state file.
`down` does the same. `down` does not signal the process by pid.

`status` prints each service's lifecycle state, restart count, and
heartbeat age. `health` exits `0` when every critical service is `ok` and
do-not-operate is clear, and exits non-zero otherwise. A failed degradable
or optional service does not, on its own, fail `health`. `--json` on
`status` and `health` prints one JSON object with `services`, `healthy`,
and `unhealthy`. `logs` prints structured log lines. `--service` keeps one
service. `--level` is a minimum (`info` includes `info` and above).
`--follow` keeps printing new lines until interrupted. `restart SERVICE`
restarts that service through the supervisor. A `never` policy is refused.
A service that restarts only while disarmed is refused when the robot is
not disarmed. Restart does not arm and does not clear do-not-operate.

Dev mode refreshes liveness for services that are already ready. Those
services run on the supervisor thread and do not have a loop of their own.
A service that failed to start is left failed.

`up` does not serve `/metrics` unless `--metrics` is set. With that flag
the existing endpoint binds `127.0.0.1`.

The `ohho` CLI builds its subcommands in-tree and does not load an entry
point for external commands. This package therefore does not register an
`ohho runtime` alias.

### Manifest file

`robot_class` is required. `log_level` and `log_format` are optional.
Each service names an entry the runtime already ships: `sim.ready`,
`sim.fail`, `entries.safety_watchdog`, `entries.transport`,
`entries.telemetry`, `entries.teleop_gateway`,
`entries.autopilot_heartbeat`, or `entries.mode_monitor`. `sim.ready` and
`sim.fail` are in-process stand-ins. They do not connect to a robot.
`restart` is `"never"` or an `on-failure` object. Loading the file does
not start a process.

```json
{
  "robot_class": "mobile_base",
  "log_level": "INFO",
  "log_format": "json",
  "services": [
    {
      "role": "safety_watchdog",
      "entry": "sim.ready",
      "restart": "never",
      "loop_hz": 20,
      "liveness_timeout_s": 30,
      "qos": ["state"]
    },
    {
      "role": "transport",
      "entry": "sim.ready",
      "depends_on": ["safety_watchdog"],
      "commands_motion": true,
      "restart": {
        "kind": "on-failure",
        "max_restarts": 3,
        "window_s": 60,
        "backoff_s": 0.5,
        "only_while_disarmed": true
      },
      "loop_hz": 50,
      "liveness_timeout_s": 30,
      "qos": ["command", "state"]
    },
    {
      "role": "telemetry",
      "entry": "sim.ready",
      "depends_on": ["transport"],
      "restart": "never",
      "loop_hz": 20,
      "liveness_timeout_s": 30,
      "qos": ["telemetry"]
    }
  ]
}
```

## Tests

From the repository root, after `pip install -e .` and
`pip install -e ./runtime --no-deps`:

```bash
python -m pytest runtime/tests -q
```

Licence: Apache-2.0.
