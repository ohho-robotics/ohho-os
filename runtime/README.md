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

## Tests

From the repository root, after `pip install -e .` and
`pip install -e ./runtime --no-deps`:

```bash
python -m pytest runtime/tests -q
```

Licence: Apache-2.0.
