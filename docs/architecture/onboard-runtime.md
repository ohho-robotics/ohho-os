# Onboard runtime

Status: Proposed

OHH-354 (RT-01). This ADR records the scope, placement, and naming of the
onboard runtime. Nothing in it is implemented. It does not add a package, a
module, a CI check, or any runtime code.

The onboard runtime is a robot-agnostic layer that runs on the robot. It is
not a third executor beside the native and ROS 2 backends, and it is not a
new profile type. Later issues RT-02 through RT-22 implement pieces of this
decision. They should follow this ADR.

## Scope

The onboard runtime covers nine areas. Each area is named here so later
issues do not grow a second supervisor, a second bus, or a second config
model.

1. **Process supervision and lifecycle.** Start, stop, and restart the
   processes on the robot. The safety watchdog is running before any service
   that can command motion, and it stops last. Readiness and shutdown order
   belong to this area. Shutdown commands the class failure action before it
   stops the watchdog.

2. **IPC.** A typed bus for small messages, and a separate path for large
   payloads (images, clouds, maps). A slow consumer must not pin a large
   buffer for the life of the process. This bus is not `Runtime.publish`.

3. **QoS.** Reliability, history depth, and a deadline on each stream. A
   missed deadline on a command or state stream is a fault in the failure
   table. After that deadline the last command is not replayed.

4. **Scheduling and real-time config.** Each service may set `nice`
   (-20..19), `policy` (`other`, `fifo`, or `rr`), `priority` (1..99, and
   only with `fifo` or `rr`), `cpu_affinity`, and `mlockall`. The default
   is `other` with the other fields unset, which changes nothing. Period
   stays a loop hint. This is configuration. It is not a promise of hard
   real-time on a stock kernel (see Non-goals). Recommended order, highest
   first: the safety watchdog and transport, then control loops, then
   perception and the agent. Do not raise a policy or a priority until a
   LoopTimer measurement for that service shows the misses the change is
   for. A host that cannot apply the declared value reports the refusal.
   It does not silently run the service at a different priority.

5. **Health monitoring and heartbeats/deadlines.** Every supervised service
   has a heartbeat or a deadline. A miss feeds the failure table. Loop timing
   is recorded with the service that owns the loop.

6. **Fault recovery and safety-preserving degradation.** The failure table
   below is the behaviour. Recovery restarts only what that table allows. It
   does not clear a safety latch and it does not arm the robot.

7. **Observability.** Logging, metrics, loop timing, and a crash bundle for a
   failed process. A crash bundle omits credentials, tokens, and the arming
   phrase.

8. **Configuration.** The resolved view of `RobotProfile` and
   `HardwareProfile`, plus the service config from the manifest (RT-02).
   There is no third profile type. A safety override that arrives through
   config can only tighten a cap. A config that fails to load leaves the
   robot do-not-operate.

9. **Deployment.** systemd units, a versioned bundle, and rollback to the
   previous bundle. The bundle identity is a version, not a floating
   "latest". A bundle that fails its health check is not left active.
   Rollback leaves the robot disarmed. No unit and no bundle sets
   `OHHO_ARM_HARDWARE`.

## Placement

The onboard runtime lives in a new top-level `runtime/` directory in this
repository. It is a separate Python distribution, laid out like a plugin
package: its own project file, its own import root, its own tests.

```
runtime/
├── pyproject.toml          # name = "ohho-runtime", Apache-2.0
├── src/ohho_runtime/       # import package ohho_runtime
└── tests/
```

`ohho-runtime` is the distribution name. `ohho_runtime` is the import
package. The directory is not `ohho/runtime/`, not `ohho/`, and not
`plugins/`. It is not an `ohho.robots` entry point.

Dependency direction is one-way. `ohho_runtime` depends on `ohho-os` and
imports only public `ohho` APIs: public modules and public names, nothing
with a leading underscore, and no adapter module. The public surface it is
allowed to use is `ohho`, `ohho.runtime` (the executor port),
`ohho.registry`, `ohho.robot_spec`, `ohho.profiles`, `ohho.safety`, and,
once their PRs have merged, `ohho.plugins` and `ohho.config`. `ohho` does
not import `ohho_runtime`. The root `pyproject.toml` does not list
`ohho-runtime` as a dependency or as an extra.

The package stays in this repository while its service contract is still
moving against `RobotSpec`, `RobotPlugin`, and `SafetyPolicy`. It moves to
its own repository when all four of these are true:

1. The import name is still `ohho_runtime`, and the dependency is still
   one-way into public `ohho` APIs.
2. `ohho` still does not import it, and the import lint passes on both
   trees.
3. Supervisor releases, image builds, or soak gates need a cadence separate
   from `ohho-os` library releases.
4. The move changes packaging only. Callers install `ohho-runtime` the same
   way. No new profile type appears.

Until those hold, a second repository would fork the contract this ADR pins
down.

## Naming

"Onboard runtime" always means the `ohho-runtime` distribution. "Executor
port" means the package that already exists at `ohho/runtime/`.

The executor port is the in-process API engines already use. `ohho/runtime/base.py`
defines `Runtime`: an in-process pub/sub bus (`subscribe`, `publish`), a
parameter store (`get_param`, `set_param`), and the abstract lifecycle
`now`, `create_timer`, `start`, and `stop`. `TimerHandle.cancel()` stops one
timer. `RuntimeUnavailable` is the error when a backend cannot be built.
Callback errors on the bus are swallowed so one subscriber cannot kill the
process.

`ohho/runtime/native.py` is `NativeRuntime`. It is a standard-library
threaded scheduler: one daemon thread (`ohho-native-runtime`), a default
tick of 5 ms, and `time.monotonic()` as the clock. It is the backend that
runs anywhere Python runs.

`ohho/runtime/ros2.py` is `Ros2Runtime`. It puts an rclpy node behind the
same port. rclpy is imported lazily. Tests inject `node_factory` and need no
ROS 2 install. The generic bus is JSON in `std_msgs/String`. Node parameters
back `get_param` / `set_param`. `start()` runs `rclpy.spin_once` on a
background thread. `stop()` drops timers, publishers, and subscriptions, and
shuts rclpy down when this object initialised it. Hardware topics
(`Twist`, `Odometry`, `JointState`) stay on the ROS 2 transport adapter.
They are not this bus.

`get_runtime("auto")` prefers `ros2` when rclpy is importable, and otherwise
returns `native`. The other legal names are `native` and `ros2`.
`available_runtimes()` lists what this environment can construct. The
onboard runtime is not a value of `get_runtime`. A supervised process may
call `get_runtime` itself. That call stays an `ohho` import. It is not a
reason for `ohho` to import `ohho_runtime`.

One string collides and should not be read as this package. `Ros2Runtime`'s
default rclpy node name is `ohho_runtime`. That is a node name inside the
executor port. It is not the `ohho-runtime` distribution, and this ADR does
not rename the node.

### Profiles the runtime consumes

Two profile types already exist. The runtime reads both. It does not add a
third.

`HardwareProfile` (`ohho/profiles.py`) is the compute topology: which role
runs on which machine. Roles are `control`, `perception`, `inference`,
`training`, and `simulation`. A `NodeSpec` carries the machine name, those
roles, a device, and an accelerator. The built-in profiles are
`pi_workstation`, `jetson_single`, `workstation_single`, `mac_dev`, and
`edge_cpu`. `detect_profile()` follows `OHHO_HW_PROFILE` when it is set, and
otherwise probes the machine. The onboard runtime uses this to decide where
a service runs. It does not choose the computer.

`RobotProfile` is the per-robot file, not the compute topology. It is not on
this branch. Open PR #72 (OHH-314) adds `RobotProfile` in `ohho/config.py`:
`~/.ohho/robots/<id>.toml`, or a repo-local `ohho.toml`, holding the default
transport URI, sim backend, asset path, network interface, and safety
overrides. An override that raises any cap is refused at load. The runtime
consumes that object as that PR defines it. Safety caps themselves stay on
`SafetyProfile` inside the gate. `RobotProfile` is allowed to tighten those
caps and is not allowed to loosen them.

## RobotSpec, plugins, and safety

The runtime consumes `RobotSpec` v2 (`docs/architecture/robot-plugins.md`,
OHH-309, PR #66). `robot_class` is the closed enum `mobile_base`,
`mobile_manipulator`, `arm`, `quadruped`, `aerial`, `humanoid`. Limits,
joint groups, assets, and `capability_params` stay on the spec. The runtime
reads them. It does not add fields to `RobotSpec`.

`RobotPlugin` is the protocol in open PR #68 (OHH-310). A plugin exposes
`specs`, `transports`, `sim_backends`, `safety_policy`, `descriptor`, and
`doctor`. The runtime uses the spec and the safety policy that discovery
already loaded. It does not import a plugin's vendor SDK, and it does not
call `doctor` in order to move the robot.

A plugin may optionally declare extra services with a `services(spec)` hook.
That hook is not on the protocol today. RT-02 adds it after PR #68 merges,
and RT-02 owns the manifest schema. `None`, or a plugin that has no such
method, means the class default services only. A declaration may add a
service. It may not remove the safety watchdog, transport, or telemetry, and
it may not name a failure action looser than the table below. The hook is
not called at import time.

Per-class `SafetyPolicy` is open PR #70 (OHH-313). That PR keeps arming, the
audit log, the e-stop latch, and the watchdog schedule on the gate, and
moves class decisions into a policy: allowlist and denylist, clamp,
`on_watchdog`, `on_estop`, arming, and fall or tilt. The policies on that PR
are ground (mobile base), arm, legged (quadruped), and ground-plus-arm
(mobile manipulator). Aerial and humanoid still use the ground policy there.
A plugin may return a stricter policy from `safety_policy(spec)`. A policy
that loosens a class cap is refused. `None` keeps the class default.

The runtime consumes those policies and never weakens them. It does not
bypass `SafetyGate`, it does not lengthen a watchdog, it does not add a
denied action, and it does not arm. Agents still have no arm or release
tool. When this table and the class policy both apply, the runtime takes the
stricter action: less commanded motion, a latch at least as sticky, and no
automatic re-arm. `docs/safety.md` remains the rule book for the gate.

`requires_hardware_arming`, `experimental`, and `flight_ready` on an aerial
spec are descriptions. They are not an arming call. `experimental` or a
false `flight_ready` means the aerial services that command motion stay
do-not-operate.

## Failure modes

These tables are the specified behaviour. They are not implemented.

Seven service roles: transport (the actuator link), safety watchdog (the
process that enforces `SafetyPolicy`), telemetry (the state stream), teleop
gateway (the operator entry), policy (the process that produces motion
commands — a learned policy or an agent loop, not `SafetyPolicy`),
perception, and recorder.

Rules for every row:

- Stale outputs of the failed service are invalidated in the same step.
  Consumers do not act on them.
- No auto re-arm. Clearing a latch stays a human `arm()` at a TTY, which
  leaves the gate disarmed until that arm succeeds. The runtime never calls
  it.
- A process restart uses backoff. It does not arm, it does not raise a cap,
  and it does not resume a latched robot. The safety watchdog is not
  restarted while any motion command is permitted.
- Transport restarts only while the robot is disarmed.
- Unknown posture does not damp. Joint commands are dropped, not driven to
  zero. The denylist still wins.

Class safe actions used in the tables:

| `robot_class` | Safe action |
|---|---|
| `mobile_base` | stop (zero velocity) |
| `mobile_manipulator` | stop the base; hold the arm |
| `arm` | hold (drop joint commands) |
| `quadruped` | stop. A failure does not command sit |
| `aerial` | do-not-operate. Land once only when the class policy names land, the vehicle is airborne, and telemetry is fresh; then send nothing else. Otherwise send no setpoints and leave the vehicle failsafe enabled |
| `humanoid` | stop. Hold a standing posture only when the last trustworthy state is upright and the class policy says to hold. Do not command a stand. Damp only when the class policy already damps a robot that is down |

### `mobile_base`

| Service | On failure |
|---|---|
| transport | stop. Latch. No auto re-arm. Restart with backoff only while disarmed. |
| safety watchdog | do-not-operate. Stop if a command path remains. Latch. No auto re-arm. |
| telemetry | invalidate stale outputs. Stop. Latch. No auto re-arm. |
| teleop gateway | stop. No auto re-arm. Restart the gateway with backoff; it sends nothing until a new operator heartbeat. |
| policy | invalidate stale outputs. Stop. Do not replay the last command. Restart with backoff stays stopped until deadlines are healthy. No auto re-arm. |
| perception | invalidate stale outputs. Stop perception-guided motion. A live operator may keep driving only while the teleop gateway, transport, telemetry, and the watchdog are healthy. Restart perception with backoff. |
| recorder | degrade. Close the episode and leave motion and the gate unchanged. Restart with backoff. |

### `mobile_manipulator`

| Service | On failure |
|---|---|
| transport | stop the base and hold the arm. Latch. No auto re-arm. Restart with backoff only while disarmed. |
| safety watchdog | do-not-operate. Stop the base and hold the arm if a command path remains. Latch. No auto re-arm. |
| telemetry | invalidate stale outputs. Stop the base and hold the arm. Latch. No auto re-arm. |
| teleop gateway | stop the base and hold the arm. No auto re-arm. Restart the gateway with backoff; it sends nothing until a new operator heartbeat. |
| policy | invalidate stale outputs. Stop the base and hold the arm. Do not replay the last command. Restart with backoff stays in that state until deadlines are healthy. No auto re-arm. |
| perception | invalidate stale outputs. Stop perception-guided base motion and hold the arm. A live operator may keep driving the base only while the teleop gateway, transport, telemetry, and the watchdog are healthy. Restart perception with backoff. |
| recorder | degrade. Close the episode and leave the base, the arm, and the gate unchanged. Restart with backoff. |

### `arm`

| Service | On failure |
|---|---|
| transport | hold. Latch. No auto re-arm. Restart with backoff only while disarmed. |
| safety watchdog | do-not-operate. Hold if a command path remains. Latch. No auto re-arm. |
| telemetry | invalidate stale outputs. Hold. Do not servo toward a stale joint target. Latch. No auto re-arm. |
| teleop gateway | hold. No auto re-arm. Restart the gateway with backoff; it sends nothing until a new operator heartbeat. |
| policy | invalidate stale outputs. Hold. Do not replay the last joint command. Restart with backoff stays held until deadlines are healthy. No auto re-arm. |
| perception | invalidate stale outputs. Hold. Do not servo on a stale pose. Restart perception with backoff. |
| recorder | degrade. Close the episode and leave the arm and the gate unchanged. Restart with backoff. |

### `quadruped`

A failure stops the robot. It does not sit the robot. Unknown posture does
not damp. Damp stays the class policy's decision for a robot that is already
lying down.

| Service | On failure |
|---|---|
| transport | stop. Latch. No auto re-arm. Restart with backoff only while disarmed. |
| safety watchdog | do-not-operate. Stop if a command path remains. Latch. No auto re-arm. |
| telemetry | invalidate stale outputs. Stop. Do not sit on a stale posture. Latch. No auto re-arm. |
| teleop gateway | stop. No auto re-arm. Restart the gateway with backoff; it sends nothing until a new operator heartbeat. |
| policy | invalidate stale outputs. Stop. Do not replay the last command. Restart with backoff stays stopped until deadlines are healthy. No auto re-arm. |
| perception | invalidate stale outputs. Stop perception-guided motion. A live operator may keep commanding only while the teleop gateway, transport, telemetry, and the watchdog are healthy. Restart perception with backoff. |
| recorder | degrade. Close the episode and leave motion and the gate unchanged. Restart with backoff. |

### `aerial`

`experimental` or a false `flight_ready` is already do-not-operate, before
any failure. A failure does not clear those flags. The runtime does not
replace the vehicle failsafe and does not disable it.

| Service | On failure |
|---|---|
| transport | do-not-operate. Land once only when the class policy names land, the vehicle is airborne, and telemetry is fresh; then send nothing else. Otherwise send no setpoints and leave the vehicle failsafe enabled. Latch. No auto re-arm. Restart with backoff only while disarmed. |
| safety watchdog | do-not-operate. Land only under the same three conditions; otherwise send no setpoints. Latch. No auto re-arm. |
| telemetry | invalidate stale outputs. Do-not-operate. Send no setpoints, including a position land. Leave the vehicle failsafe enabled. Latch. No auto re-arm. |
| teleop gateway | do-not-operate. Land once only when the class policy names land, the vehicle is airborne, and telemetry is fresh. Otherwise send no setpoints. No auto re-arm. Restart the gateway with backoff; it does not resume flight. |
| policy | invalidate stale outputs. Do not replay the last thrust or setpoint. Land once only when the class policy names land, the vehicle is airborne, and telemetry is fresh; otherwise do-not-operate. Restart with backoff does not resume flight. No auto re-arm. |
| perception | invalidate stale outputs. Drop perception-derived setpoints. If the active behaviour requires perception to stay airborne, treat it as a policy failure. Otherwise degrade and leave non-perception commands in place only while telemetry and the watchdog are healthy. Restart perception with backoff. |
| recorder | degrade. Close the episode and leave the flight and the gate unchanged. Restart with backoff. |

### `humanoid`

| Service | On failure |
|---|---|
| transport | stop. Hold only under the humanoid safe action. Latch. No auto re-arm. Restart with backoff only while disarmed. |
| safety watchdog | do-not-operate. Stop, and hold only under the humanoid safe action, if a command path remains. Latch. No auto re-arm. |
| telemetry | invalidate stale outputs. Stop. Hold only when the last trustworthy state was upright and the class policy says to hold. Unknown posture does not stand and does not damp. Latch. No auto re-arm. |
| teleop gateway | stop. Hold only under the humanoid safe action. No auto re-arm. Restart the gateway with backoff; it sends nothing until a new operator heartbeat. |
| policy | invalidate stale outputs. Stop. Do not change mode and do not replay the last joint command. Restart with backoff stays stopped until deadlines are healthy. No auto re-arm. |
| perception | invalidate stale outputs. Stop perception-guided walking. Hold only under the humanoid safe action. Restart perception with backoff. |
| recorder | degrade. Close the episode and leave posture and the gate unchanged. Restart with backoff. |

## Licensing

`ohho-runtime` is Apache-2.0, the same licence as `ohho-os`.

The base install's required dependencies are permissive and compatible with
Apache-2.0 (Apache-2.0, MIT, BSD, or the Python standard library). Importing
`ohho_runtime` with no extras must not load a GPL, LGPL, or proprietary
module. The safety watchdog and the class safe action run on that base
install.

GPL, LGPL, and proprietary libraries are optional extras only. They are never
required. Isolation, which is the same lazy-import rule `ohho-os` already
uses for its extras:

- Each such library is its own extra in `runtime/pyproject.toml`. The default
  install and any extra that the watchdog needs do not pull them in.
- The import sits inside the function that uses the library, not at package
  import.
- Public modules of `ohho_runtime` do not re-export the library.
- The runtime does not vendor copyleft or proprietary source into `runtime/`.
- A plugin's SDK stays in that plugin. The runtime does not import it.
- RT-02 adds a CI check that imports `ohho_runtime` in a clean environment
  without those extras and fails if a GPL, LGPL, or proprietary module was
  imported.

## Non-goals

- No hard real-time guarantees on stock kernels. A missed deadline is a
  fault in the failure table, not a best-effort slip that keeps the last
  command.
- No replacement for ROS 2. `Ros2Runtime` stays the ROS 2 executor. RT-22
  integrates with ROS 2; it does not reimplement the ROS 2 graph.
- No hardware control in CI. Tests use a fake clock and fake transports. They
  do not connect to a robot, do not set `OHHO_ARM_HARDWARE`, and do not start
  systemd on a machine.
- No new `get_runtime` name, and no new profile type.
- No change to safety caps, watchdogs, arming, or the e-stop latch.

## CI checks

Specified here, implemented in RT-02, not added by this change.

**Import lint.** `ohho/` must not import `ohho_runtime`. A static scan of
`ohho/**/*.py` fails the job on any of these:

- `import ohho_runtime`
- `from ohho_runtime ...`
- `importlib.import_module("ohho_runtime...")`
- `__import__("ohho_runtime...")`

The root `pyproject.toml` fails the same job if `dependencies` or
`optional-dependencies` lists `ohho-runtime`. The scan does not import either
package. `runtime/` may import public `ohho` modules. Docs may name the
package.

**Vendor-name grep.** C-07, specified in `docs/architecture/robot-plugins.md`,
scans core Python under `ohho/` and the root `pyproject.toml`. Plugin trees
may name a vendor. Core may not. RT-02 extends that scan to the onboard
runtime: `runtime/src/**/*.py` and `runtime/pyproject.toml`, same tokens and
the same allowlist rules. `runtime/tests/` follows the existing fixture
whitelist (unlisted tests are not scanned). Markdown, including this ADR,
stays unscanned. `plugins/` stays exempt.

## Related

OHH-154 is the compute decision: a Jetson Orin-class main computer versus
the alternatives. `HardwareProfile` already describes several topologies,
including `jetson_single` and split edge / workstation profiles. This ADR
does not pick a board. The runtime reads the `HardwareProfile` it is given.

Also related: `docs/architecture/robot-plugins.md` (RobotSpec v2, OHH-309)
and `docs/safety.md` (SafetyGate).

## Follow-up issues

Short names only. RT-02 is the first change that adds `runtime/` and the two
CI checks above, including `services(spec)` after PR #68 merges.

| Issue | Name |
|---|---|
| RT-02 | manifest schema |
| RT-03 | service API |
| RT-04 | supervisor |
| RT-05 | degradation |
| RT-06 | deadline primitive |
| RT-07 | safety watchdog |
| RT-08 | QoS |
| RT-09 | typed bus |
| RT-10 | shared-memory IPC |
| RT-11 | LoopTimer |
| RT-12 | scheduling config |
| RT-13 | logging/metrics |
| RT-14 | crash bundle |
| RT-15 | runtime CLI |
| RT-16 | resolved config |
| RT-17 | systemd units |
| RT-18 | deploy bundle |
| RT-19 | fault injection |
| RT-20 | soak gates |
| RT-21 | multi-arch CI |
| RT-22 | ROS 2 integration |

Hardware timing baselines are later and hardware-gated. They are not part of
RT-02 through RT-22.

## Consequences

- `ohho/` stays free of an `ohho_runtime` import. Executor selection stays
  `auto`, `native`, and `ros2`.
- Safety caps, watchdogs, arming, and the e-stop latch stay where they are.
- The failure table is the degradation contract for all six robot classes.
- A package added later under `runtime/` follows this ADR. This change adds
  only this file.
