# Heartbeat and deadline

Status: Proposed

OHH-359 (RT-06). This is the wire contract for one monotonic deadline and
the operator heartbeat. The primitive is `ohho_runtime.deadline`. It reads
`cmd_timeout_s` and `state_timeout_s` from a public `ohho.safety.SafetyProfile`.
It does not arm a robot, latch an e-stop, or send a command.

`ohho_runtime.heartbeat` stays the parent/worker socket (`ready` / `beat`
lines). It is not this topic.

## Wire

A default `SafetyProfile()` supplies the timeouts below. A stricter profile
shortens them. Nothing in this contract lengthens them.

| Name | Value |
|---|---|
| Heartbeat topic | `/ohho/heartbeat` |
| Heartbeat type | `std_msgs/Empty` |
| Rate | 10 Hz (period 0.1 s) |
| Heartbeat and command timeout | `SafetyProfile.cmd_timeout_s` (0.300 s on the default profile) |
| State timeout | `SafetyProfile.state_timeout_s` (0.5 s on the default profile) |
| Gate tick | 0.010 s |
| Stop budget | 0.350 s on the default profile. A shorter command timeout keeps the same slack and stays under 0.350 s. |
| Safety-state topic | `/ohho/safety/state` |
| Safety-state type | `std_msgs/String` |
| Safety-state JSON | `{"armed","latched","reason"}` (keys sorted) |
| E-stop reset | `/ohho/estop_reset` (`std_srvs/Trigger`) |

A sample is fresh while its age is strictly less than the timeout, and
expired when the age is greater than or equal to the timeout. No sample,
or a clock that moves backwards, is expired.

A missed heartbeat or a missed command zeros base velocity and does not
latch. The safety-state payload reports the latch; it does not clear it.
`/ohho/estop_reset` is the only clear. Publishing `/emergency_stop` false
does not clear a latch. The teleop gate starts latched (`reason` `startup`),
so a heartbeat alone cannot arm it. Reset is refused while the stop input
is still asserted.

The published stop budget is 0.350 s. Do not raise it. The gate ticks
every 10 ms, so a miss is acted on within one tick after the command
timeout. That sum stays inside 0.350 s on the default profile. A shorter
profile keeps the slack between the class-maximum command timeout and
0.350 s, and its budget is not longer than 0.350 s.

`wire_contract(profile)` copies the two timeouts from `profile`. The topic
names and the 10 Hz rate stay fixed. `Deadline.command` /
`Heartbeat` use `cmd_timeout_s`. `Deadline.state` uses `state_timeout_s`.
The clock is `time.monotonic` unless the caller injects one.

`class_timeout_ceiling(robot_class)` is the class maximum: the default
profile's two timeouts, for every robot class. `check_timeout_conformance`
fails when a declared `cmd_timeout_s` or `state_timeout_s` is longer.
`check_registered_timeouts` runs that check on `profile_for` for every
registered spec.

## Other repositories

Described from the contract those pull requests already publish. This tree
does not vendor them.

ohho-vr pull request 26 (`OhhoSafetyContract`):

- `/ohho/heartbeat`, `std_msgs/Empty`, 10 Hz, period 0.1 s.
- Watchdog 0.300 s. Do not raise. A miss zeros the base and does not latch.
- Stop budget 0.350 s, the watchdog plus one 10 ms tick. Do not raise.
- `/ohho/safety/state`, `std_msgs/String`, JSON fields `armed`, `latched`,
  `reason`.
- `/ohho/estop_reset`, `std_srvs/Trigger`, the only clear for a latched
  e-stop.
- The ROS heartbeat is `std_msgs/Empty`. A session log may carry a sequence
  and a client time; those fields are not the ROS message.

ohho-omnibot pull request 57 (teleop gate and `teleop_safety`):

- Clients publish `/ohho/heartbeat` as `std_msgs/Empty` at 10 Hz.
- The heartbeat timeout equals the command timeout, 0.300 s. A longer
  watchdog is rejected. The node exposes no timeout parameter.
- Command loss and heartbeat loss zero base velocity and do not latch.
- The gate starts latched with reason `startup`. A heartbeat cannot arm it.
- `/ohho/safety/state` is `std_msgs/String` JSON with `armed`, `latched`,
  and `reason`. The publisher is reliable, transient-local, depth 1.
- `/ohho/estop_reset` (`std_srvs/Trigger`) is the only clear. False on
  `/emergency_stop` does not clear the latch. Reset is refused while that
  stop input is still true.
- The node ticks every 10 ms on a monotonic clock, so a dead client is
  zeroed inside 0.350 s. `use_sim_time` true is rejected.
- Clients also publish `/cmd_vel/teleop_raw`, `/arm/joint_commands/teleop_raw`,
  and `/emergency_stop` (`std_msgs/Bool`). The gate publishes the mux inputs
  `/cmd_vel/teleop` and `/arm/joint_commands/teleop`.

Those numeric timeouts match a default `SafetyProfile` in this repository.
A profile here may only shorten them.

## Implementations in this repository

| Location | Role |
|---|---|
| `ohho/safety.py` `CMD_TIMEOUT_S` (0.3) and `STATE_TIMEOUT_S` (0.5) | Class ceiling. `SafetyProfile` copies them and refuses a longer value. |
| `ohho/safety.py` `SafetyGate.tick` | Command age `>= cmd_timeout_s` while moving zeros velocity and does not latch. State age `>= state_timeout_s`, or no state, stops and latches. |
| `ohho/safety.py` setpoint check | Fresh while age `< cmd_timeout_s`. |
| `ohho/sim_serve.py` `DEADMAN_TIMEOUT_S = 0.300` | Sim-serve deadman. `gated_velocity` is fresh while age `< timeout_s`. `call_later` uses the same constant. This is a second copy of the command timeout. |
| `ohho/verify.py` | Drives faster than `CMD_TIMEOUT_S`. The 300 ms text is a comment. The number is the constant. |
| `ohho/testing/conformance/runner.py` `_watchdog_zeroes` | Reads `gate.profile.cmd_timeout_s` and `state_timeout_s`. |
| `ohho/descriptor.py`, `ohho/robots/omnibot/descriptor_doc.py`, `ohho/robots/unitree/descriptor_doc.py` | Declare `"heartbeat": "/ohho/heartbeat"`. They do not publish it. |
| `docs/sim-serve.md`, `docs/safety.md` | Describe the 300 ms command watchdog and the 500 ms state watchdog. |
| `ohho/adapters/_link_resilience.py` | Serial telemetry timeout, default 2 s. Not this watchdog. |
| `ohho/robots/unitree/dds.py` `_STALE_STATE_SEC` | Link-health stale limit, 1 s. Not this watchdog. |
| `ohho_runtime/heartbeat.py` | Unix-socket `ready` / `beat` lines for a subprocess. Not `/ohho/heartbeat`. |
| `ohho_runtime/service/base.py` `heartbeat` | Service liveness. The window is `liveness_timeout_s` on the service, not the wire timeout. |
| `ohho_runtime/manifest.py` safety-watchdog service | `liveness_timeout_s=0.5`, the same number as the default state timeout, written as a literal. |
| `ohho_runtime/qos.py` | `command` deadline 0.1 s (the 10 Hz period). `state` deadline 0.5 s (the default state timeout). |
| `ohho_runtime/service/transport.py` | When the transport is a `SafetyGate`, liveness adopts `profile.state_timeout_s` and will not lengthen it. |
| `ohho_runtime/deadline.py` | This contract. Timeouts come from the profile. |

## Follow-up

This change does not edit `ohho/sim_serve.py` or `ohho/safety.py`.

`ohho/safety.py` keeps `CMD_TIMEOUT_S` and `STATE_TIMEOUT_S`. Those are the
ceiling `SafetyProfile` validates. Do not delete them.

`ohho/sim_serve.py` still has its own watchdog literal. Replace that
assignment so the deadman uses the public profile and the name
`DEADMAN_TIMEOUT_S` stays. `gated_velocity` and `call_later` already use
the name, so the behaviour does not change.

Add this import with the other local imports:

```python
from .safety import SafetyProfile
```

Replace:

```python
DEADMAN_TIMEOUT_S = 0.300
```

with:

```python
DEADMAN_TIMEOUT_S = SafetyProfile().cmd_timeout_s
```

After that edit, `rg "0\.3(00)?\b" ohho/sim_serve.py` finds no watchdog
literal. Until then the literal remains, and it is still equal to
`SafetyProfile().cmd_timeout_s`.

The same follow-up can point the safety-watchdog `liveness_timeout_s` in
`ohho_runtime/manifest.py` and the `state` QoS deadline in
`ohho_runtime/qos.py` at `SafetyProfile().state_timeout_s`, and the
`command` QoS deadline at `1.0 / HEARTBEAT_HZ`. Those are separate
literals. They are not the sim-serve watchdog, and this change does not
retune them.
