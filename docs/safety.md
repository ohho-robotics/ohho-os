# SafetyGate (`ohho.safety`)

`SafetyGate` wraps a robot `Transport` and decides which commands reach the
motors. It implements the normal `Transport` interface, so `Robot`, the agent
and the training code do not change.

`Robot.connect` wraps every transport whose `protocol` is not `"simulated"`.
`sim://` (and the default "no transport" simulator) is not wrapped. You can
still wrap a `SimTransport` yourself for tests or demos; a simulated link arms
without a TTY prompt.

```python
from ohho import Robot

bot = Robot.connect("unitree-go2", "dds://eth0")   # gated, disarmed
bot.gated            # True
bot.drive(vx=0.3)    # refused: zero velocity sent, refusal audited
bot.arm()            # needs OHHO_ARM_HARDWARE=1, a TTY and the typed phrase
bot.drive(vx=2.0)    # clamped to 0.5 m/s, then acceleration-limited
bot.transport.command("stand_up")    # allowlisted sport action
bot.transport.command("front_flip")  # always refused
bot.emergency_stop() # latched
bot.release_stop()   # needs a human at a TTY; leaves the gate disarmed
bot.arm()            # re-arm before motion is accepted again
```

From the CLI, every command that takes a robot (`connect`, `drive`, `agent`,
`nav`, `look`, `market run`) accepts:

| Flag | Effect |
|---|---|
| `--arm` | Arm the hardware link after connecting (env var + typed confirmation). |
| `--estop-damp` | After an e-stop, call `Damp()` even if the robot is standing. |
| `--safety-config PATH` | Load caps/ceilings from a JSON file (default `$OHHO_SAFETY_CONFIG`). |

Without `--arm`, a hardware link stays disarmed and motion is refused.

## Rules

Every refused command also sends a zero-velocity command. Refused joint
position or effort commands are dropped, not "zeroed", because a zero joint
position would move the arm.

| Rule | Behaviour |
|---|---|
| Velocity caps | Clamped per axis (`max_vx`, `max_vy`, `max_wz`), never rejected. |
| Hard ceiling | 1.5 m/s linear, 2.0 rad/s angular. Profiles above it are lowered. Only a safety config file (`hard_ceiling`) can raise it. |
| Acceleration caps | Per axis, per second. Speeding up or reversing changes by at most `accel × dt`, with `dt` capped at 300 ms. Slowing toward zero is never limited. |
| Joint caps | Position clamped to `joint_limits[name]`, then to `max_joint_step` from the measured (or last commanded) position. Joints with no limit, no reference, or not in the robot's joint list are refused. |
| Effort caps | `send_joint_effort(name, effort)` is clamped to `±effort_limits[name]`. Refused when the joint has no limit or the transport has no `send_joint_effort`. |
| Command watchdog | No velocity command for 300 ms while moving → zero velocity (not latched). |
| State watchdog | No robot state for 500 ms while armed → stop and latch. On robots with `require_orientation` (Go2), a state without roll/pitch does not count. |
| E-stop latch | `emergency_stop()` calls the transport's stop (`StopMove()` on Go2) first, then `Damp()` only if the robot is already lying down (body height < 0.15 m) or `estop_damp` is set. Unknown posture never damps. |
| Release | `release_stop()` needs an interactive TTY and the word `release` on hardware, is refused in CI, is refused while the latest state is still tilted or faulted, and leaves the gate **disarmed**. |
| Allowlist | `command(action, *args)` forwards only allowlisted actions. Go2: `stand_up`, `stand_down`, `balance_stand`, `recovery_stand`, `stop_move`, `move`, `euler` (each angle clamped to ±0.3 rad), `sit`, `rise_sit`. Names are normalised (`StandUp` → `stand_up`). `stop_move` is accepted even when disarmed or latched. |
| Denylist | Any action containing `flip`, `jump`, `pounce`, `dance`, `handstand`, `upright`, `cross_step`, or starting with `free`, is refused even if a config file lists it (the config loader rejects it). |
| Tilt / fault guard | `|roll|` or `|pitch|` > 0.6 rad, unparseable orientation, or a non-zero `error_code` → stop and latch, armed or not. |
| Fail closed | A transport exception, a watchdog error or an unwritable audit log trips the e-stop or disarms. |

## Arming

The gate starts disarmed. Nothing arms it automatically, including tests and CI.

`arm()` succeeds only when the e-stop is not latched, the transport reports
`connected`, and a fresh state (< 500 ms) has arrived. On hardware it also needs
all of:

1. `OHHO_ARM_HARDWARE=1` in the environment;
2. no CI environment (`CI`, `GITHUB_ACTIONS`, `GITLAB_CI`, `BUILDKITE`,
   `JENKINS_URL`, `TF_BUILD`);
3. `sys.stdin` is an interactive TTY;
4. the operator types
   `I am physically present, the area is clear, the remote is in my hand`
   (case and extra spaces are ignored).

"Hardware" is every link except a simulator (`protocol == "simulated"`) and a
DDS link on a loopback interface with a non-zero DDS domain, e.g.
`dds://lo?domain=1` for a local Unitree simulator. DDS domain 0 is the real
robot's domain, so `dds://lo` (domain 0) still needs confirmation.

Agents never get `arm` or `release_stop` as tools. The TTY check stops a
headless agent or CI job. It does not stop code that runs in the same Python
process and reaches into private attributes; this is a software guard, not a
replacement for the robot's own e-stop or the remote.

## Profiles

| Robot | `max_vx` | `max_vy` | `max_wz` | accel x / y / yaw | Notes |
|---|---|---|---|---|---|
| `unitree-go2` | 0.5 | 0.3 | 1.0 | 1.0 / 0.6 / 2.0 | `require_orientation`, full Go2 allowlist |
| `omnibot` | 0.2 | 0.2 | 1.0 | 0.5 / 0.5 / 2.0 | SO-101 joint limits (mirrors `feetech.py`), allowlist `move`, `stop_move` |
| other robots | ≤ 0.3 | ≤ 0.3 (0 if not holonomic) | ≤ 1.0 | 0.5 / 0.5 / 1.0 | allowlist `move`, `stop_move` |

Timeouts (300 ms / 500 ms), the tilt limit (0.6 rad) and the euler limit
(0.3 rad) can be made stricter in a profile but not looser.

## Safety config file

```json
{
  "hard_ceiling": {"max_lin": 2.0, "max_ang": 3.0},
  "profiles": {
    "unitree-go2": {"max_vx": 0.8, "allowlist": ["move", "stop_move"]},
    "omnibot": {"effort_limits": {"arm_gripper": 1.0}}
  }
}
```

Pass it with `Robot.connect(..., safety_config=PATH)`, `--safety-config PATH`
or `OHHO_SAFETY_CONFIG=PATH`. Unknown keys, malformed values, denied actions or
caps above the file's own ceiling raise `ValueError`, so `Robot.connect` fails
instead of running with a half-read config.

## Audit log

Each event is one JSON object per line in `~/.ohho/safety/<UTC date>.jsonl`
(override the directory with `OHHO_SAFETY_DIR`). Every record has `ts`
(ISO 8601 UTC), `t` (epoch seconds), `event`, `robot` and `protocol`.

| `event` | When |
|---|---|
| `connect` | Gate connected (`hardware`: whether arming needs confirmation). |
| `arm` / `arm_refused` | Arming succeeded (with `user`, `host`) or was refused (`reason`: `env`, `ci`, `no_tty`, `not_confirmed`, `not_connected`, `no_fresh_state`, `estop_latched`). |
| `disarm` | `disarm()` or `disconnect()` while armed. |
| `clamp` | A command was changed (`requested`, `applied`, `reasons`). |
| `reject` | A command was refused (`command`, `reason`). |
| `watchdog_trip` | `kind`: `command` (zeroed) or `state` (followed by `estop`). |
| `estop` | E-stop latched (`reason`: `operator`, `tilt`, `fault`, `watchdog_state`, `transport_error`, `watchdog_error`). |
| `damp` | `Damp()` was sent after an e-stop (`why`: `lying_down` or `estop_damp`). |
| `estop_release` / `release_refused` | Release succeeded or was refused. |

## Tests and coverage

```bash
python -m unittest tests.test_safety
python -m coverage run --branch --include=ohho/safety.py -m unittest tests.test_safety
python -m coverage report -m --include=ohho/safety.py --fail-under=95
```

The tests use a fake clock, fake transports and injected TTY/env. They do not
sleep, start real timers, or touch hardware. CI runs the coverage command on
every matrix job and fails below 95 % line + branch coverage.

## Not covered here

- The MuJoCo end-to-end check (a 2 m/s command arrives as 0.5 m/s; killing the
  client stops the robot within 300 ms) is OHH-121. This repository has no
  MuJoCo launcher (`ohho sim-serve --backend mujoco` prints `not built`).
- `ohho selftest` and `tests/hil/` build adapters directly, not through
  `Robot.connect`, so they are not gated. They have their own operator consent
  (`--allow-spin`, `OHHO_HIL=1`).
- A consumer-side SafetyGate (OHH-143) and an MCU e-stop are separate work.
