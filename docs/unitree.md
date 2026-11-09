# Unitree Go2 (`unitree-go2`)

This page documents how OhhO OS talks to a **Unitree Go2** over DDS (`dds://`).
It is the honest status page for the `UnitreeDdsTransport` adapter — what is
implemented, what is tested, and what is **not** verified on real hardware.

## Verification status

| Environment | Status |
|---|---|
| **unitree_mujoco** (or other local DDS sim on loopback, domain **1**, iface **`lo`**) | Adapter + `SafetyGate` exercised in **unit tests** with injected state; suitable for software integration only |
| **Physical Go2** | **Not tested** by OhhO maintainers. No on-robot bring-up, no HIL run, no camera/lidar validation on hardware |

**Nothing in this repository has been validated on a physical Unitree Go2.** Do not
treat CI green or mock/DDS-loopback tests as evidence that a real robot will
behave correctly.

**Hardware bring-up** for a real Go2 is tracked as **Vision** until a Go2 EDU is
available and an operator is on site — see **OHH-135**. Until then, use
`sim://` or a local DDS simulator; do not connect production code to a live
robot on domain 0 without a deliberate bench plan.

## Install

The base `ohho-os` wheel does not include CycloneDDS or Unitree’s SDK.

```bash
pip install -e '.[unitree]'
```

You also need **`unitree_sdk2_python`** (`unitree_sdk2py`) on your machine.
OhhO imports it lazily inside `UnitreeDdsTransport.connect()`. If it is missing,
connect raises `AdapterUnavailable` with install hints.

Check that the adapter is visible:

```bash
ohho doctor
```

## Transport URI

| URI | Meaning |
|---|---|
| `dds://<iface>` | DDS on network interface `<iface>` (default **domain 0** — the real robot domain) |
| `dds://lo?domain=1` | Loopback + **domain 1** — typical for **unitree_mujoco** / local simulators |
| `dds://eth0` | Example hardware iface (needs arming; see below) |

Query parameter: `?domain=<n>` (parsed by the adapter). **Domain 0 on any
interface is treated as hardware** for arming policy. **Loopback + non-zero
domain** (e.g. `lo` + domain `1`) can arm without the hardware phrase when
other gate checks pass — see [`docs/safety.md`](safety.md).

```python
from ohho import Robot

# Software / sim DDS (domain 1, loopback) — still not a substitute for hardware test
bot = Robot.connect("unitree-go2", "dds://lo?domain=1")

# Hardware-shaped URI — gated, disarmed until arm(); not validated on real Go2 here
bot = Robot.connect("unitree-go2", "dds://eth0")
```

CLI example (hardware-shaped; **do not run against a real robot** until OHH-135):

```bash
OHHO_ARM_HARDWARE=1 ohho drive unitree-go2 --transport dds://eth0 --arm --vx 0.2
```

## SafetyGate and arming

Every non-simulated transport, including `dds://`, is wrapped in
`ohho.safety.SafetyGate` at `Robot.connect`. The robot starts **disarmed**;
`drive()` and sport actions are refused (zero velocity is sent) until `arm()`
succeeds.

On **hardware** (non-loopback DDS, or DDS **domain 0**), arming requires **all**
of:

1. `OHHO_ARM_HARDWARE=1`
2. An **interactive TTY** (`sys.stdin.isatty()`)
3. Typing the confirmation phrase (see [`docs/safety.md`](safety.md))
4. No CI environment (`CI`, `GITHUB_ACTIONS`, etc.)

**Agents, CI jobs, and non-TTY processes fail closed** — they cannot arm and
cannot release an e-stop on hardware.

For local DDS simulators, prefer **`dds://lo?domain=1`** so domain 0 (the live
robot bus) is never touched by mistake.

Velocity caps on Go2 (after arming): 0.5 m/s forward, 0.3 m/s lateral,
1.0 rad/s yaw, with acceleration limits; sport `euler` angles clamped to ±0.3 rad.
Full rules: [`docs/safety.md`](safety.md).

## Sport mode: allowlist vs prohibited commands

High-level poses go through `bot.transport.command(action, *args)`, which the
gate filters before `UnitreeDdsTransport.sport_command()` calls the SDK.

**Allowlisted** (normalised names, e.g. `StandUp` → `stand_up`):

- `stand_up`, `stand_down`, `balance_stand`, `recovery_stand`
- `stop_move` (accepted even when disarmed or e-stopped)
- `move`, `euler` (±0.3 rad per axis)
- `sit`, `rise_sit`

**Always refused** (denylist wins over config files): any action whose name
contains `flip`, `jump`, `pounce`, `dance`, `handstand`, `upright`,
`cross_step`, or starts with `free` — including `front_flip`, `backflip`,
`walk_upright`, `free_walk`, etc.

`damp` exists on the adapter for e-stop handling but is **not** on the agent
allowlist; the gate controls when `Damp()` may run. Agents and market skills do
not receive `arm` or `release_stop` tools.

## What the adapter implements today

- **Velocity** → `SportClient.Move(vx, vy, vyaw)`
- **State** → polling thread (~20 Hz) → `Telemetry` (odom, battery, roll/pitch
  in `custom` when the SDK provides IMU)
- **Sport commands** → mapped SDK methods (`stand_up`, `euler`, …)
- **No arm** — `send_joint_command` is a no-op; low-level leg control is out of scope

Unit tests use a **`state_factory`** injection path — no DDS and no hardware
required (`tests/test_unitree.py`).

## Sim vs hardware gaps

These are known limitations relative to a full Go2 product stack:

| Area | In OhhO today | On real Go2 (unverified here) |
|---|---|---|
| Sport / locomotion | High-level `Move` + allowlisted sport actions | Same API surface; SDK/version shim differences possible (`GetState` vs `_state` fallback in adapter) |
| Perception RGB / depth / scan | Listed on `RobotSpec` for parity | **No** camera or lidar driver in `UnitreeDdsTransport`; `ohho look` / scan-based nav need another pipeline |
| `ohho selftest` | OmniBot-focused bench suite | **No** Go2 selftest profile in this package |
| `ohho sim-serve --backend mujoco` | Prints `not built` | Use **unitree_mujoco** (or similar) + `dds://lo?domain=1`, not `sim-serve` mujoco backend |
| HIL | Contract only | See below |

## Hardware-in-the-loop (HIL) contract

`tests/hil/test_hil.py` defines `TestHilUnitreeGo2`:

- **Skipped unless** `OHHO_HIL=1`
- Uses `OHHO_GO2_IFACE` (default `eth0`)
- Connects, reads telemetry, sends zero velocity, disconnects

This test is a **stub for bench operators**. It has **not** been run against a
physical Go2 in OhhO CI or by maintainers. Enabling `OHHO_HIL=1` is an explicit
opt-in to touch real hardware — follow OHH-135 and local safety procedures.

```bash
# Contract only — do not run without a bench plan and cleared workspace
OHHO_HIL=1 OHHO_GO2_IFACE=eth0 python -m unittest tests.hil.test_hil.TestHilUnitreeGo2 -v
```

## Related docs

- [`docs/safety.md`](safety.md) — SafetyGate rules, profiles, audit log
- [`README.md`](../README.md) — install, HIL table, safety overview

## Maintainer checklist (OHH-133)

- [ ] Doc states **simulation / unit-test only** for Go2 — no hardware claims
- [ ] OHH-135 gates real bring-up
- [ ] Arming: `OHHO_ARM_HARDWARE=1` + TTY; CI/agents fail closed
- [ ] Default sim DDS: `dds://lo?domain=1`
- [ ] Allowlist vs denylist documented
- [ ] `TestHilUnitreeGo2` documented as `OHHO_HIL=1` only
