# OhhO OS CLI reference

The `ohho` command is the entry point for simulation, hardware bring-up, navigation,
memory, skills, and the record→train→serve loop. Install with `pip install ohho-os`
(or `pip install -e .` from a checkout).

## Environment and robots

| Command | Description |
|---------|-------------|
| `ohho doctor` | Python version, runtimes, adapters, robots, optional extras |
| `ohho list` | Built-in robot specs and capabilities |
| `ohho version` | Package version (`ohho -V` is equivalent) |

## Connect and drive

Robot subcommands take a **robot id** (from `ohho list`). Use `--transport sim://`
for in-process simulation (no hardware, no safety gate).

| Command | Description |
|---------|-------------|
| `ohho connect <robot> [--transport sim://]` | Connect, print status and telemetry, disconnect |
| `ohho sim --robot omnibot --seconds 2` | Drive a scripted pattern in simulation |
| `ohho drive <robot> [--transport sim://] --vx … --seconds …` | Open-loop velocity for a duration |
| `ohho agent <robot> [--transport sim://] "<goal>"` | Run the agent loop (scripted brain without `[agent]` extra) |

Non-sim transports are wrapped in **SafetyGate** and start disarmed; see
[`safety.md`](safety.md).

## Navigation, perception, memory

| Command | Description |
|---------|-------------|
| `ohho nav map <robot> [--transport sim://] [--world demo]` | Spin-scan and print occupancy grid |
| `ohho nav explore <robot> … [--frontiers N] [--world demo]` | Frontier exploration |
| `ohho nav goto <robot> … --x X --y Y [--timeout SEC]` | Closed-loop goto |
| `ohho look <robot> [--transport sim://] [--world demo]` | Sim perception → spatio-temporal memory |
| `ohho memory show <robot>` | Summary of remembered objects |
| `ohho memory where <robot> <label>` | Last known position of a label |
| `ohho memory near <robot> --x X --y Y [--radius R]` | Objects near a point |
| `ohho memory clear <robot>` | Erase persisted memory for that robot |

## Local skill registry (`skills`)

Skills are **registered in-process** (Python `@skill` decorators and imports). There is
no network catalog and no package manager — only behaviors available in your
environment.

| Command | Description |
|---------|-------------|
| `ohho skills list` | List registered skills |
| `ohho skills run <robot> [--transport sim://] <skill_name>` | Execute a skill |

`ohho market` is a **deprecated alias** for `ohho skills` (same subcommands). Register
skills with `ohho.market.skill`; see [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Profiles and inference

| Command | Description |
|---------|-------------|
| `ohho profile list` | Hardware deployment profiles |
| `ohho profile show <name>` | Profile details |
| `ohho profile detect` | Auto-detect this machine |
| `ohho serve [--mock] [--port 8000]` | FastAPI policy server (`[serve]` extra; `--mock` for CI/sim) |

With `--mock`, endpoints include `GET /health`, `POST /load_model`, and `POST /predict`.

## Simulation services

| Command | Description |
|---------|-------------|
| `ohho sim-serve --backend sim` | WebSocket teleop + telemetry (`[sim-serve]` / websockets) |
| `ohho sim-serve --backend isaac\|mujoco` | Exit non-zero with `not built` |
| `ohho sim-loop [--mock]` | Record → validate → train → serve → eval on CPU (see README) |
| `ohho selftest --sim` | Bench diagnostics against simulated hardware (CI-safe) |

## Smoke tests

CI runs `python -m unittest discover -s tests/cli_smoke` after installing
`ohho-os[serve]`, covering the commands above on `sim://` and `--mock` paths only.
