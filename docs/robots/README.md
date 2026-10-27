# Supported robots

Built-in robot ids come from `ohho.registry` (`ohho doctor list`).

## OmniBot (`omnibot`)

- **Base:** Yahboom Rosmaster X3 over USB serial (`serial://`, `[serial]` extra).
- **Arm:** SO-101 Feetech bus (`feetech://`, `[arm]` extra).
- **Status:** Software-complete; mock/loopback unit tests in CI. On-robot bring-up
  is validated via `tests/hil/` (see `scripts/run_hil.sh`).

Serial adapters implement telemetry timeout, watchdog stop on link loss, and
automatic reconnect (see `ohho/adapters/yahboom.py`, `ohho/adapters/feetech.py`).

## Unitree Go2 (`unitree-go2`)

- **Transport:** `dds://<interface>` (`[unitree]` extra).
- **Status:** **experimental — untested on hardware.** The DDS adapter is covered by
  unit tests with injectable fakes; a full bench run on a physical Go2 has not been
  recorded in this repository. Use `OHHO_ARM_HARDWARE=1` and the safety gate
  (`docs/safety.md`) if you try it on a real robot.

## Generic sim (`sim`)

In-process holonomic simulator (`sim://`). No hardware.
