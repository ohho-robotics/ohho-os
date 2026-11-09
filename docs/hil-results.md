# HIL results (OHH-35)

Bench record for the four hardware-in-the-loop tests in `tests/hil/test_hil.py`.
The committed copy of this file is the empty template. On the bench,
`ohho hil --hardware --preflight-ok` rewrites it after `docs/hil-preflight.md`.
CI runs `ohho hil --mock` and leaves this file empty.

| Field | Value |
|---|---|
| Date (UTC) |  |
| Operator |  |
| Host |  |
| Revision |  |
| Mode |  |
| OHHO_HIL |  |
| OmniBot base port |  |
| OmniBot arm port |  |
| Go2 interface |  |
| Bag (2-min MCAP) |  |

## Pre-flight

Checklist: `docs/hil-preflight.md`. The harness does not measure these.

| Check | Result | Notes |
|---|---|---|
| Batteries |  |  |
| Wiring |  |  |
| E-stop |  |  |
| Lidar |  |  |
| Cameras |  |  |
| Pi 5 Jazzy image |  |  |

## Tests

| Test | Adapter | Status | Detail |
|---|---|---|---|
| OmniBot base drive and read | yahboom |  |  |
| OmniBot arm move and read | feetech |  |  |
| OmniBot composite base and arm | yahboom + feetech |  |  |
| Unitree Go2 drive and read | unitree DDS |  |  |

## Notes

