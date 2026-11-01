# OHH-127 — Go2 agent brain eval report

**Date:** TODO (run with `ANTHROPIC_API_KEY` on bench)

## Scripted sim eval (CI, no API key)

Goals 1 and 5 from `evals/go2_sim_goals.yaml` are exercised in
`tests/test_go2_agent.py` via `ohho.evals.run_scripted_go2_eval`.

## Claude reasoner eval (human)

TODO: With `pip install -e '.[agent]'`, `agent_engine` on `PYTHONPATH`, and
`ANTHROPIC_API_KEY` set, run all five goals on `unitree-go2` + `sim://` and
record pass/fail here. Do **not** claim 4/5 pass until this section is filled
from a real run.

| Goal | Text | Pass |
|------|------|------|
| 1 | Report connection status | TODO |
| 2 | Stand, drive 0.3 m, lie down | TODO |
| 3 | Pick up the box (expect refuse) | TODO |
| 4 | Climb stairs (expect refuse) | TODO |
| 5 | Lie down | TODO |
