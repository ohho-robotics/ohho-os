# DRAFT — pending Varun review — CV, ML and robotics

**Status:** DRAFT — pending Varun review. Do not send, export to PDF, or publish until the contact block and any employment or education lines are filled in by hand.

**Varun Vaidhiya**

Learning and agent loops for a mobile manipulator, wired so the same robot API records data, trains a small policy, and serves it. GPU training stacks are separate and are not claimed as finished real-robot results.

| | |
|---|---|
| GitHub org | https://github.com/ohho-robotics |
| Site | https://ohho-robotics.com |
| YouTube | https://www.youtube.com/@varun.vaidhiya/videos |
| X | https://x.com/varunvaidhiya |
| Email / phone / location | _not in the repositories — fill in_ |
| Employment / education | _not in the repositories — fill in_ |

Public handles above are the ones already printed on the OmniBot README. The git author name on this checkout is Varun Vaidhiya.

## Summary

Built the record → train → serve path inside OhhO OS, plus a native agent loop that falls back to a scripted brain when the optional model stack is absent. Closed-loop checks that are actually specified run on the in-process simulator or on CPU. No accuracy, success rate, or latency percentile from a physical robot or from a finished nightly run is stated in the sources below, so none is stated here.

## Data, train, serve (OhhO OS)

Repository: https://github.com/ohho-robotics/ohho-os (README Source URL still says https://github.com/ohho-robotics/ohho-sdk). Tree version 1.1.3. Published PyPI release: `ohho-os` 1.1.2.

- **Record.** `ohho.data.Recorder` wraps a `Robot`, intercepts `drive` and `move_joints`, and polls telemetry. Frames are 9-D (6 arm + 3 base) on a mobile manipulator and 3-D on a base-only robot. Writes a LeRobot v2.0 layout: `meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`, and Parquet when the `[data]` extra (pyarrow) is installed, JSON Lines otherwise.
- **Train.** `ohho.train.finetune()` delegates to `lerobot_engine` when torch and lerobot are installed (`[train]`). Policy names the code accepts: `smolvla`, `act`, `diffusion`, `openvla`. `mock=True` writes a checkpoint from dataset stats and does not need a GPU.
- **Tiny ACT on CPU.** `ohho/train/act.py`: Action Chunking Transformer (CVAE + Transformer). The sim loop trains it on CPU.
- **Serve.** FastAPI app (`ohho.serve`): `GET /health`, `POST /load_model`, `POST /predict`. `ohho serve` starts it. A mock model path exists for machines without a checkpoint loader.
- **Acceptance shape in tests.** `AGENTS.md` describes `tests/test_data.py` recording 10 frames from a simulated OmniBot, mock-training a smolvla checkpoint, and `POST /predict` returning a 9-D action. That is a unit-level sim loop, not a robot success rate.

## CPU sim loop (specified, nightly first run pending)

`ohho sim-loop` / `python scripts/sim_loop.py` (README):

1. Record at least 5 scripted teleop episodes (100 frames in the default description) from `omnibot` on `sim://` into a LeRobot v2.0 dataset.
2. Validate the schema and load it with `LeRobotDataset`.
3. Train tiny ACT for 200 steps on CPU. Checkpoint files: `policy.pt`, `config.json`, `metrics.json`.
4. Boot FastAPI, load that checkpoint, run at least 50 closed-loop steps on the simulated robot.
5. Compute serve-latency p50 and p95. **No numeric latency is published in the README**, so this draft does not invent one.

Defaults also appear as `--episodes 5 --steps-per-episode 20 --train-steps 200 --eval-steps 50`. `--mock` runs without PyTorch.

`.github/workflows/sim-loop.yml` is cron `0 2 * * *` and `workflow_dispatch`, Ubuntu, Python 3.12, CPU PyTorch wheels, `lerobot[dataset]==0.3.3`, extras `[data,serve,train]`, 20-minute job timeout, artifact upload. README: designed for CPU runners, **first run pending**.

## Agent, navigation, memory

- `HarnessBrain` (`ohho/brains.py`) runs perceive → reason → act → reflect when `agent_engine` imports. Tools are built from the robot's capabilities. `ClaudeToolCallingReasoner` is used when `ANTHROPIC_API_KEY` and the `[agent]` extra are present; otherwise an echo fallback. `ScriptedBrain` is the no-dependency path. Agents are not given `arm` or `release_stop`. They may call `emergency_stop`.
- Navigation (`ohho/nav.py`): grid mapping from scans, inflated costmap, A*, frontiers, stuck detection and goal diversion. CLI: `ohho nav goto`, `explore`, `map`.
- Memory: object permanence and spatial/temporal queries, JSON file per robot. `ohho look` and `ohho memory show|where|near|clear`. `SimPerceptor` needs no camera. `VlmPerceptor` is optional and calls Claude vision when configured.
- Base-install tests skip the brain tests that need `agent_engine` and numpy (13 skips in the README's 197-test Linux run).

## What lives on the OmniBot repo, and what it needs

https://github.com/ohho-robotics/OmniBot

| Piece | Needs | Where it is documented |
|---|---|---|
| Kinematics, Yahboom codec, arm tick math, learning engine, agent engine, data engine tests | CPU | `cpu` job in `.github/workflows/ci.yml` |
| LeRobot teleop record launch | The robot | `omnibot_lerobot` `teleop_record.launch.py`. README marks this launch "On the physical robot (docs-test skips this)". |
| Nav2 and SLAM | Gazebo, not the robot | README publishes local sim times: kitchen, desk, and door, 5/5 each (15.7/17.0/17.5 s, 18.8/20.0/21.6 s, 8.2/12.8/13.4 s, min/median/max). Pass rule: Nav2 success within 60 s and final `/odom` within 0.30 m and 0.50 rad. Local Docker, not a copied GitHub Actions table. The roadmap checkbox for the same item is still open. |
| OpenVLA / SmolVLA weights | GPU (`torch` in `vla_engine/requirements.txt`) | `omnibot-ai-engines/vla_engine/README.md` |
| Isaac Lab RL | GPU (`isaaclab`, `onnxruntime-gpu`) | `omnibot-ai-engines/rl_engine/README.md` |
| Headless Gazebo Harmonic, `world:=flat` | ROS 2 Jazzy and Gazebo, not a GPU | `sim-smoke` job. Launch test expects `/odom`, `/scan`, `/imu`, and `/camera/front/image_raw` within 30 s, and `/cmd_vel` moving the **simulated** robot at least 0.2 m |

Headset-side recording (https://github.com/ohho-robotics/OhhO-VR) captures arm state, odometry, and commands at 30 Hz into JSONL on the device, then POSTs the file to `http://<robot>:8765/upload_episode`. The README says that HTTP bridge is **not in the VR repository**.

## Do not claim

- A policy trained on real OmniBot demonstrations, or a success rate, grasp rate, or collision rate.
- A published p50 or p95 from `sim-loop`. The pipeline computes them; the README does not print a number, and the nightly workflow's first run is pending.
- That OpenVLA, SmolVLA, or Isaac Lab training has been run to a reported score. Those trees are GPU requirements, not results tables.
- That the agent has run a Claude tool-calling session on hardware. The default without the extra and an API key is the scripted or echo path.
