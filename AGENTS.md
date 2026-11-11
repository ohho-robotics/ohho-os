# OhhO OS — Engineering Handoff & Roadmap

> **Read this first.** This file is the single source of truth for continuing
> work on the OhhO OS SDK. It is written so that a fresh AI agent (or a human)
> with no prior context can understand what exists, why it's built this way, and
> exactly what to do next. Keep it up to date as milestones land.

> **This checkout is ohho-os** (Python `ohho-os` at the repository root).
> Paths written below as `sdk/` refer to that root (`tests/`, not `sdk/tests/`).
> The TypeScript SDK moved to [ohho-sdk](https://github.com/ohho-robotics/ohho-sdk);
> `ts/README.md` points there. The licence is Apache-2.0 (OHH-18).
> JSON schemas in `schemas/` are checked by `python scripts/export_json_schemas.py --check`.

- **Package:** `ohho-os` (imports as `ohho`) — repository root of
  [ohho-os](https://github.com/ohho-robotics/ohho-os).
- **What it is:** the open-source, robot-agnostic engine that powers the OhhO
  platform. One API controls any robot, **with or without ROS**, and carries the
  whole stack from perception to training.
- **Status:** **M0–M5 complete — v1.0.1 (verified).** All milestones shipped and
  independently verified from a fresh install: 151 unittest cases green in a
  fully-equipped env; in a **bare** env the optional-dep tests skip cleanly
  (agent_engine/numpy → brains tests; fastapi → the record→train→serve e2e; 4 HIL
  tests skip unless `OHHO_HIL=1` and an explicit device variable). `ohho` CLI works end-to-end with `doctor list
  version connect sim drive agent serve market profile`.
- **v1.0.1 hardening:** `Agent`/`ohho agent` no longer crash on a bare install —
  `brains.harness_available()` probes the real import chain and `HarnessBrain.run`
  falls back to `ScriptedBrain` with an explanatory log line; `ohho doctor` now
  reports agent-brain/serve/data/train availability dynamically (the stale
  "ROS 2 arrives later" line is gone — auto prefers ros2 when rclpy is present).
- **v1.1.0 — native nav + spatio-temporal memory:** the two former DimOS-parity
  gaps are closed. `ohho/nav.py` (occupancy-grid mapping from `Telemetry.scan`,
  costmap inflation, A*, frontier exploration, closed-loop `Navigator` with
  stuck detection + goal diversion; CLI `ohho nav goto|explore|map`) and
  `ohho/memory.py` + `ohho/perception.py` (entity store with object permanence,
  spatial/temporal queries, JSON persistence at `~/.ohho/memory/<robot>.json`;
  `SimPerceptor` + optional Claude-vision `VlmPerceptor`; CLI `ohho look`,
  `ohho memory show|where|near|clear`). The sim gained labeled obstacles,
  ray-cast scans and **solid-object collision**; the agent's tool registry gained
  `navigate_to, explore, look_around, where_is, objects_near, remember_note` and
  injects the memory summary into every goal. All stdlib — base stays dep-free.
- **Current version:** `1.2.0` (see `ohho/__init__.py` `__version__`). Not tagged and not on PyPI. The human checklist is [`docs/releases/ohho-os-1.2.0.md`](docs/releases/ohho-os-1.2.0.md). PyPI still has 1.1.2.

---

## 0. TL;DR — current state

| Area | State |
|---|---|
| Robot Abstraction Layer (`Robot`) | ✅ working |
| Capability model + registry | ✅ working (OmniBot, Unitree Go2, generic sim) |
| RobotSpec v2 | ✅ **OHH-309** — additive `robot_class`, joint groups, per-class limits, assets, `ci_sim`, `source`, `capability_params`. Safety caps unchanged. ADR `docs/architecture/robot-plugins.md` is proposed, awaiting Varun's review. |
| Robot plugins (`ohho.robots`) | ✅ **OHH-310** — `RobotPlugin` in `ohho/plugins.py`. Built-ins (sim, aerial), the mobile manipulator (`ohho.robots.omnibot`, **OHH-318**), and the quadruped (`ohho.robots.unitree`, **OHH-319**) load through the `ohho.robots` entry point. A source checkout without a reinstall falls back to every shipped entry in `pyproject.toml`. `ohho list --plugins` shows each plugin, its specs and any load error. **OHH-313** reads `safety_policy(spec)` when the gate loads (discovery still does not call it). |
| Humanoid plugin package | ✅ **OHH-330 / OHH-331** — `plugins/ohho-booster` is a separate distribution. Installing it registers `booster-t2` (`robot_class: humanoid`, 31 joints, `sim_backend=humanoid-mujoco`). No vendor SDK import at load. The description archive and `t2_walk.pt` are fetched at pinned commits with sha256 and are not vendored. `ohho sim --robot booster-t2` is headless MuJoCo. The checkpoint's training command range is not an Ohho cap. No `booster://` transport in this package. |
| Transport + sim registries | ✅ **OHH-311** — `resolve_transport` asks `RobotPlugin.transports()` first, then the existing scheme dict (entries unchanged). `sim://` and auto use `RobotSpec.sim_backend` when set (`kinematic` and `humanoid-mujoco` are the built-in backends); an empty name keeps today's simulator. Unknown schemes list each plugin's schemes. `dds://` stays Unitree. |
| Robot-id resolver | ✅ **OHH-314** — `ohho.config.resolve_robot_id`: `--robot`, then `OHHO_ROBOT`, then `robot = "..."` in `~/.ohho/config.toml`, then `sim`. `RobotProfile` (`~/.ohho/robots/<id>.toml` or repo-local `ohho.toml`) sets transport, sim backend, asset path, network interface, and safety overrides that may only tighten caps. `ohho config show` prints the id and source (`flag`, `env`, `config`, `default`). `SafetyGate` selects the per-class `SafetyPolicy` from that resolved spec's `robot_class`. |
| Conformance kit | ✅ **OHH-315** — `python -m ohho.testing.conformance --robot <id>` runs the class scenario on the plugin sim backend only. Quadruped runs `quadruped_walk` on the kinematic backend (stand before drive, lie/sit block motion, euler clamped to the safety limit). Aerial and humanoid skip with a reason until a sim backend is registered. Scenarios that open `SafetyGate` exercise the per-class `SafetyPolicy` (watchdog, e-stop, joint clamp). Vendor-name allowlist: `ohho/testing/conformance/vendor_allowlist.txt`. |
| Sim CI matrix | ✅ **OHH-316** — `ohho list --json --ci-sim` emits one row per spec with `ci_sim.enabled` (robot id, class, plugin, scenario). `.github/workflows/sim-matrix.yml` runs the class conformance scenario once per row on CPU, sim backends only, and writes that row's runtime to the job summary. Optional plugin `asset_source()` pins `owner/name` plus a commit; the workflow caches it with `actions/cache` and does not vendor meshes. `sim` (builtin) and the mobile manipulator (`ohho.robots.omnibot`) opt in. The quadruped plugin (`ohho.robots.unitree`) stays out of the matrix. |
| Runtime port + **native** backend | ✅ working (threaded scheduler, pub/sub, timers, params) |
| **ROS 2** runtime backend | ✅ **M4 complete** — `Ros2Runtime` (rclpy node: timers, pub/sub via topics, params via node params) + `Ros2Transport` (bridges /cmd_vel, /odom, /arm/joint_states). `get_runtime("auto")` prefers ros2 when rclpy present. `ros2://` scheme registered. Fully testable with `FakeRos2Node` (no ROS 2 needed). |
| Simulator adapter | ✅ working (holonomic physics + joint servoing) |
| **Hardware adapters** | ✅ M1 software complete — `serial://` (Yahboom base) + `feetech://` (SO-101 arm) + `composite` (base+arm merge for mobile-manipulators) + `dds://` (Unitree Go2 with real state polling) all built and mock-tested. **On-robot bring-up still pending** (hardware-gated). DJI/others not built. |
| Agent loop | ✅ **real brain** — `HarnessBrain` wires `agent_engine.AgentHarness` (perceive→reason→act→reflect). `ToolRegistry` built from robot capabilities. `ClaudeToolCallingReasoner` over `ReasoningRouter` (cloud Claude when `ANTHROPIC_API_KEY` + `[agent]` extra, echo fallback otherwise). `ScriptedBrain` remains the no-dep fallback. |
| Training / data / serve | ✅ **M3 complete** — `ohho.data.Recorder` (capture state+action from Robot → LeRobot v2.0), `ohho.train.finetune()` (delegates to lerobot_engine, mock mode for sim), `ohho.serve` (FastAPI server, `ohho serve` CLI). Record→train→serve loop works on sim. |
| PPO recipe (`ppo`) | ✅ **OHH-510** — `ohho.recipe/v1` family `ppo`: built-in CPU PyTorch PPO, `ohho.rl_tasks` entry points (CartPole, MuJoCo inverted pendulum, Isaac velocity), optional rsl_rl and skrl, episode reward and success on the train metrics stream, ONNX actor export. Reference task `mujoco-inverted-pendulum`, reward threshold 950. Isaac Lab GPU training is follow-up. Docs `docs/train/ppo.md`. |
| Serve bench harness | ✅ **OHH-431** — `ohho bench --artifact --target` writes `ohho.bench/v1` (schema from **OHH-473**) with mean/median/p95/min/max, per-stage timing, baseline deltas, and `--slo` / `--max-regression` gates. `--target` is an OHH-427 profile id. **OHH-472** samples optional power, memory, and thermal sources at 10 Hz during the measured window and adds `--sustained` p95 drift. **OHH-474** `ohho bench --pipeline` replays a camera stream (sim, LeRobot, rosbag fixture) through decode, preprocess, h2d, inference, d2h, postprocess, and publish. **OHH-498** `--export` publishes that result through optional W&B and Prometheus plugins (mocked in tests; failures are warnings). `export_result` is the bench and optimize hook. Console post-run export is out of scope. Docs `docs/serve/bench.md`. |
| Imitation recipes | ✅ **OHH-508** — ACT and Diffusion Policy recipes on LeRobot datasets through `ohho train`. Defaults, ranges, the validation stream, and the serve checkpoint layout: `docs/train/il-recipes.md`. Tiny CPU and single-GPU examples: `ohho/train/examples/il/`. |
| VLA recipes | ✅ **OHH-509** — SmolVLA and OpenVLA (LoRA and full) fine-tune recipes. BF16, gradient checkpointing, and LoRA rank/alpha/target modules are family parameters (OHH-524 will unify the shared precision field). Memory estimate, LoRA merge export, and base-model licence metadata: `docs/train/vla-recipes.md`. Launch examples: `ohho/train/examples/vla/`. |
| **LLM serving backends** | ✅ **OHH-475** — `ServingBackend` in `ohho/serve/backends` (`launch`, `health`, OpenAI `endpoint_url`, `teardown`, `version`) for vLLM, SGLang, TensorRT-LLM (`trtllm-serve` or `tritonserver`), and llama.cpp. Engines are external programs, not base dependencies. `ohho bench llm` writes `ohho.bench/v1` with an optional `ohho.bench.llm/v1` block (TTFT, inter-token latency, tokens/s, request throughput). GPU and Jetson runs are not in CI. Docs `docs/serve/llm-backends.md`. |
| **Generate-mode bench** | ✅ **OHH-436** — `ohho bench --mode generate` measures TTFT, TPOT, end-to-end latency, and tokens/s for `llm`, `vlm`, and `vla`. Default backend is a CPU stand-in (concurrency 1/4/16, prompt and image sweeps, FP16 vs bitsandbytes 4-bit / AWQ / GPTQ with weight memory and top-1 action-token agreement). `--slo ttft_p95<=X,tpot_p95<=Y` gates the primary cell. `transformers` and the vLLM / SGLang / TensorRT-LLM adapters are optional and only where the target supports them. Docs `docs/serve/bench.md`. |
| **LLM serving sweep** | ✅ **OHH-476** — `ohho bench llm --sweep` varies concurrency, max batch tokens, prefix cache, chunked prefill, and speculative decoding where the serving adapter supports each one. Workloads are short commands, long-context prompts with an image, and a shared system prompt. The `ohho.bench/v1` result carries a Pareto table of p95 TTFT against tokens/s and, with `--target-p95-ttft`, the highest-throughput configuration under that TTFT. Unsupported axes are skipped with a reason. CPU uses a stand-in; a one-GPU engine run is not in CI. Docs `docs/serve/bench.md`. |
| **LLM cost model** | ✅ **OHH-477** — `ohho bench` writes `ohho.bench.cost/v1` on inference, generate (`llm`, `vlm`, `vla`), and `ohho bench llm` results, including sweeps. Cloud rows are USD per GPU-hour. Edge rows are amortised hardware plus measured watts times a configurable electricity rate. Derived figures are USD per 1M output tokens, USD per 1k actions at the control rate, and actions per watt. `estimate` is true and `inputs` records the price and watts. `ohho bench cost` recomputes from saved throughput. Price table `ohho/targets/cost_prices.json`. The console comparison view is a separate change. Docs `docs/serve/bench.md`. |
| Dataset version splits and augmentation | ✅ **OHH-515** — seeded episode, session/day, task, and held-out-scene splits plus a train-time augmentation config (crop, resize, colour jitter, blur, state/action noise, temporal stride, per-camera toggles) stored on the dataset version and included in its content hash. `ohho.datasets.apply_augmentation` is what ACT and Diffusion Policy loaders call. Docs `docs/dataset-versions.md`. |
| **Training sweeps** | ✅ **OHH-518 / OHH-519 / OHH-520 / OHH-522 / OHH-523 / OHH-516** — `ohho.sweep/v1` (`ohho sweep plan`) plus grid, random, built-in TPE (`tpe`, `bayesian`), and multi-objective (`multi`) samplers. The sweep seed derives the sampler seed and each trial's training seed. A stopped run persists sampler state and the trial table; `ohho sweep resume <id>` continues it. The Pareto set is `ohho.sweep.leaderboard/v1` on the sampler state. ASHA and median early stopping (`ohho.train.sweep_early`) prune on intermediate metrics; `pruned` is distinct from `failed`, and resume keeps that status. The job fan-out controller is not in this package. Optional W&B backend: `to_wandb_sweep_config` maps a spec to a W&B sweep config; `ohho sweep start --backend native` (default) returns no W&B URL; `--backend wandb` creates the sweep through the W&B API when `WANDB_API_KEY` is set and returns that client's URL. Agents run `python -m ohho.train.sweep_agent`. `wandb` is an optional import. A `dataset` dimension is categorical over dataset version ids; each trial records `dataset_version_id`, a worker cache resolves each id once, and `ohho sweep group <id>` reports trials by dataset. Docs `docs/train/sweeps.md`. |
| Safe model load + worker sandbox | ✅ **OHH-485** — `ohho.model_safety` scans pickle opcodes and loads safetensors, ONNX, or `torch.load(weights_only=True)`. Managed workers (`OHHO_MANAGED_WORKER=1`) refuse other `.pt`/`.bin` pickle loads. `ohho.worker_sandbox` is a code-only container spec (read-only root, no host mounts, non-root, CPU/memory/GPU limits, wiped scratch, console API and registry egress, job token only). Human security sign-off is still required before external cloud users. Docs: `docs/security/model-files-and-worker-sandbox.md`. |
| **Navigation (native, no-ROS)** | ✅ **v1.1.0** — `ohho.nav`: grid mapping, costmap, A*, frontiers, `Navigator` (collision-aware). CLI `ohho nav goto/explore/map` |
| **Memory + perception** | ✅ **v1.1.0 + OHH-125** — `ohho.memory` (object permanence, temporal queries, JSON persistence) + `ohho.perception` (`SimPerceptor`, `CameraPerceptor`, Claude-vision `VlmPerceptor` with 1 Hz rate-limiting and cost/latency hooks). Go2 front-camera frames over DDS (`get_camera_frame()`), odom + bearing projection from image x with fixed-range assumption. CLI `ohho look`, `ohho memory`, and `ohho look unitree-go2 --transport dds://lo?domain=1`. |
| **Simulation WebSocket (`sim-serve`)** | ✅ **OHH-91 / OHH-89** — `ohho.sim_serve` JSON session shared with the Connect console: velocity/joints/deadman/estop in, odom/joint state/camera or `no_camera`/backend id out. Missing deadman or a dropped socket zeroes velocity within 300 ms (watchdog is not tied to the telemetry rate). `ros2` bridges `/cmd_vel`, `/odom`, joint state, front camera, `/emergency_stop` via rosbridge (tests use a fake bridge, no Gazebo). `isaac`/`mujoco` return `not-implemented` and the CLI prints `not built`. CLI `ohho sim-serve`. Default port **8790** (8765 is OmniBot's Foxglove bridge). |
| **Hardware self-test (`selftest`)** | ⏳ **OHH-87 code + sim done; real-robot run pending (Varun)** — `ohho.selftest`: Bench v0 bring-up suite (serial link, wheel spin direction with encoder integration, IMU 20+ Hz rate, STS3215 servo diagnostics [ID/voltage/temp/pos], camera frames) with dated JSON report + `--sim` CI mode. Real-robot bench run pending (Varun). CLI `ohho selftest` |
| **HIL harness (OHH-35)** | ⏳ **code done; bench run pending (Varun)** — `tests/hil` skips unless `OHHO_HIL=1` and an explicit port (no default `/dev/ttyUSB0`, `/dev/ttyACM0`, or `eth0`). `ohho hil --mock` / `tests/test_hil_mock.py` run the four scenarios on fake transports in CI. `docs/hil-preflight.md`, `docs/hil-bag.md` (2-min MCAP), empty `docs/hil-results.md`. CLI `ohho hil` |
| **Nightly CPU sim loop (`sim-loop`)** | ✅ **OHH-86 complete** — `ohho.sim_loop` (teleop record >=5 episodes -> LeRobot v2.0 dataset schema check + LeRobotDataset loading -> tiny ACT CPU train -> FastAPI serve -> >=50 closed-loop sim steps + latency p50/p95). CLI `ohho sim-loop` + `python scripts/sim_loop.py` + `.github/workflows/sim-loop.yml`. |
| **Go2 MuJoCo Sim Demo (`sim-demo`)** | ✅ **OHH-134 complete** — `ohho.sim_demo` (6-shot scripted demo harness: doctor -> sim connection -> typed goal -> tool-call log -> SafetyGate clamp/e-stop demo -> end card; eval JSON schema + example log; recording checklist; labeled Simulation only everywhere; video recording pending Varun). CLI `ohho sim-demo` + `python scripts/sim_demo_harness.py`. |
| **Train eval (`ohho eval`)** | ✅ **OHH-535** — `ohho eval <checkpoint> --suite gazebo-nav-regression` runs seeded kinematic-sim episodes. Proof scenario files are referenced (OmniBot `nav_regression_scenarios.yaml`), not vendored. Report schema `ohho.eval/v1` in `schemas/ohho.eval.v1.json`. Training and sweep trials call `ohho.eval.evaluate_after_train` or `run_eval_job`. Docs `docs/eval.md`. |
| **Serve examples** | ✅ **OHH-491** — `examples/serve/` CPU scripts and notebooks. CI and release run `python examples/serve/run_cpu.py`; publish waits on that job. `ohho bench validate` checks the schema fixtures. Jetson FP16 TensorRT (`ohho optimize`, OHH-429), CPU ORT INT8, vLLM bench (OHH-436), and W&B/Prometheus export stay placeholders until those commands are on main. `.github/workflows/serve-examples-gpu.yml` runs the GPU scripts nightly when a self-hosted runner is online. |
| **SafetyGate (`ohho.safety`)** | ✅ **OHH-117** — fail-closed `Transport` wrapper inserted by `Robot.connect` for every `protocol != "simulated"`: velocity/accel/joint/effort caps, 300 ms command + 500 ms state watchdogs, latched e-stop, sport-action allowlist, tilt/fault latch, hardware arming, JSONL audit. **OHH-313** delegates those decisions to a `SafetyPolicy` per `robot_class` (Ground, Arm, Legged) without changing caps or behaviour. Rules in §9a. CLI `--arm --estop-damp --safety-config` |
| **Humanoid telemetry** | ✅ **OHH-327** — `Telemetry.custom["humanoid"]` on `ohho.telemetry.v1`: mode, vendor mode (logs only), fall state, base orientation, per-group joints, planar odom shadowed onto `Telemetry.odom`, battery, experimental. Unknown or missing fall state is not upright. |
| **Humanoid MuJoCo sim** | ✅ **OHH-329** — `humanoid-mujoco` sim backend. A plugin supplies the MJCF, joint order, PD gains, observation layout and an ONNX or TorchScript checkpoint (default 50 Hz). `drive` stores a body velocity; the policy writes joint targets; PD runs every model step (`tau = kp (q_target - q) - kd dq`). `prepare`, `damp` and `custom` run with no policy. Protocol stays `simulated`, so SafetyGate is unchanged. Extra `humanoid-sim`. |
| **MCP (`ohho mcp`)** | ✅ **OHH-47** folded into the OHH-173 stdio server (`mcp_server.py` + `mcp_tools.py`): sim tools (`connect`, `drive`, `telemetry`, `nav_goto`, `look`) on `sim://`, plus `emergency_stop`. **OHH-278**: hardware URIs need a human `--allow-hardware`; the process never arms or releases an e-stop; results carry `backend` and velocity clamps. **OHH-275** `ohho mcp connect --client` merges that local server into Claude Code, Codex, Cursor, VS Code, Gemini CLI, Windsurf, and OpenCode (`.bak`, atomic write). `--hosted` stays refused until hosted MCP auth (OHH-274). Docs `docs/mcp.md` |
| **Agent rules (`ohho rules`)** | ✅ **OHH-271 phase 6** — one source `ohho/agent_rules/ohho-rules.md`; `ohho rules install --client` merges a marked block into the seven MCP clients (user scope, or `--project DIR`). `ohho doctor --agent` calls `installed_clients()`. Docs `docs/rules.md` |
| **CLI account (`login`)** | ✅ **OHH-273** — `ohho login` / `whoami` / `logout` (RFC 8628 device flow). Token in `~/.ohho/credentials.json` mode 0600 (Windows: current-user DACL). `OHHO_TOKEN` overrides the file and is never written. No CLI revoke endpoint yet (OHH-272). Docs `docs/cli.md` |
| **Run manifest (`ohho.run/v1`)** | ✅ **OHH-541** — `ohho train` writes `run_manifest.json` at `train_start` and finalises it on success, failure, and interrupt. Fields: resolved config, seeds, git commit and dirty flag, dataset version id and hash, base-weights hash, container image digest, library versions, GPU/driver/CUDA, start/end, status. `ohho run show`. **OHH-543** — `ohho run clone` tweaks that resolved recipe (`--set`, validated `ohho.recipe/v1`) and records `cloned_from`. `ohho run diff` is the shared config diff. DB storage is a local `RunStore` hook only. Docs `docs/train/run-manifest.md`. |
| **Train determinism** | ✅ **OHH-542** — `deterministic: true` seeds batch order, pins DataLoader workers, and requests torch / cuDNN deterministic algorithms. `ohho run rerun <id>` re-runs a manifest and compares final metrics. Expected variance and the within-noise tolerance: `ohho.train.variance`. Docs `docs/train/determinism.md`. |
| **Mixed precision** | ✅ **OHH-524** — recipe `precision` is `fp32`, `fp16` (CUDA GradScaler), `bf16`, or `fp8` (Transformer Engine, compute capability 8.9+). The shared runner checks the device before training and does not fall back. `ohho.run/v1` records precision, loss scaling, backend, and per-precision throughput and peak memory. Docs `docs/train/precision.md`. |
| **Pruning** | ✅ **OHH-526** — optional recipe `pruning` block: magnitude unstructured, structured channel and attention-head pruning (the checkpoint shrinks), and N:M 2:4 masks. Schedules are one-shot, gradual cubic, and iterative prune-then-fine-tune. Per-layer sparsity, parameter counts, and export sizes land on `ohho.run/v1`. The 2:4 export is `ohho.pruning.2to4/v1`. Building a TensorRT sparse engine is a GPU follow-up. Docs `docs/train/pruning.md`. |
| **Knowledge distillation** | ✅ **OHH-527** — optional recipe `distillation` block. The teacher is a registry model version or a local tiny checkpoint. The student is any imitation-learning recipe. Losses are action regression or temperature-scaled KL, plus optional feature matching, with a tunable mix weight. Teacher outputs are cached by teacher version and dataset version. The student manifest records `lineage.teacher.version`. CPU tiny models only; SmolVLA to ACT is a GPU follow-up. Docs `docs/train/distillation.md`. |
| **QAT (INT8 / INT4)** | ✅ **OHH-525** — optional recipe `qat` block: fake-quant weights and activations (per-tensor or per-channel, symmetric or asymmetric, weight-only or W/A), calibration and fine-tune lengths, Q/DQ ONNX for a linear ACT-like stack. `ohho.run/v1` records the resolved block. TensorRT engine build and full-transformer export are follow-ups. Docs `docs/train/qat.md`. |
| **Artifact retention** | ✅ **OHH-545** — per-run layout (`checkpoints/`, `logs/`, `eval/`). Default keep is best + last checkpoint and every eval report and log. `ohho run retain` deletes intermediate checkpoints of pruned trials after N days (default 14), with `--dry-run`. Pinned and promoted runs and checkpoints are never deleted. Deletions are recorded in `artifact_index.json`; `run_manifest.json` and `metrics.json` are not rewritten. `storage_used` / `storage_used_experiment` for the console. Docs `docs/train/artifacts.md`. |
| **Imitation learning recipes** | ✅ **OHH-508** — ACT and Diffusion Policy on LeRobot v2.1 datasets. Headless, headless-eval, and mixed-precision configs live under `ohho/train/examples/il/` (`act-tiny-cpu.yaml`, `act-gpu.yaml`, `diffusion-tiny-cpu.yaml`, `diffusion-gpu.yaml`). `ohho train <recipe.yaml>`. Docs `docs/train/il-recipes.md`. |
| **Optimize (`ohho optimize`)** | ✅ **OHH-429** — ONNX opset 17 (ACT-style, MLP, TorchScript, submodule selection), TensorRT FP16/INT8 with an attached entropy calibrator, ONNX Runtime sessions checked against `ohho.targets`, AWQ/GPTQ/bitsandbytes/FP8 interfaces. GPU paths fail fast. Manifest `ohho.optimize_manifest.v1`. **OHH-467** — TensorRT 10+ uses explicit Q/DQ (8/9 stay implicit), `--calib-max-images`, and a calibration-cache sidecar reused or invalidated from the model sha256 and preprocessing. **OHH-468** — `ohho.quantizers` entry points (ORT static INT8, TensorRT FP8 on Ada/Hopper, AWQ/GPTQ INT4, bitsandbytes NF4) behind `--precision`. **OHH-469** — reproducible builds: the manifest records the model sha256, calibration-cache hash, toolchain versions, host GPU, and build duration; a local cache keyed by model, target, flags, and toolchain versions; `--explain` and `--no-cache`; an engine is refused when the host GPU architecture or TensorRT version differs. |
| **Parity gate (`ohho parity`)** | ✅ **OHH-430** — `ohho.parity/v1` compares a reference model to an optimized artifact (mean abs, max abs, RMSE, cosine; action-head and classifier extras). Default fail is mean abs diff above 0.05. `ohho optimize` runs the gate before `manifest.json` and does not publish a failure unless `--allow-parity-fail` is set. Docs `docs/serve/parity.md`. |
| CLI (`ohho`) | ✅ `doctor` (`--agent`, `--robot`), `config show`, `verify --sim`, `list` (`--plugins`, `--json --ci-sim`) `version login whoami logout connect sim drive agent serve sim-serve sim-loop sim-demo train eval sweep` (`plan`, `start --backend native\|wandb\|auto`, `resume`, `group`) `market profile targets bench nav look memory selftest hil optimize parity mcp` (`mcp serve`, `mcp connect`), `rules` (`install`, `remove`, `list`, `show`), `dataset` (`build`, `pin`, `stats`), `run` (`show`, `clone`, `diff`, `retain`, `storage`), `run rerun` |
| Tests | Python 3.12, Windows, base install: `python -m unittest discover -s tests`. `tests/test_serve_examples.py` covers the OHH-491 serve examples. `tests/test_auth.py` covers the OHH-273 device flow. `tests/test_release_1_2_0.py` checks the 1.2.0 version, changelog, and wheel. `tests/test_sim_demo.py` covers the OHH-134 harness. `tests/test_safety_policy.py` covers the OHH-313 class policies. `tests/test_config.py` covers the OHH-314 robot-id resolver. `tests/test_eval.py` covers the OHH-535 checkpoint eval runner. `tests/test_humanoid_telemetry.py` covers the OHH-327 humanoid telemetry block. `tests/test_humanoid_mujoco.py` covers the OHH-329 humanoid-mujoco backend and skips when MuJoCo is absent. `tests/test_ohho_booster.py` covers the booster plugin. `tests/test_booster_t2_walk.py` covers the OHH-331 headless walk and skips when MuJoCo or the pinned assets are absent. `tests/test_conformance.py` covers the OHH-315 kit. `tests/test_ppo_recipe.py` covers the OHH-510 PPO recipe. `tests/test_hil_mock.py` covers the OHH-35 mock scenarios. `tests/test_optimize.py` covers OHH-429 and OHH-467. `tests/test_quantizers.py` covers OHH-468. `tests/test_repro_builds.py` covers OHH-469. `tests/test_parity.py` covers OHH-430. `tests/test_bench_schema.py` covers ohho.bench/v1. `tests/test_bench_harness.py` covers the OHH-431 harness. `tests/test_bench_generate.py` covers the OHH-436 generate-mode bench. `tests/test_bench_sweep.py` covers the OHH-476 serving sweep. `tests/test_bench_cost.py` covers the OHH-477 cost model. `tests/test_serve_backends.py` covers the OHH-475 serving adapters. `tests/test_sweep.py`, `tests/test_sweep_sampler.py`, `tests/test_sweep_resume.py`, `tests/test_sweep_early.py`, `tests/test_sweep_early.py`, and `tests/test_sweep_wandb.py` cover `ohho.sweep/v1` plans, samplers, derived seeds, resumable runs, ASHA / median early stopping, and the optional W&B backend. `tests/test_train_recipe.py` covers `ohho.recipe/v1` and the recipe runner. `tests/test_run_manifest.py` covers `ohho.run/v1` manifests and `ohho run show`. `tests/test_run_clone.py` covers the OHH-543 clone and config diff. `tests/test_precision.py` covers OHH-524 mixed precision. `tests/test_artifacts.py` covers OHH-545 retention. `tests/test_qat.py` covers the OHH-525 QAT block, fake quant, and Q/DQ ONNX export. `tests/test_determinism.py` covers the OHH-542 flag, re-run comparison, and CPU ACT exact re-run (skipped when torch is absent). `tests/test_distillation.py` covers OHH-527 knowledge distillation. `tests/test_pruning.py` covers OHH-526 pruning. Hardware HIL stays skipped unless `OHHO_HIL=1`. |
| CI | `.github/workflows/ci.yml` (multi-OS, Py 3.10-3.13; includes `ohho selftest --sim` smoke check and a ≥95% line+branch coverage gate on `ohho/safety.py`; installs `plugins/ohho-booster`). Ubuntu + Python 3.12 also installs `[humanoid-sim]` after a CPU torch wheel so the humanoid-mujoco tests run, and caches and fetches the pinned description archive and walk checkpoint. That same torch install runs the OHH-542 tiny ACT deterministic re-run; other cells skip it when torch is absent. Other cells skip those tests when the packages or assets are absent. `.github/workflows/sim-loop.yml` is the nightly CPU sim loop. Serve CPU examples run via `python examples/serve/run_cpu.py` (OHH-491). |

**M1 (two-robot hardware vertical slice) — software complete.** All four adapters
(`yahboom`, `feetech`, `composite`, `unitree`) are built and unit-tested against
the protocols with mock/loopback bytes (no hardware). The Unitree DDS state
subscription is wired (polling reader thread → `_on_state` → `Telemetry`). The
OmniBot arm adapter (Feetech bus) is built and composes with the Yahboom base via
`CompositeTransport`. What remains is **on-robot bring-up** only — running the HIL
tests against real OmniBot + Go2 hardware. See §8.

---

## 1. What OhhO OS is (vision & positioning)

OhhO OS is the **engine** beneath the OhhO product family — it is **not** a
product itself and does **not** rename or absorb the products (Build, Train,
etc.). It is the open-core that those consoles run on. Open-core model:

```
   ohho.com           OhhO Cloud (the ~19 paid consoles)         ← Stripe tiers
                                 │ same APIs
   pip install →      OhhO OS  (this package; Apache-2.0)  ← you are here
```

**The four differentiators (competitor-agnostic, as advertised on `/os`):**
1. **ROS optional — never required.** Native runtime *or* ROS 2, switch with one
   argument. (Most agentic robot stacks force one or the other.)
2. **Truly robot-agnostic.** Write a behavior once; run it on a wheeled base, a
   quadruped, a humanoid, or an arm. Swapping hardware ≠ rewriting code.
3. **The whole lifecycle, one engine.** Not just control — perception, data,
   training, sim, fleet, safety.
4. **Open source, zero lock-in.**

Marketing + user docs already shipped: `website/app/os/page.tsx` and
`website/docs/ohho-os/*.md` (install, quickstart, runtimes, robots, training,
architecture). **Keep the SDK and those docs in sync.**

---

## 2. Architecture

The core idea: **the robot and the middleware are swappable backends, not forks
of your code.**

```
   your code  ·  Agent (Mind)  ·  Train / Data / Serve     ← runtime-agnostic
            │
   ┌────────┴─────────┐  Robot Abstraction Layer (RAL = ohho.robot.Robot)
   │  Runtime (port)  │  pub/sub · timers · params
   └────────┬─────────┘
      ┌──────┴───────┐
  native runtime    ros2 runtime (stub)
  (no ROS)          (rclpy + ROS 2 stack)
  asyncio/threads   Nav2 · SLAM · MoveIt 2
  direct drivers    DDS multi-machine
            │
      ┌─────┴──────┐  Adapters (one per robot family)
   sim · yahboom · unitree · dji · …   (only `sim` exists today)
```

**THE ONE INVARIANT RULE (do not break):**
> Engines (Robot, agent, future train/perception) depend **only** on the
> `Runtime` port and the `Transport` interface — they must **never import
> `rclpy`** or a specific adapter directly. This is what makes "switch runtime /
> robot with one argument" true. New backends/adapters implement the interface;
> nothing above them changes.

**Capability-typed commands.** Commands are gated by capability. A robot that
lacks a capability turns the call into a safe no-op (e.g. `move_joints` on a
robot with no arm) instead of crashing. This is what makes one behavior portable
across heterogeneous hardware.

---

## 3. Current file map (`sdk/`)

```
sdk/
├── pyproject.toml        # name=ohho-os; extras: serial/arm/agent/data/unitree/dji/serve/train/wandb/ros2/yaml/dev/all; console_script `ohho`
├── README.md             # user-facing quickstart
├── AGENTS.md             # ← THIS FILE (handoff + roadmap)
├── .gitignore            # egg-info, __pycache__, caches
├── ohho/
│   ├── __init__.py       # exports Robot, connect, schema types, registry fns; __version__
│   ├── schema.py         # Velocity, Odometry, JointReading, Telemetry, TransportStatus, ConnectionState, clamp()
│   ├── capabilities.py   # capability string constants + ALL + is_known()
│   ├── capability_params.py  # typed parameters for those capabilities (OHH-312)
│   ├── registry.py       # RobotSpec v2 (frozen dataclass) + _BUILTINS + get_spec/list_specs/register/load_manifest
│   ├── robot_spec.py     # RobotClass, per-class limits, assets, CiSim, category table (OHH-309)
│   ├── transport.py      # Transport (ABC) + BaseTransport (callback bookkeeping)
│   ├── runtime/
│   │   ├── __init__.py   # get_runtime("auto"|"native"|"ros2"), available_runtimes()
│   │   ├── base.py       # Runtime (ABC): pub/sub + params (concrete) + now/create_timer/start/stop (abstract); TimerHandle; RuntimeUnavailable
│   │   ├── native.py     # NativeRuntime — threaded scheduler
│   │   └── ros2.py       # Ros2Runtime — rclpy node (timers, pub/sub, params); node_factory injectable for tests
│   ├── adapters/
│   │   ├── __init__.py        # resolve_transport: plugin registries, then scheme dict
│   │   ├── errors.py          # AdapterUnavailable
│   │   ├── sim.py             # SimTransport — holonomic physics + joint servoing; step(dt) deterministic
│   │   ├── feetech.py         # FeetechTransport — feetech:// (SO-101 arm); lerobot lazy; rad↔tick; torque; reader thread
│   │   ├── composite.py       # CompositeTransport — merges base + arm into one Transport (mobile-manipulators)
│   │   ├── ros2.py            # Ros2Transport — ros2:// (bridges /cmd_vel, /odom, /joint_states); msg_factory injectable
│   │   └── humanoid_mujoco.py # HumanoidMujocoTransport — any plugin MJCF, policy, then PD; mujoco imported on connect
│   ├── humanoid.py       # generic humanoid modes and Robot methods (posture, gaze, hands, joints)
│   ├── humanoid_telemetry.py  # custom["humanoid"] block (ohho.telemetry.v1): validator, JSON, odom shadow
│   ├── humanoid_sim.py   # HumanoidSimConfig, observation layout, bind_sim_config (stdlib; no mujoco import)
│   ├── policy_runner.py  # LocomotionPolicy — ONNX or TorchScript on CPU; runtime imported on load
│   ├── robot.py          # Robot (RAL) — connect/drive/move_joints/telemetry/emergency_stop/has/disconnect; connect() shortcut
│   ├── agent.py          # Agent + ScriptedBrain (fallback); _default_brain() lazily loads HarnessBrain
│   ├── brains.py         # HarnessBrain — wires agent_engine (ToolRegistry from caps, RobotPerceptor, ClaudeToolCallingReasoner)
│   ├── data/             # ohho.data — Recorder (capture state+action), LeRobot v2.0 writer + reader
│   │   ├── __init__.py   # exports Recorder, Episode
│   │   ├── recorder.py   # Recorder — wraps Robot, intercepts drive/move_joints, captures frames
│   │   ├── writer.py     # write_dataset — LeRobot v2.0 (Parquet via pyarrow, JSON Lines fallback)
│   │   └── reader.py     # DatasetReader — lightweight reader (no torch needed), stats()
│   ├── datasets/         # dataset versions (OHH-513) plus splits and train-time augmentation (OHH-515)
│   │   ├── __init__.py   # build/pin/stats, assign_splits, apply_augmentation, preview_augmentation
│   │   ├── model.py      # DatasetVersion, content hash, JSON Schema
│   │   ├── build.py      # materialise a version from local episodes
│   │   ├── split.py      # seeded episode, session/day, task, and scene splits
│   │   ├── augment.py    # pure-Python apply and preview for recipe loaders
│   │   ├── store.py      # local version directory
│   │   ├── cli.py        # ohho dataset build|pin|stats
│   │   └── dataset_version.schema.json
│   ├── train/            # ohho.train — finetune() delegates to lerobot_engine, mock mode for sim
│   │   ├── __init__.py   # finetune(), SUPPORTED_POLICIES, TrainUnavailable, _mock_train()
│   │   ├── run_manifest.py # ohho.run/v1 manifest, validator, ohho run show
│   │   ├── determinism.py  # deterministic flag: seeded order, fixed workers, torch/cuDNN
│   │   ├── variance.py     # expected variance and within-noise tolerance per recipe
│   │   ├── rerun.py        # ohho run rerun
│   │   ├── run_clone.py  # ohho run clone / diff; cloned_from; shared config diff
│   │   ├── precision.py  # fp32 / fp16 loss scaling / bf16 / fp8 capability check (OHH-524)
│   │   ├── artifacts.py  # per-run artifact layout, retention, storage_used (OHH-545)
│   │   ├── compression/  # optional train-time compression methods
│   │   │   └── pruning.py # magnitude, structured, and 2:4 pruning (OHH-526)
│   │   ├── compression/
│   │   │   └── distillation.py  # teacher reference, output cache, tiny CPU student (OHH-527)
│   │   ├── act.py        # TinyACTPolicy + train_act() CPU Action Chunking Transformer
│   │   ├── ppo.py        # OHH-510 PPO trainer, optional rsl_rl/skrl, ONNX actor export
│   │   ├── rl_tasks.py   # ohho.rl_tasks: cartpole, mujoco-inverted-pendulum, isaac-velocity
│   │   ├── vla/          # OHH-509 SmolVLA + OpenVLA: params, memory estimate, licence, LoRA merge
│   │   ├── sweep.py      # ohho.sweep/v1 schema, validator, grid/random expander
│   │   ├── sweep_early.py # ASHA and median early stopping on intermediate metrics (OHH-520)
│   │   ├── sweep_seed.py # sampler seed and per-trial training seed derived from the sweep seed
│   │   ├── sweep_run.py  # ohho.sweep.run/v1 persisted sampler state and trial table
│   │   ├── sweep_sampler.py  # TPE, multi-objective Pareto set, resumable sampler state
│   │   └── sweep_dataset.py  # dataset-version cache and ohho sweep group
│   │   ├── sweep_wandb.py  # ohho.sweep/v1 → W&B config; native or W&B launch
│   │   └── sweep_agent.py  # one W&B trial as an OhhO train job (CPU)
│   ├── compression/      # compression passes; each pass is its own module
│   │   └── qat.py        # QAT fake-quant, recipe block, Q/DQ ONNX (OHH-525)
│   ├── serve/            # ohho.serve — FastAPI policy server; OHH-475 backends/ and loadgen.py
│   │   ├── __init__.py   # serve(), build_app(), ServeUnavailable, _MockModel
│   │   └── act.py        # ACTModel inference adapter
│   ├── bench/            # ohho.bench — ohho.bench/v1 schema (OHH-473), Serve harness (OHH-431), pipeline bench (OHH-474), sensor sampler (OHH-472), generate mode (OHH-436), LLM sweep (OHH-476), cost model (OHH-477)
│   ├── bench/            # ohho.bench — ohho.bench/v1 schema (OHH-473), Serve harness (OHH-431), pipeline bench (OHH-474), sensor sampler (OHH-472), generate mode (OHH-436), LLM sweep (OHH-476), export plugins (OHH-498)
│   ├── model_safety.py   # OHH-485 safe weight load: safetensors, ONNX, pickle scan, weights_only
│   ├── worker_sandbox.py # OHH-485 code-only worker sandbox spec, scratch wipe, egress check
│   ├── sim_serve.py      # ohho.sim_serve — WebSocket teleop and telemetry server (`sim` and `ros2`)
│   ├── sim_loop.py       # ohho.sim_loop — record -> validate -> tiny ACT train -> serve -> eval pipeline
│   ├── sim_demo.py       # ohho.sim_demo — OHH-134 6-shot Go2 sim demo harness + eval log
│   ├── selftest.py       # ohho.selftest — hardware self-test suite (motors, encoders, IMU, servos, camera)
│   ├── hil.py            # ohho.hil — HIL runner, fake-transport scenarios, results markdown (OHH-35)
│   ├── safety.py         # ohho.safety — SafetyGate(Transport), SafetyProfile, AuditLog, arming policy
│   ├── safety_policy.py  # SafetyPolicy per robot class (Ground, Arm, Legged); plugin caps may only tighten (OHH-313)
│   ├── profiles.py       # HardwareProfile, detect_profile(), `ohho profile` CLI
│   ├── optimize/         # ohho optimize — ONNX, TensorRT, ORT, quantizer plugins, reproducible builds (OHH-429, OHH-467, OHH-468, OHH-469)
│   ├── parity/           # ohho parity — numerical gate, ohho.parity/v1 (OHH-430)
│   ├── market.py         # Skill registry, @skill decorator, run_skill(), `ohho market` CLI
│   ├── hardware.py       # resolve_device("auto") -> cuda/mps/cpu (lazy torch)
│   ├── cli.py            # argparse: doctor (--agent, --robot), config, verify, list(--plugins, --json --ci-sim), version, login, whoami, logout, connect, sim, drive, agent, serve, sim-serve, sim-loop, sim-demo, train, eval, sweep, dataset, bench, run, selftest, hil, optimize, parity, mcp, rules; main()
│   ├── robots/
│   │   ├── __init__.py   # package for the built-in plugin and example manifests
│   │   ├── builtin.py    # built-in plugin (sim, aerial); vendor SDKs stay lazy
│   │   ├── example.json  # example manifest (nested dof/limits) for load_manifest()
│   │   ├── omnibot/      # OHH-318: spec, serial:// composite, ground+arm policy, descriptor, doctor
│   │   └── unitree/      # OHH-319: quadruped spec, dds://, MuJoCo backend, legged policy, descriptor, doctor
│   └── testing/
│       ├── __init__.py
│       └── conformance/  # class scenarios + vendor-name grep (OHH-315); SafetyGate checks use the per-class SafetyPolicy
├── plugins/
│   └── ohho-booster/     # separate distribution: spec, mode map, pinned assets, t2_walk sim config (OHH-330, OHH-331)
├── docs/
│   ├── architecture/
│   │   └── robot-plugins.md  # ADR: six plugin rules + RobotSpec v2 (OHH-309, proposed)
│   ├── safety.md         # SafetyGate rules, profiles, config file, audit format
│   ├── security/
│   │   └── model-files-and-worker-sandbox.md  # OHH-485 load rules, sandbox spec, review checklist
│   ├── selftest.md       # Bench v0 self-test documentation, safety rules, JSON schema
│   ├── hil-preflight.md  # bench checklist before hardware HIL (OHH-35)
│   ├── hil-bag.md        # 2-minute MCAP recording on the Pi 5 Jazzy image
│   ├── hil-results.md    # empty HIL results template; the hardware runner fills it
│   ├── sim-demo.md       # OHH-134 sim demo recording checklist and eval log schema
│   ├── eval.md           # ohho eval command and ohho.eval/v1 report (OHH-535)
│   ├── mcp.md            # stdio MCP server + `ohho mcp connect` client configs
│   ├── rules.md          # `ohho rules install` paths, markers, and safety text
│   ├── cli.md            # CLI reference; ohho login / whoami / logout and whoami --json schema
│   ├── serve/parity.md   # ohho.parity/v1 report (OHH-430)
│   ├── unitree.md        # Go2 DDS adapter — sim-tested only; hardware OHH-135
│   └── dataset-versions.md  # dataset version layout, splits, augmentation, preview shape
├── examples/
│   └── serve/            # OHH-491 CPU serve examples, notebooks, release gate
├── scripts/
│   ├── check_classifiers.py
│   ├── hil_run.py        # one-command HIL runner (delegates to ohho.hil)
│   ├── sim_loop.py       # one-liner reproducible sim loop runner
│   └── sim_demo_harness.py  # OHH-134 sim demo harness entry point
├── schemas/
│   └── ohho.eval.v1.json # ohho.eval/v1 checkpoint eval report (OHH-535)
└── tests/                # unittest suite including test_eval.py, test_sim_loop.py, test_sim_serve.py, test_sim_demo.py, test_selftest.py, test_humanoid_telemetry.py, test_humanoid_mujoco.py, test_conformance.py, test_hil_mock.py, test_serve_backends.py, test_artifacts.py, and tests/fixtures/humanoid (small MJCF, no meshes)
```

---

## 4. Public API quick reference (as built — keep stable)

```python
from ohho import Robot, connect

bot = Robot.connect("omnibot")                 # transport=None -> spec adapter, falls back to sim
bot = Robot.connect("omnibot", "sim://")       # explicit transport URI
bot = Robot.connect("omnibot", runtime="native")  # "auto" | "native" | "ros2"

bot.drive(vx=0.2, vy=0.0, w=0.3)               # clamped to spec limits; vy ignored if not holonomic; no-op if no base.drive
bot.stop()
bot.move_joints([0, -0.5, 0.5, 0, 0, 0.2])     # no-op unless capabilities include "manipulation"
bot.prepare()                                  # no-op without humanoid.posture; walking speed stays drive()
bot.move_joints("left_arm", [0.0, -0.2])       # whole_body.joint; refused unless CUSTOM, sim, and policy allow
bot.emergency_stop(); bot.release_stop()
t = bot.telemetry()                            # Telemetry(odom, joints, battery, custom, timestamp)
bot.has("manipulation")                        # capability check
bot.status()                                   # TransportStatus
bot.disconnect()                               # also: `with Robot.connect(...) as bot:`

from ohho.agent import Agent
Agent(bot).run("explore the room")             # returns list[str] log lines (scripted brain for now)
```

**Schema (`ohho/schema.py`):** `Velocity(linear_x, linear_y, angular_z)`,
`Odometry(x, y, theta, vx, vy, omega)`, `JointReading(name, position, velocity?)`,
`Telemetry(odom?, joints[], battery?, custom{}, timestamp)`,
`TransportStatus(protocol, state, label, latency_ms, msg_rate, error?, connected_since?)`,
`ConnectionState` enum, `clamp(value, lo, hi)`.
`custom["humanoid"]` (`ohho.telemetry.v1`, `ohho.humanoid_telemetry`) carries
`mode`, `vendor_mode`, `fall_state`, `base_rpy`, `base_ang_vel`, per-group
`joints` (`q`, `dq`, `tau`, `temp` where reported), planar `odom` (shadowed
onto `Telemetry.odom`), `battery` (fraction, voltage) and `experimental`.
`is_upright()` is false for `unknown` and for a missing fall state.

**RobotSpec (`ohho/registry.py`, v2 in `ohho/robot_spec.py`):** existing fields
`id, name, category, capabilities(tuple), adapter, dof_base, dof_arm, max_lin,
max_ang, joint_names(tuple)`, `.has(cap)`, plus `robot_class`, `joint_groups`,
`limits`, `assets`, `ci_sim`, `source`, `capability_params`, `sim_backend`
(empty keeps today's `sim://`). `category` still
loads; it maps onto `robot_class`. Shipped specs: `omnibot` (mobile manipulator
plugin, 6-DOF arm, max_lin 0.2), `unitree-go2` (quadruped plugin, max_lin 1.5, `dds://`), `sim`
(mobile base, generic holonomic). Safety caps are unchanged. ADR:
`docs/architecture/robot-plugins.md`.

**Robot id (`ohho/config.py`):** `resolve_robot_id` returns the id and a source
of `flag`, `env`, `config`, or `default`. Precedence is the explicit argument,
then `OHHO_ROBOT`, then `robot` in `~/.ohho/config.toml` (`OHHO_HOME` relocates
that directory for tests), then `sim`. `load_robot_profile` reads
`~/.ohho/robots/<id>.toml` and a repo-local `ohho.toml`. A profile that raises
any safety cap is refused at load. `Robot.connect` applies a tighter override
through the existing `SafetyConfig`. The gate selects `SafetyPolicy` from that
resolved spec's `robot_class`. MCP
callers use `ohho.session.RobotSession`, which sends only the resolved id.

**Robot plugins (`ohho/plugins.py`):** `RobotPlugin` declares `specs()`,
`transports()` (scheme → factory), `sim_backends()`, `safety_policy(spec)`
(`None` = class default; `SafetyGate` loads it and refuses a looser cap), `descriptor(spec)`
(`None` until C-04 / OHH-214; the built-in hook returns ``None``. RobotDescriptor v1 is `ohho.descriptor`) and `doctor(spec, uri, read_only=True)`.
Discovery uses `importlib.metadata.entry_points(group="ohho.robots")`. ohho-os
registers `builtin = ohho.robots.builtin:plugin`. Plugins must not import a
vendor SDK at module import. `ohho list --plugins` prints each plugin, its
specs and any load error. A duplicate robot id, transport scheme or sim
backend is a load error naming both plugins. `resolve_transport` uses the
factories from plugins that loaded, then the scheme dict.

**Capabilities (`ohho/capabilities.py`, parameters in `ohho/capability_params.py`):**
`base.drive`, `base.holonomic_drive`, `legged.pose`, `aerial`,
`humanoid.locomotion`, `humanoid.posture`, `whole_body.joint`, `manipulation`,
`manipulation.dual_arm`, `end_effector.gripper`, `end_effector.dex_hand`,
`head.gaze`, `perception.rgb`, `perception.depth`, `perception.scan`. `has()`
stays a string check. Typed parameters live on `spec.capability_params` and are
descriptive; they are not safety caps. `register()` rejects an unknown name.

**Transport interface (`ohho/transport.py`):** `connect()`, `disconnect()`,
`status()`, `read()->Telemetry`, `send_velocity(Velocity)`,
`send_joint_command(name, position)`, `emergency_stop()`, `release_stop()`,
`on_telemetry(cb)->unsub`, `on_status(cb)->unsub`. Subclass `BaseTransport` (it
does the callback bookkeeping). **New adapters = subclass `BaseTransport`.**

**Runtime port (`ohho/runtime/base.py`):** `subscribe/publish` (concrete),
`get_param/set_param` (concrete), `now()`, `create_timer(period_s, cb)->TimerHandle`,
`start()`, `stop()` (abstract). **New runtimes = subclass `Runtime`.**

---

## 5. Develop / run / test

```bash
# from repo root
pip install -e .                         # editable, dependency-free base
# (add extras as they gain code: pip install -e '.[unitree]')

python3 -m unittest discover -s tests
ohho doctor                              # environment / runtimes / adapters / robots
ohho sim --robot omnibot --seconds 5     # drive a pattern in simulation
ohho agent omnibot "explore the room"

# Lint exactly like CI (run from repo root — see §7):
pip install ruff
ruff format --check --exclude robot_ws/src/omnibot_firmware .
ruff check        --exclude robot_ws/src/omnibot_firmware .
```

---

## 6. Conventions & invariants (follow these)

- **Dependency-free `base`.** Core (`schema`, `capabilities`, `registry`,
  `transport`, `runtime/native`, `adapters/sim`, `robot`, `cli`) must import only
  the standard library. Anything heavier (torch, rclpy, cyclonedds, pymavlink,
  numpy, fastapi) goes behind an **extra** and is **imported lazily** inside the
  component that needs it (see `hardware.resolve_device` and
  `registry.load_manifest` for the pattern).
- **Style:** `from __future__ import annotations` at the top of every module;
  full type hints; docstrings on public classes/functions. **Code must pass
  `ruff format` and `ruff check`** (CI runs both repo-wide — see §7). After
  editing, run `ruff format ohho tests` before committing.
- **Tests:** stdlib `unittest` (no pytest dependency), under `tests/`,
  deterministic (build `Robot` over a `SimTransport` and call `step(dt)`
  manually rather than relying on the live thread — see `tests/test_robot.py`).
  Add tests with every new module.
- **The invariant rule from §2** — engines depend only on `Runtime`/`Transport`.
- **Keep `RobotSpec` constants synced** with the real platform: OmniBot arm joint
  names mirror the ROS arm driver; OmniBot `max_lin=0.2` mirrors the Yahboom
  driver clamp. If you change one, change both and note it here.
- **Keep website docs in sync** (`website/docs/ohho-os/*`, `website/app/os`).

---

## 7. CI / git notes

- This repository's workflow is `.github/workflows/ci.yml`. It installs with
  `pip install -e .`, runs `python -m unittest discover -s tests`, then
  `ohho doctor` and `ohho sim --robot omnibot --seconds 2`, on Ubuntu, macOS,
  and Windows for Python 3.10, 3.11, 3.12, and 3.13. The same workflow runs
  `python examples/serve/run_cpu.py` (OHH-491).
- `.github/workflows/release.yml` runs that CPU suite and the publish job
  needs it, so a broken example fails the release. `.github/workflows/serve-examples-gpu.yml`
  runs the GPU examples nightly when a self-hosted runner labeled `linux`
  and `gpu` is online.
- `.github/workflows/sim-matrix.yml` (OHH-316) is additive. It does not replace
  `ci.yml`. Rows come from `ohho list --json --ci-sim`.
- There is no PyPI publish workflow here (that is OHH-20).
- Editable-install artifacts (`ohho_os.egg-info`, `__pycache__`) are gitignored.
- The notes that used to live in this section described OmniBotPro's ROS
  workflow (`ros2_ci.yml`, Vercel, `sdk/` paths). They do not apply to this
  repository.

---

## 8. Roadmap (M0 → M5)

Estimates are split into **software time** (compresses well with an AI agent) and
**hardware-gated time** (needs the physical robots on a bench — does *not*
compress). M0 is done.

### ✅ M0 — Scaffold + sim slice (DONE, merged)
RAL, native runtime, sim adapter, registry, CLI, 40 tests. Acceptance: met.

### ✅ M1 — Two-robot hardware vertical slice (SOFTWARE COMPLETE)
**Goal:** the *same* code drives two very different real robots, proving
robot-agnosticism. Pair: **OmniBot** (wheeled, USB serial) + **Unitree Go2**
(quadruped, DDS).

**Done (v0.3.0):**
- `ohho/adapters/yahboom.py` — `YahboomTransport(BaseTransport)` over pyserial,
  `serial://` scheme. Reuses the vendored codec `_yahboom_proto.py`. Reader thread
  decodes `0xFB` packets → odometry integration + IMU yaw fusion. Clean
  `AdapterUnavailable` when `[serial]` missing.
- `ohho/adapters/feetech.py` — `FeetechTransport(BaseTransport)` for the SO-101
  6-DOF arm, `feetech://` scheme. Lazy `lerobot` import (`[arm]` extra). rad↔tick
  conversion (4096 ticks/rev, home 2048), joint clamping, torque enable/disable,
  reader thread for present positions. Fake bus injectable for tests.
- `ohho/adapters/composite.py` — `CompositeTransport(base, arm)` merges a base
  link + arm link into one `Transport`: `send_velocity`→base,
  `send_joint_command`→arm, `read`→merged telemetry, estop→both. This is what
  lets a single `Robot` drive the full OmniBot mobile-manipulator.
- `ohho/robots/unitree/dds.py` — `UnitreeDdsTransport(BaseTransport)` over
  `unitree_sdk2py`, `dds://` scheme. **DDS state subscription wired**: polling
  reader thread calls `SportClient.GetState(SportModeState)` → `_on_state` →
  `Telemetry` at 20 Hz. `state_factory` injectable for mock tests. Maps
  `Velocity` → `SportClient.Move(vx, vy, vyaw)`.
- `resolve_transport` auto-composes: `serial://<base>,<arm>` for a robot with
  `manipulation` capability returns a `CompositeTransport(Yahboom, Feetech)`.
- Extras: `[serial]` (pyserial), `[arm]` (lerobot), `[unitree]` (cyclonedds).
- **Mock/loopback tests** (27 new, 85 total): `test_feetech.py`,
  `test_composite.py`, updated `test_unitree.py` + `test_adapters.py`.
- **HIL test harness**: `tests/hil/test_hil.py` — 4 tests (OmniBot base,
  OmniBot arm, OmniBot composite, Go2 DDS), skipped unless `OHHO_HIL=1` and
  the matching device variable is set (no default port). Fake-transport
  twins live in `tests/test_hil_mock.py`. Bench entry is `ohho hil`.

**Remaining (hardware-gated only):** on-robot bring-up against the real OmniBot
+ Go2 — checklist in `docs/hil-preflight.md`, then `ohho hil --hardware
--preflight-ok` with `OHHO_HIL=1` and explicit ports. DJI / MAVLink adapter
not started.

### ✅ M2 — Real agent brain (DONE)
Wired `ohho.agent._default_brain()` to **`agent_engine`** via the new
`ohho/brains.py` module. `HarnessBrain` builds a `ToolRegistry` from the robot's
capabilities (drive, move_joints, stop, emergency_stop, get_telemetry,
get_status), creates a `RobotPerceptor` that turns `Robot.telemetry()` into a
`WorldState`, and runs a `ClaudeToolCallingReasoner` over a `ReasoningRouter`
(cloud Claude when `ANTHROPIC_API_KEY` + `anthropic` present, echo fallback
otherwise). `ScriptedBrain` remains the no-dependency fallback when
`agent_engine` isn't installed. Acceptance met: `Agent(bot).run(goal)` performs
a real perceive→reason→act→reflect loop; 11 new tests in `test_brains.py`.

### ✅ M3 — Training & data pipelines (DONE)
Built `ohho.data`, `ohho.train`, `ohho.serve` — the record→train→serve loop.

- **`ohho.data.Recorder`** — wraps a `Robot`, intercepts `drive()` and
  `move_joints()` to capture actions, polls `telemetry()` for state. Frames are
  9-D (6 arm + 3 base) for mobile-manipulators, 3-D for base-only robots. Manual
  mode (`start_episode`/`capture_frame`/`stop_episode`) and timer mode
  (`record_episode`/`stop_recording`). Writes LeRobot v2.0 layout (Parquet via
  pyarrow `[data]` extra, JSON Lines fallback without it) with `info.json`,
  `tasks.jsonl`, `episodes.jsonl`.
- **`ohho.data.DatasetReader`** — lightweight reader (no torch needed) that
  loads meta + episode data (Parquet or JSON Lines), provides `stats()`,
  `iter_episodes()`, `all_frames()`.
- **`ohho.train.finetune()`** — delegates to `lerobot_engine` when torch + lerobot
  are installed (`[train]` extra). Supports `smolvla`, `act`, `diffusion`,
  `openvla`. `mock=True` writes a dummy checkpoint from dataset stats (no GPU
  needed — for the sim acceptance loop).
- **`ohho.serve`** — `build_app()` returns a FastAPI inference server
  (testable with `TestClient`); `serve()` launches it via uvicorn. Endpoints:
  `GET /health`, `POST /load_model`, `POST /predict`. `mock_model=True` uses a
  no-op model (for the sim loop). Added `ohho serve` CLI command.
- **Acceptance test** (`test_data.py::TestEndToEndRecordTrainServe`): records 10
  frames from a sim OmniBot → mock-trains a smolvla checkpoint → launches a mock
  serve app → `POST /predict` returns a 9-D action vector. 12 new tests.

### ✅ M4 — ROS 2 runtime backend (DONE)
Implemented `Ros2Runtime` (rclpy node behind the existing `Runtime` port) and
`Ros2Transport` (bridges the unified Transport to ROS 2 topics).

- **`Ros2Runtime`** (`runtime/ros2.py`) — creates an rclpy node (lazy import,
  injectable `node_factory` for tests). Overrides `subscribe`/`publish` to bridge
  to ROS 2 topics (JSON-encoded `std_msgs/String` for the generic bus).
  Overrides `get_param`/`set_param` to use the node parameter API.
  `create_timer` uses rclpy timers. `start()` spins the node in a background
  thread; `stop()` cleans up timers, publishers, subscriptions, and shuts down
  rclpy.
- **`Ros2Transport`** (`adapters/ros2.py`) — bridges to real ROS 2 topics:
  publishes `geometry_msgs/Twist` on `/cmd_vel`, subscribes to
  `nav_msgs/Odometry` on `/odom`, subscribes to `sensor_msgs/JointState` on
  `/arm/joint_states`, publishes joint commands on `/arm/joint_commands`.
  Supports namespace prefixes (`ros2:///robot1`). All message types are
  imported lazily; `msg_factory` injectable for tests.
- **`get_runtime("auto")`** — now prefers `ros2` when rclpy is available, falls
  back to `native` otherwise. Same code, zero changes above the port.
- **`ros2://` scheme** registered in `resolve_transport`; `available_adapters()`
  probes for `rclpy`.
- **Tests** (`test_ros2.py`, 21 new): `FakeRos2Node` simulates the full rclpy
  node interface — runtime pub/sub, params, timers, transport publishers/
  subscribers, odometry + joint state callbacks, namespace prefix, estop, auto
  runtime selection. All run without ROS 2 installed.
- **Acceptance met:** `runtime="ros2"` runs the identical Robot/agent/train code
  with zero changes above the port. `Robot.connect("omnibot", "ros2://",
  runtime="ros2")` drives through `/cmd_vel` and reads `/odom`.

### ✅ M5 — Packaging, profiles & community (DONE)
- **`ohho.profiles`** — `HardwareProfile` + `NodeSpec` with 5 built-in profiles
  (`pi_workstation`, `jetson_single`, `workstation_single`, `mac_dev`,
  `edge_cpu`). `detect_profile()` auto-detects via `OHHO_HW_PROFILE` env var or
  platform probes. `describe()` for logs/W&B. Added `ohho profile list/show/detect`
  CLI.
- **`ohho.market`** — skill registry: `@skill` decorator, `register_skill()`,
  `list_skills()`, `run_skill()`. 4 built-in skills (`patrol`, `wave`, `stop`,
  `status`). Capability-gated execution (`SkillRequirementsNotMet` if the robot
  lacks required caps). Added `ohho market list/run` CLI.
- **`CONTRIBUTING.md`** — "add a robot = one manifest + one adapter + one test"
  guide with code examples, the architecture invariants, and a PR checklist.
- **Root `CLAUDE.md`** — updated to document the `sdk/` area.
- **Version bumped to `1.0.0`** — all milestones (M0–M5) complete.
- 22 new tests (`test_market.py` + `test_profiles.py`).
- **PyPI:** not published from this repository. OHH-20 covers trusted
  publishing. Do not add a publish workflow here.

Post-M5 train reproducibility on this tree: OHH-541 run manifests
(`docs/train/run-manifest.md`); OHH-545 artifact retention
(`ohho run retain`, `docs/train/artifacts.md`).

### ✅ Post-M5 — checkpoint eval (OHH-535)
- **`ohho eval`** runs N seeded episodes per scenario on the CPU kinematic sim.
  The Proof Gazebo nav regression suite is referenced from
  `ohho/eval/suites.json` (set `OHHO_PROOF_ROOT` to that checkout, or pass
  `--scenarios`) and is not copied into this repo. The report is
  `ohho.eval/v1` (`schemas/ohho.eval.v1.json`): Wilson 95% CI, episode length,
  failure-mode counts, optional trace paths. `evaluate_after_train` and
  `run_eval_job` are the hooks a training run or sweep trial calls.

### ✅ Post-M5 — Serve examples (OHH-491)
- **`examples/serve/`** is the runnable Serve set: Jetson FP16 TensorRT, CPU
  ORT INT8, a vLLM bench, the release CI gate, and W&B/Prometheus export.
  Only `ohho bench validate` and the gate wiring run today. The other four
  examples are not backed by a command on main, so those scripts are placeholders with no
  benchmark numbers. CPU examples run in `ci.yml` and block `release.yml`.
  GPU examples run from `serve-examples-gpu.yml` when a self-hosted runner
  is online.

---

## 9. START HERE: post-1.0 development

All milestones (M0–M5) are complete. This tree is OhhO OS v1.2.0, not tagged. To continue:

1. `pip install -e .` and run `python -m unittest discover -s tests`.
   On a base install (Python 3.12.3, Linux) that reported
   `Ran 197 tests` / `OK (skipped=18)`.
2. **PyPI publish** is OHH-20. This repository does not publish.
3. **Installer:** `curl | sh` behind `ohho.com/install.sh`.
4. **New robots/skills:** follow `CONTRIBUTING.md` — one manifest + one
   adapter + one test.
5. **On-robot M1 bring-up:** `docs/hil-preflight.md`, then
   `OHHO_HIL=1` plus explicit ports and `ohho hil --hardware --preflight-ok`
   on the bench. Record the 2-minute MCAP from `docs/hil-bag.md`.
6. Keep `AGENTS.md` §0/§3/§8 in sync as you add features.

### M1 on-robot bring-up (when hardware is available)

1. Install extras: `pip install -e '.[serial,arm,unitree]'`.
2. Complete `docs/hil-preflight.md` (batteries, wiring, e-stop, lidar,
   cameras, Pi 5 Jazzy image) and record the bag in `docs/hil-bag.md`.
3. From an interactive terminal, with the device nodes you confirmed:
   `OHHO_HIL=1 OHHO_OMNIBOT_PORT=... OHHO_OMNIBOT_ARM=... OHHO_GO2_IFACE=... ohho hil --hardware --preflight-ok`
   (writes `docs/hil-results.md`). Omit a variable to skip that device.
4. Or drive manually:
   ```python
   from ohho import Robot
   # OHHO_ARM_HARDWARE=1 and an interactive terminal; arm() asks for the phrase
   with Robot.connect("omnibot", "serial:///dev/ttyUSB0,/dev/ttyACM0") as bot:
       bot.arm(); bot.drive(vx=0.05); bot.move_joints([0, -0.5, 0.5, 0, 0, 0.2])
   with Robot.connect("unitree-go2", "dds://eth0") as bot:
       bot.arm(); bot.drive(vx=0.3, w=0.2); print(bot.telemetry().odom)
   ```

### Bench v0 hardware self-test (OHH-87)

Run the bring-up diagnostic suite for OmniBot or simulated test bench:
1. **Simulation / CI:** `ohho selftest --sim` (runs all 5 checks against simulated hardware, exits 0).
2. **On-robot bench:** `ohho selftest --robot omnibot --allow-spin` (checks base serial, wheel encoders with short pulse, IMU rate >= 20Hz, STS3215 arm bus diagnostics, and camera frames). Generates dated JSON report (`selftest-YYYYMMDD-HHMMSS.json`).
3. **Real-robot run:** Pending on-bench execution by Varun (hardware-gated).

---

## 9a. Safety rules (SafetyGate, OHH-117) — do not weaken

`ohho/safety.py` and `ohho/safety_policy.py`; user docs in `docs/safety.md`.
These are invariants. Changing any of them needs an explicit Linear issue,
not a drive-by edit. OHH-313 delegates the decisions below to a per-class
`SafetyPolicy` (Ground, Arm, Legged). A plugin policy may tighten caps and
is refused at load if it loosens them. Observable caps, arming, watchdogs,
the latch and e-stop stay as written here. Audit lines add a `policy` field.

1. **Always gated.** `Robot.connect` wraps every transport whose
   `protocol != "simulated"` in `SafetyGate`. Do not add a bypass flag. Do not
   give the gate a `__getattr__` passthrough (it would expose `send_motor`,
   raw `sport_command`, etc.).
2. **Fail closed.** Any refusal sends zero velocity. Transport exceptions,
   watchdog errors → latched e-stop. Unwritable audit log → disarm. Refused
   joint/effort commands are dropped, never "zeroed".
3. **Caps come from the per-robot `SafetyProfile`.** Go2: 0.5 m/s forward,
   0.3 m/s lateral, 1.0 rad/s yaw, acceleration-limited. Hard ceiling 1.5 m/s /
   2.0 rad/s; only a safety config file (`hard_ceiling`) raises it, never the
   caller or an agent. Clamp, don't reject.
4. **Watchdogs.** No command for 300 ms → zero velocity. No state for 500 ms
   while armed → stop + latch. Profiles may make these stricter, never looser.
5. **E-stop.** Latched. Stop first (`StopMove()` on Go2), then `Damp()` only
   if the robot is already lying down or `estop_damp` / `--estop-damp` is set.
   Unknown posture never damps. `release_stop()` needs a human at a TTY,
   refuses while tilted/faulted, and leaves the gate disarmed (re-arm needed).
6. **Agents and policies never get `arm` or `release_stop` as tools**
   (`brains.py`, `market.py`). They may call `emergency_stop`.
7. **Allowlist.** `stand_up, stand_down, balance_stand, recovery_stand,
   stop_move, move, euler (±0.3 rad), sit, rise_sit`. Always refuse flips,
   jumps, pounce, dance, handstand, walk_upright, `free_*`, cross_step — the
   denylist wins over any config file. Humanoid names (`prepare`, `walk_mode`,
   `damp`, `get_up`, `lie_down`, `look`, `move_ee`, `gripper`, `hand`) are not
   on those lists. A profile may name them; `command` then calls
   `send_humanoid` and never a sport call. `move_joints` stays refused at the
   gate.
8. **Tilt / fault.** `|roll|` or `|pitch|` > 0.6 rad, or `error_code != 0` →
   stop + latch, armed or not.
9. **Arming.** Never automatic, including tests and CI. Any non-loopback
   interface or DDS domain 0 needs `OHHO_ARM_HARDWARE=1` **and** an interactive
   TTY confirmation ("I am physically present, the area is clear, the remote is
   in my hand"), logged with a timestamp. CI env or non-TTY → refused.
10. **Audit.** Every arm/disarm, e-stop, release, clamp, refusal and watchdog
    trip goes to `~/.ohho/safety/<UTC date>.jsonl`.
11. **Tests.** Fake clock + fake transports only; no sleeps, no hardware. CI
    fails below 95 % line + branch coverage on `ohho/safety.py`.

Not gated: `ohho selftest` and `tests/hil/` construct adapters directly (they
have their own consent: `--allow-spin`, or `OHHO_HIL=1` plus an explicit
device variable). `ohho hil --hardware` also requires `--preflight-ok` and a
TTY. Do not put default serial ports or DDS interfaces back. The MuJoCo 2 m/s →
0.5 m/s and 300 ms client-kill check is OHH-121. The `ohho-quadruped` repo's
AGENTS.md needs the same rules section (not edited from here).

---

## 10. Decisions made (don't re-litigate without reason)

- **Name:** "OhhO OS" is the **engine**, sits beside the products, renames
  nothing. Confirmed by the user.
- **Runtime posture:** **both** ROS and no-ROS, user chooses; the chooser
  explains trade-offs (shipped on `/os` + `docs/ohho-os/runtimes.md`).
- **Launch robots:** OmniBot + Unitree Go2 (max contrast: wheeled/serial/has-arm
  vs legged/DDS/no-arm).
- **Language:** Python-first; the web `Transport` types are the shared contract.
- **License:** Apache-2.0 (OHH-18). `pyproject.toml` sets
  `license = "Apache-2.0"` and `license-files = ["LICENSE"]`.
- **OSS home:** the canonical public Python repo is
  https://github.com/ohho-robotics/ohho-os. `pyproject.toml` `Homepage`,
  `Source`, `Issues`, and `Documentation` point at ohho.com / ohho-os.
  TypeScript packages live at https://github.com/ohho-robotics/ohho-sdk.

---

## 11. Gotchas

- `get_runtime("auto")` returns **native** in M0 (ros2 backend unimplemented);
  flip this in M4.
- `Robot.connect("<any>")` with no transport **silently runs in simulation**
  today (auto fallback). An **explicit** non-sim URI raises `AdapterUnavailable`
  — that's intentional (honest error vs convenient default).
- An explicit non-sim URI that does connect comes back **gated and disarmed**:
  `bot.drive()` sends zero velocity until `bot.arm()` succeeds (see §9a).
  `bot.transport` is the `SafetyGate`, not the adapter.
- The sim's `start_sim()` runs a background thread; for deterministic tests
  construct `Robot(spec, SimTransport(spec), NativeRuntime())` directly and call
  `transport.step(dt)` (don't start the thread).
- `ohho_os.egg-info` / `__pycache__` are gitignored — if `git add sdk` ever tries
  to stage them, the `.gitignore` is missing.

---

## 12. References

- **Website surfaces:** `website/app/os/page.tsx` (marketing),
  `website/docs/ohho-os/{README,install,quickstart,runtimes,robots,training,architecture}.md`.
- **Transport contract (TS source):** `website/lib/connect/types.ts`;
  **adapter catalog / joint maps:** `website/lib/bridge/adapters.ts`;
  **robot taxonomy:** `website/lib/garage/types.ts`.
- **Reuse targets:** `packages/yahboom_ros2`, `packages/mecanum_drive_ros2`,
  `agent_engine/`, `learning_engine/`, `lerobot_engine/`, `vla_engine/`,
  `packages/vla_serve`, `data_engine/`, `robot_ws/` (ROS backend).
- **Merged PRs:** #163 (brand + docs), #164 (SDK M0 core).
- **Repo guide:** root `CLAUDE.md` (monorepo conventions, physical constants,
  topic map).

---

_Last updated: M5 complete — OhhO OS v1.0.0. All milestones (M0–M5) shipped.
151 tests pass. Hardware profiles, skill market, CONTRIBUTING.md, root CLAUDE.md
updated. Ready for PyPI publish._
