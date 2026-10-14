"""Nightly CPU sim loop: record -> validate -> tiny ACT train -> serve -> drive sim.

Executes the complete Data -> Train -> Serve -> Evaluate pipeline on CPU
in-process with the simulator.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .adapters.sim import SimTransport
from .data import Recorder
from .data.reader import DatasetReader
from .data.recorder import _telemetry_to_state
from .registry import get_spec
from .robot import Robot
from .runtime import NativeRuntime


def record_sim_episodes(
    robot_id: str = "omnibot",
    num_episodes: int = 5,
    steps_per_episode: int = 20,
    fps: float = 10.0,
    repo_id: str = "local/sim_loop",
    output_dir: str = "./sim_loop_output/dataset",
) -> str:
    """Record simulated teleoperation episodes and save a LeRobot v2 dataset."""
    spec = get_spec(robot_id)
    tp = SimTransport(spec)
    tp.connect()
    bot = Robot(spec, tp, NativeRuntime())

    rec = Recorder(bot, repo_id=repo_id, fps=fps)

    try:
        for ep in range(num_episodes):
            task_desc = f"simulated reach and grasp demonstration {ep + 1}"
            rec.start_episode(task=task_desc)

            for step_i in range(steps_per_episode):
                t = (step_i + 1) / max(steps_per_episode, 1)

                # Generate smooth mobile-manipulation trajectory
                shoulder_pan = 0.25 * math.sin(2.0 * math.pi * t)
                shoulder_lift = -0.35 + 0.15 * math.cos(math.pi * t)
                elbow_flex = 0.45 * math.sin(math.pi * t)
                wrist_flex = -0.15 * math.sin(math.pi * t)
                wrist_roll = 0.10 * math.cos(2.0 * math.pi * t)
                gripper = 0.50 if t > 0.5 else 0.05

                vx = 0.08 * math.cos(math.pi * t)
                vy = 0.0
                w = 0.04 * math.sin(math.pi * t)

                if bot.has("manipulation"):
                    bot.move_joints(
                        [
                            shoulder_pan,
                            shoulder_lift,
                            elbow_flex,
                            wrist_flex,
                            wrist_roll,
                            gripper,
                        ]
                    )
                bot.drive(vx=vx, vy=vy, w=w)

                # Advance simulation physics deterministically
                tp.step(1.0 / fps)
                rec.capture_frame()

            rec.stop_episode()
    finally:
        bot.stop()
        bot.disconnect()

    dataset_path = rec.save(output_dir)
    return dataset_path


def validate_schema(dataset_dir: str) -> bool:
    """Validate LeRobot v2.0 dataset on-disk schema."""
    p = Path(dataset_dir).expanduser()
    info_file = p / "meta" / "info.json"
    tasks_file = p / "meta" / "tasks.jsonl"
    episodes_file = p / "meta" / "episodes.jsonl"

    if not info_file.exists():
        raise FileNotFoundError(f"meta/info.json missing in {dataset_dir}")
    if not tasks_file.exists():
        raise FileNotFoundError(f"meta/tasks.jsonl missing in {dataset_dir}")
    if not episodes_file.exists():
        raise FileNotFoundError(f"meta/episodes.jsonl missing in {dataset_dir}")

    reader = DatasetReader(str(p))
    if reader.episode_count < 1:
        raise ValueError(f"dataset has 0 episodes in info.json: {dataset_dir}")
    if reader.frame_count < 1:
        raise ValueError(f"dataset has 0 frames in info.json: {dataset_dir}")
    if reader.state_dim <= 0 or reader.action_dim <= 0:
        raise ValueError(
            f"invalid dims in info.json: state_dim={reader.state_dim}, action_dim={reader.action_dim}"
        )

    # Verify each episode can be loaded and has sequential frames
    for ep_meta, frames in reader.iter_episodes():
        if not frames:
            raise ValueError(f"episode {ep_meta.get('episode_index')} has no frames")
        if not frames[-1].next_done:
            raise ValueError(
                f"last frame in episode {ep_meta.get('episode_index')} must have next_done=True"
            )

    return True


def _lerobot_revision_from_info(info: Dict[str, Any]) -> str:
    """Revision tag for LeRobotDataset — matches ``meta/info.json`` codebase_version."""
    raw = str(info.get("codebase_version", "v2.1"))
    if raw and raw[0].isdigit():
        return f"v{raw}"
    return raw


def validate_lerobot_dataset(dataset_dir: str, mock: bool = False) -> bool:
    """Attempt to load the dataset using Hugging Face's LeRobotDataset.

    Loads from disk only: uses ``repo_id`` and codebase revision from ``meta/info.json``
    and sets ``HF_HUB_OFFLINE=1`` so a local validation failure cannot fall through to
    the Hub (which would 401 for bogus repo ids like the output folder name ``dataset``).

    In non-mock mode (mock=False), if lerobot is missing, or LeRobotDataset raises,
    or 0 frames are loaded, an exception is raised so the run exits non-zero.
    In mock mode (mock=True), failures are caught and False is returned.
    """
    try:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except (ImportError, ModuleNotFoundError):
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except (ImportError, ModuleNotFoundError):
                from lerobot.datasets import LeRobotDataset
    except (ImportError, ModuleNotFoundError) as e:
        if mock:
            return False
        raise RuntimeError(
            f"lerobot is not installed ({e}). In non-mock mode, LeRobotDataset verification is required. "
            "Install with: pip install '.[train]'"
        ) from e

    p = Path(dataset_dir).expanduser()
    info_path = p / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"meta/info.json missing in {dataset_dir}")

    import json

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    repo_id = info.get("repo_id") or "local/dataset"
    revision = _lerobot_revision_from_info(info)

    prev_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        ds = LeRobotDataset(repo_id=repo_id, root=str(p), revision=revision)
        if len(ds) > 0:
            return True
        raise ValueError(f"LeRobotDataset loaded 0 frames from {dataset_dir}")
    except Exception as e:
        if mock:
            return False
        raise RuntimeError(
            f"LeRobotDataset failed to load dataset at {dataset_dir}: {e}"
        ) from e
    finally:
        if prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_offline


def validate_dataset(dataset_dir: str, mock: bool = False) -> Dict[str, Any]:
    """Run schema verification and optional LeRobotDataset loading."""
    schema_ok = validate_schema(dataset_dir)
    if not schema_ok:
        raise ValueError(f"dataset schema check failed for {dataset_dir}")

    lerobot_loaded = validate_lerobot_dataset(dataset_dir, mock=mock)

    reader = DatasetReader(dataset_dir)
    return {
        "schema_valid": schema_ok,
        "lerobot_dataset_loaded": lerobot_loaded,
        "total_episodes": reader.episode_count,
        "total_frames": reader.frame_count,
        "state_dim": reader.state_dim,
        "action_dim": reader.action_dim,
    }


def compute_percentiles(values: List[float]) -> Tuple[float, float]:
    """Compute p50 (median) and p95 percentiles."""
    if not values:
        return 0.0, 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    p50_idx = int(round(0.50 * (n - 1)))
    p95_idx = int(round(0.95 * (n - 1)))
    return sorted_vals[p50_idx], sorted_vals[p95_idx]


def evaluate_policy_sim(
    client: Any,
    robot_id: str = "omnibot",
    eval_steps: int = 50,
    instruction: str = "simulated reach and grasp demonstration",
) -> Tuple[int, List[float]]:
    """Execute closed-loop policy steps on simulated robot and measure serve latency."""
    spec = get_spec(robot_id)
    tp = SimTransport(spec)
    tp.connect()
    bot = Robot(spec, tp, NativeRuntime())

    latencies: List[float] = []
    completed_steps = 0

    try:
        for step_i in range(eval_steps):
            state = _telemetry_to_state(bot)

            t0 = time.monotonic()
            resp = client.post(
                "/predict",
                json={"instruction": instruction, "config": {"state": state}},
            )
            latency_ms = (time.monotonic() - t0) * 1000.0
            latencies.append(latency_ms)

            if resp.status_code != 200:
                raise RuntimeError(
                    f"serve endpoint error at step {step_i} (HTTP {resp.status_code}): {resp.text}"
                )

            data = resp.json()
            action_vec = data.get("action", {}).get("vector", [])
            if not action_vec:
                raise ValueError(f"empty action vector returned at step {step_i}")

            n_arm = len(spec.joint_names) if bot.has("manipulation") else 0
            if n_arm > 0 and len(action_vec) >= n_arm:
                bot.move_joints(action_vec[:n_arm])
            if len(action_vec) >= n_arm + 3:
                bot.drive(
                    vx=action_vec[n_arm],
                    vy=action_vec[n_arm + 1],
                    w=action_vec[n_arm + 2],
                )
            tp.step(0.1)
            completed_steps += 1
    finally:
        bot.stop()
        bot.disconnect()

    return completed_steps, latencies


def format_summary(results: Dict[str, Any]) -> str:
    """Format the sim loop execution results as a markdown job summary."""
    episodes = results.get("episodes_recorded", 0)
    total_frames = results.get("total_frames", 0)
    schema_status = "Pass" if results.get("schema_valid") else "Fail"
    is_mock = results.get("mock", False)
    lerobot_loaded = results.get("lerobot_dataset_loaded", False)
    if lerobot_loaded:
        lerobot_status = "Pass"
    elif is_mock:
        lerobot_status = "Skipped (mock mode)"
    else:
        lerobot_status = "Fail"
    model_name = results.get("model_name", "Tiny ACT (Action Chunking Transformer)")
    train_steps = results.get("train_steps", 0)
    loss_curve = results.get("loss_curve", [])
    initial_loss = results.get("initial_loss", 0.0)
    final_loss = results.get("final_loss", 0.0)
    eval_steps = results.get("eval_steps_taken", 0)
    eval_errors = results.get("eval_errors", 0)
    p50 = results.get("latency_p50_ms", 0.0)
    p95 = results.get("latency_p95_ms", 0.0)

    lines = [
        "## OhhO OS Nightly Sim Loop Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Sim Episodes Recorded** | {episodes} ({total_frames} total frames) |",
        f"| **Dataset Schema Check** | {schema_status} (LeRobot v2.1 format) |",
        f"| **LeRobotDataset Loading** | {lerobot_status} |",
        f"| **Policy Architecture** | {model_name} |",
        f"| **CPU Train Steps** | {train_steps} steps |",
        f"| **Train Loss** | initial={initial_loss:.4f} -> final={final_loss:.4f} |",
        f"| **Policy Sim Steps** | {eval_steps} steps ({eval_errors} errors) |",
        f"| **Serve Latency (p50)** | {p50:.2f} ms |",
        f"| **Serve Latency (p95)** | {p95:.2f} ms |",
        "",
    ]

    if loss_curve:
        lines.append("### Train Loss Curve")
        lines.append("| Step | Loss |")
        lines.append("|---|---|")
        for step, loss_val in loss_curve:
            lines.append(f"| {step} | {loss_val:.4f} |")
        lines.append("")

    summary_text = "\n".join(lines)

    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        try:
            with open(step_summary_path, "a", encoding="utf-8") as f:
                f.write(summary_text + "\n")
        except Exception as e:
            print(
                f"Warning: could not write to GITHUB_STEP_SUMMARY: {e}", file=sys.stderr
            )

    return summary_text


def run_sim_loop(
    robot_id: str = "omnibot",
    episodes: int = 5,
    steps_per_episode: int = 20,
    train_steps: int = 200,
    eval_steps: int = 50,
    device: str = "cpu",
    output_dir: str = "./sim_loop_output",
    mock: bool = False,
    batch_size: int = 8,
    lr: float = 1e-4,
) -> Dict[str, Any]:
    """Execute end-to-end CPU sim loop: Record -> Validate -> Train -> Serve -> Evaluate."""
    root = Path(output_dir).expanduser()
    dataset_dir = root / "dataset"
    checkpoint_dir = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)

    print(
        f"[1/5] Recording {episodes} sim episodes ({steps_per_episode} steps each) on {robot_id}..."
    )
    dataset_path = record_sim_episodes(
        robot_id=robot_id,
        num_episodes=episodes,
        steps_per_episode=steps_per_episode,
        output_dir=str(dataset_dir),
    )
    print(f"      Dataset saved to: {dataset_path}")

    print("[2/5] Validating dataset schema & LeRobot loading...")
    val_info = validate_dataset(dataset_path, mock=mock)
    if not val_info.get("schema_valid"):
        raise ValueError(f"dataset schema validation failed for {dataset_path}")
    if not mock and not val_info.get("lerobot_dataset_loaded"):
        raise RuntimeError(
            f"LeRobotDataset verification failed for dataset at {dataset_path}"
        )
    print(
        f"      Schema valid: {val_info['schema_valid']} | "
        f"Episodes: {val_info['total_episodes']} | "
        f"Frames: {val_info['total_frames']} | "
        f"Dims: state={val_info['state_dim']}, action={val_info['action_dim']}"
    )

    print(f"[3/5] Training tiny ACT model for {train_steps} CPU steps (mock={mock})...")
    from .train.act import train_act

    ckpt_path = train_act(
        dataset=dataset_path,
        output_dir=str(checkpoint_dir),
        device=device,
        train_steps=train_steps,
        batch_size=batch_size,
        lr=lr,
        mock=mock,
    )
    print(f"      Trained checkpoint saved to: {ckpt_path}")

    # Read training loss metrics
    metrics_file = Path(ckpt_path) / "metrics.json"
    loss_curve: List[Tuple[int, float]] = []
    final_loss = 0.0
    initial_loss = 0.0
    if metrics_file.exists():
        import json

        with open(metrics_file, encoding="utf-8") as f:
            m = json.load(f)
            loss_curve = [tuple(item) for item in m.get("loss_curve", [])]  # type: ignore[misc]
            final_loss = float(m.get("final_loss", 0.0))
            if loss_curve:
                initial_loss = float(loss_curve[0][1])

    print(
        f"[4/5] Booting policy serve endpoint & taking {eval_steps} policy steps in sim..."
    )
    try:
        from .serve import build_app
        from fastapi.testclient import TestClient
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            f"fastapi is not installed ({e}). Sim loop serve evaluation requires the [serve] extra. "
            "Install with: pip install '.[serve]'"
        ) from e

    app = build_app(
        model_class="act",
        model_path=ckpt_path,
        device=device,
        mock_model=mock,
        auto_load=True,
    )
    client = TestClient(app)

    # Health check
    h = client.get("/health")
    if h.status_code != 200 or not h.json().get("model_loaded"):
        raise RuntimeError(f"serve health check failed: {h.text}")

    completed_steps, latencies = evaluate_policy_sim(
        client=client,
        robot_id=robot_id,
        eval_steps=eval_steps,
    )
    p50, p95 = compute_percentiles(latencies)
    print(
        f"      Closed-loop evaluation: {completed_steps}/{eval_steps} steps | "
        f"Latency p50={p50:.2f}ms, p95={p95:.2f}ms"
    )

    results: Dict[str, Any] = {
        "episodes_recorded": episodes,
        "total_frames": val_info["total_frames"],
        "dataset_path": str(dataset_dir),
        "schema_valid": val_info["schema_valid"],
        "lerobot_dataset_loaded": val_info["lerobot_dataset_loaded"],
        "mock": mock,
        "model_name": "Tiny ACT (Action Chunking Transformer)",
        "train_steps": train_steps,
        "loss_curve": loss_curve,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "checkpoint_path": str(checkpoint_dir),
        "eval_steps_taken": completed_steps,
        "eval_errors": 0,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
    }

    print("[5/5] Generating job summary...")
    summary = format_summary(results)
    print("\n" + summary)

    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run nightly CPU sim loop: Record -> Validate -> Train ACT -> Serve -> Drive sim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot", default="omnibot", help="robot spec ID to simulate")
    parser.add_argument(
        "--episodes", type=int, default=5, help="number of episodes to record"
    )
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=20,
        help="number of frames recorded per episode",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=200,
        help="number of CPU training steps for ACT",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="number of closed-loop policy evaluation steps",
    )
    parser.add_argument(
        "--device", default="cpu", help="device to run training and inference on"
    )
    parser.add_argument(
        "--output-dir",
        default="./sim_loop_output",
        help="output directory for dataset and checkpoints",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="run in mock mode (no torch or GPU dependencies required)",
    )
    args = parser.parse_args(argv)

    try:
        run_sim_loop(
            robot_id=args.robot,
            episodes=args.episodes,
            steps_per_episode=args.steps_per_episode,
            train_steps=args.train_steps,
            eval_steps=args.eval_steps,
            device=args.device,
            output_dir=args.output_dir,
            mock=args.mock,
        )
        return 0
    except Exception as e:
        print(f"Error executing sim loop: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
