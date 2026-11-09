"""Teleoperation episode recorder for legged robots via sim-serve (OHH-129).

Drives the registered quadruped in simulation via ``SimServe`` WebSocket
commands, capturing 31-D legged observation states, 4-D actions (clamped +
raw), and optional front-camera video frames.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any, Dict, Tuple

from ..sim_serve import SimServe
from .reader import DatasetReader
from .recorder import Recorder


def make_sim_camera_frame(
    robot_x: float,
    robot_y: float,
    target_x: float = 2.0,
    target_y: float = 0.2,
    width: int = 160,
    height: int = 120,
) -> Any:
    """Synthesize a front-camera RGB frame rendering the ground and target red box."""
    try:
        import numpy as np  # type: ignore

        # Sky / room background (light grey) and floor (darker grey)
        img = np.full((height, width, 3), 200, dtype=np.uint8)
        horizon = height // 2
        img[horizon:, :] = [140, 140, 140]  # Floor

        # Distance to target
        dist = math.hypot(target_x - robot_x, target_y - robot_y)
        # Bounding box size scales inversely with distance
        box_size = int(max(10, min(height // 2, int(80.0 / max(dist, 0.3)))))
        center_x = width // 2
        center_y = horizon + box_size // 4

        y1 = max(0, center_y - box_size // 2)
        y2 = min(height, center_y + box_size // 2)
        x1 = max(0, center_x - box_size // 2)
        x2 = min(width, center_x + box_size // 2)

        # Red box: [R=220, G=30, B=30]
        img[y1:y2, x1:x2] = [220, 30, 30]
        return img
    except ImportError:
        return None


async def _run_teleop_recording_async(
    output_dir: str,
    num_episodes: int = 10,
    steps_per_episode: int = 20,
    fps: float = 10.0,
    repo_id: str | None = None,
    task: str = "walk to the red box",
    with_video: bool = True,
    target_x: float = 2.0,
    target_y: float = 0.2,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Internal coroutine for recording teleop episodes via sim-serve WebSocket."""
    import websockets

    from ..plugins import plugin_text
    from ..registry import list_specs
    from ..robot_spec import RobotClass

    if not repo_id:
        repo_id = plugin_text("teleop_repo_id") or "ohho-robotics/legged-sim-teleop-v0"
    robot_id = "sim"
    for spec in list_specs():
        if spec.robot_class is RobotClass.QUADRUPED:
            robot_id = spec.id
            break

    server = SimServe(
        backend="sim",
        robot=robot_id,
        host="127.0.0.1",
        port=0,
        rate_hz=fps,
    )
    await server.start()

    # 2. Place target red box in the sim world
    if hasattr(server._bot.transport, "add_object"):
        server._bot.transport.add_object(target_x, target_y, 0.35, "red box")

    # Camera frame synthesizer
    def camera_fn():
        if not with_video:
            return None
        t = server._bot.telemetry()
        px = t.odom.x if t.odom else 0.0
        py = t.odom.y if t.odom else 0.0
        return make_sim_camera_frame(px, py, target_x, target_y, width=160, height=120)

    # 3. Setup Recorder
    rec = Recorder(server._bot, repo_id=repo_id, fps=fps, camera_fn=camera_fn)

    try:
        ws_url = f"ws://127.0.0.1:{server.port}"
        async with websockets.connect(ws_url) as ws:
            # Drain initial telemetry
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.5)
            except Exception:
                pass

            for ep_idx in range(num_episodes):
                # Reset sim pose for episode reproducibility
                server._bot.transport._odom.x = 0.0
                server._bot.transport._odom.y = 0.0
                server._bot.transport._odom.theta = 0.0

                rec.start_episode(task=task)
                rec.set_mode("walk")  # mode_id = 2.0

                for step_i in range(steps_per_episode):
                    t = (step_i + 1) / max(steps_per_episode, 1)

                    # Commanded base velocity walking towards red box
                    vx = 0.35 + 0.05 * math.sin(math.pi * t)
                    vy = 0.02 * math.cos(math.pi * t)
                    w = 0.03 * math.sin(2.0 * math.pi * t)

                    # Send teleop command over WebSocket
                    cmd = {
                        "deadman": True,
                        "velocity": {
                            "linear": {"x": vx, "y": vy},
                            "angular": {"z": w},
                        },
                    }
                    await ws.send(json.dumps(cmd))

                    # Allow event loop to process WebSocket message
                    await asyncio.sleep(0.005)

                    # Advance sim physics deterministically
                    server._bot.transport.step(1.0 / fps)

                    # Capture frame in recorder
                    rec.capture_frame()

                rec.stop_episode()
    finally:
        await server.stop()

    # Save dataset to disk
    dataset_path = rec.save(output_dir)

    # Compute stats using DatasetReader
    reader = DatasetReader(dataset_path)
    stats = reader.stats()

    return dataset_path, stats


def record_legged_teleop_episodes(
    output_dir: str,
    num_episodes: int = 10,
    steps_per_episode: int = 20,
    fps: float = 10.0,
    repo_id: str | None = None,
    task: str = "walk to the red box",
    with_video: bool = True,
    target_x: float = 2.0,
    target_y: float = 0.2,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Record teleoperation episodes for the quadruped in sim via sim-serve.

    Parameters:
        output_dir: directory to write the LeRobot v2.0 dataset.
        num_episodes: number of episodes to record (default: 10).
        steps_per_episode: timesteps per episode (default: 20).
        fps: recording frequency in Hz (default: 10.0).
        repo_id: dataset identifier for metadata / card.
        task: task text description (default: "walk to the red box").
        with_video: whether to synthesize and encode front-camera video.
        target_x: x coordinate of the red box in simulation.
        target_y: y coordinate of the red box in simulation.

    Returns:
        tuple (dataset_path, stats_dict)
    """
    return asyncio.run(
        _run_teleop_recording_async(
            output_dir=output_dir,
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            fps=fps,
            repo_id=repo_id,
            task=task,
            with_video=with_video,
            target_x=target_x,
            target_y=target_y,
        )
    )
