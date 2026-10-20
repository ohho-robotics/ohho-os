"""Record one humanoid episode from sim-serve teleop.

The session uses whatever sim backend the spec declares. A headless
humanoid-mujoco link is stepped explicitly so the episode is deterministic.
Velocity, posture, and optional end-effector targets go through the same
WebSocket commands a teleop client sends. The dataset is LeRobot v2.1 with
``robot_type`` from the spec.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from collections.abc import Sequence
from typing import Any, Optional

from .. import capabilities as caps
from ..sim_serve import SimServe
from .humanoid_features import ARM_ACTIONS
from .recorder import Recorder


def record_humanoid_teleop_episode(
    robot_id: str,
    output_dir: str,
    *,
    steps: int = 8,
    fps: float = 10.0,
    repo_id: str = "local/humanoid-sim-teleop",
    task: str = "walk",
    arm_action: Optional[str] = None,
    vx: float = 0.2,
    vy: float = 0.0,
    wz: float = 0.0,
    ee_pose: Optional[Sequence[float]] = None,
) -> str:
    """Record one episode and return the dataset directory.

    ``arm_action`` is ``ee_pose`` or ``arm_joints``. When omitted, custom mode
    records arm joints and every other mode records end-effector poses.
    A head image is included only when the model has a camera and a video
    encoder is installed.
    """
    if arm_action is not None and arm_action not in ARM_ACTIONS:
        raise ValueError("arm_action must be 'ee_pose' or 'arm_joints'")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return asyncio.run(
        _record(
            robot_id,
            output_dir,
            steps=steps,
            fps=fps,
            repo_id=repo_id,
            task=task,
            arm_action=arm_action,
            vx=vx,
            vy=vy,
            wz=wz,
            ee_pose=None if ee_pose is None else tuple(float(v) for v in ee_pose),
        )
    )


def _video_encoder_available() -> bool:
    return (
        importlib.util.find_spec("av") is not None
        or importlib.util.find_spec("imageio") is not None
    )


async def _record(
    robot_id: str,
    output_dir: str,
    *,
    steps: int,
    fps: float,
    repo_id: str,
    task: str,
    arm_action: Optional[str],
    vx: float,
    vy: float,
    wz: float,
    ee_pose: Optional[tuple[float, ...]],
) -> str:
    import websockets

    server = SimServe(
        backend="sim",
        robot=robot_id,
        host="127.0.0.1",
        port=0,
        rate_hz=fps,
    )
    await server.start()
    bot = server._bot
    transport = bot.transport
    stop_sim = getattr(transport, "stop_sim", None)
    if callable(stop_sim):
        stop_sim()

    # Without an encoder, skip frames so a camera does not fail the write.
    camera_fn = None if _video_encoder_available() else (lambda: None)
    recorder = Recorder(
        bot,
        repo_id=repo_id,
        fps=fps,
        camera_fn=camera_fn,
        arm_action=arm_action,
    )
    try:
        url = f"ws://127.0.0.1:{server.port}"
        async with websockets.connect(url) as ws:
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.5)
            except Exception:
                pass
            await _send(ws, vx=0.0, vy=0.0, wz=0.0, request="prepare")
            await asyncio.sleep(0.02)
            await _send(ws, vx=0.0, vy=0.0, wz=0.0, request="walk_mode")
            await asyncio.sleep(0.02)
            recorder.start_episode(task=task)
            for _ in range(steps):
                await _send(ws, vx=vx, vy=vy, wz=wz, request=None)
                await asyncio.sleep(0.02)
                step = getattr(transport, "step", None)
                if callable(step):
                    step(1.0 / fps)
                if ee_pose is not None and bot.has(caps.MANIPULATION_DUAL_ARM):
                    bot.move_ee("left", ee_pose, 0.0)
                    bot.move_ee("right", ee_pose, 0.0)
                recorder.capture_frame()
            recorder.stop_episode()
    finally:
        await server.stop()
    return recorder.save(output_dir)


async def _send(
    ws: Any,
    *,
    vx: float,
    vy: float,
    wz: float,
    request: Optional[str],
) -> None:
    message: dict[str, object] = {
        "deadman": True,
        "velocity": {
            "linear": {"x": vx, "y": vy},
            "angular": {"z": wz},
        },
    }
    if request is not None:
        message["modes"] = {"request": request}
    await ws.send(json.dumps(message))
