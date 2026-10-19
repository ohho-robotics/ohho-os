"""Teleoperation episode recorders via sim-serve.

OHH-129 records a Unitree Go2: 31-D legged state, 4-D action, optional
front-camera video. OHH-292 records ``px4-sitl`` on the kinematic drone-sim:
11-D aerial state and the 4-D ``BODY_FRD`` velocity setpoint. No camera and
no LiDAR in the aerial M1 dataset. Neither recorder connects to hardware.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, Callable, Dict, List, Tuple

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
    repo_id: str = "ohho-robotics/go2-sim-teleop-v0",
    task: str = "walk to the red box",
    with_video: bool = True,
    target_x: float = 2.0,
    target_y: float = 0.2,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Internal coroutine for recording teleop episodes via sim-serve WebSocket."""
    import websockets

    # 1. Start SimServe for Go2 on random port
    server = SimServe(
        backend="sim",
        robot="unitree-go2",
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


def record_go2_teleop_episodes(
    output_dir: str,
    num_episodes: int = 10,
    steps_per_episode: int = 20,
    fps: float = 10.0,
    repo_id: str = "ohho-robotics/go2-sim-teleop-v0",
    task: str = "walk to the red box",
    with_video: bool = True,
    target_x: float = 2.0,
    target_y: float = 0.2,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Record teleoperation episodes for Unitree Go2 in sim via sim-serve.

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


# Body-frame velocity legs of a flat square. vz is down in BODY_FRD; 0 holds climb.
_SQUARE_LEGS: List[Tuple[float, float, float, float]] = [
    (0.4, 0.0, 0.0, 0.0),
    (0.0, 0.4, 0.0, 0.0),
    (-0.4, 0.0, 0.0, 0.0),
    (0.0, -0.4, 0.0, 0.0),
]


def _drone(snap: Any) -> Dict[str, Any]:
    custom = getattr(snap, "custom", None)
    if not isinstance(custom, dict):
        return {}
    drone = custom.get("drone")
    return drone if isinstance(drone, dict) else {}


def _step_until(server: SimServe, pred: Callable[[Any], bool], limit: int = 80) -> Any:
    """Advance drone-sim by fixed ``dt``. The broadcast loop is not the clock."""
    transport = server._bot.transport
    snap = transport.read()
    for _ in range(limit):
        if pred(snap):
            return snap
        snap = transport.step(0.1)
    if pred(snap):
        return snap
    raise RuntimeError(f"drone-sim did not reach the expected state: {_drone(snap)}")


async def _send(ws: Any, payload: Dict[str, Any]) -> None:
    await ws.send(json.dumps(payload))
    await asyncio.sleep(0.05)


async def _wait_drone(
    server: SimServe, pred: Callable[[Dict[str, Any]], bool], timeout: float = 2.0
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _drone(server._bot.telemetry())
        if pred(last):
            return last
        await asyncio.sleep(0.02)
    raise RuntimeError(f"drone-sim condition not met: {last}")


async def _send_setpoint(
    ws: Any, server: SimServe, vx: float, vy: float, vz: float, yaw_rate: float
) -> None:
    """Send one BODY_FRD setpoint and wait until drone-sim stored it."""
    await ws.send(
        json.dumps(
            {
                "deadman": True,
                "estop": False,
                "frame": "BODY_FRD",
                "velocity": {
                    "linear": {"x": vx, "y": vy, "z": vz},
                    "angular": {"z": yaw_rate},
                },
            }
        )
    )
    deadline = time.monotonic() + 1.0
    transport = server._bot.transport
    while time.monotonic() < deadline:
        await asyncio.sleep(0.02)
        if (
            getattr(transport, "_phase", None) == "offboard_velocity"
            and abs(float(transport._cmd_vx) - vx) < 1e-6
            and abs(float(transport._cmd_vy) - vy) < 1e-6
            and abs(float(transport._cmd_vz) - vz) < 1e-6
            and abs(float(transport._cmd_yaw_rate) - yaw_rate) < 1e-6
        ):
            return
    raise RuntimeError("sim-serve did not apply the BODY_FRD velocity setpoint")


async def _arm_and_takeoff(ws: Any, server: SimServe, alt_m: float) -> None:
    """Arm and climb on backend=sim. Does not set ``OHHO_ARM_HARDWARE``."""
    await _send(ws, {"flight": {"request": "arm"}, "deadman": False, "estop": False})
    await _wait_drone(server, lambda drone: drone.get("armed") is True)
    await _send(
        ws,
        {
            "flight": {"request": "takeoff", "alt_m": alt_m},
            "deadman": False,
            "estop": False,
        },
    )
    await _wait_drone(
        server,
        lambda drone: drone.get("flight_mode") in {"takeoff", "hold"},
    )
    _step_until(
        server,
        lambda snap: (
            _drone(snap).get("flight_mode") == "hold"
            and _drone(snap).get("landed_state") == "in_air"
        ),
    )


async def _land(ws: Any, server: SimServe) -> None:
    await _send(ws, {"flight": {"request": "land"}, "deadman": False, "estop": False})
    await _wait_drone(
        server,
        lambda drone: (
            drone.get("flight_mode") == "land"
            or (
                drone.get("landed_state") == "on_ground" and drone.get("armed") is False
            )
        ),
    )
    _step_until(
        server,
        lambda snap: (
            _drone(snap).get("landed_state") == "on_ground"
            and _drone(snap).get("armed") is False
        ),
    )


async def _run_aerial_teleop_async(
    output_dir: str,
    num_episodes: int,
    fps: float,
    repo_id: str,
    alt_m: float,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    import websockets

    server = SimServe(
        backend="sim",
        robot="px4-sitl",
        host="127.0.0.1",
        port=0,
        rate_hz=fps,
    )
    await server.start()
    # No camera_fn: observation.images.front stays absent. LiDAR is not recorded.
    rec = Recorder(server._bot, repo_id=repo_id, fps=fps)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{server.port}") as ws:
            try:
                await asyncio.wait_for(ws.recv(), timeout=1.0)
            except Exception:
                pass
            for index in range(num_episodes):
                square = index % 2 == 1
                task = "sim square" if square else "sim takeoff hover land"
                await _arm_and_takeoff(ws, server, alt_m)
                rec.start_episode(task=task)
                if square:
                    for vx, vy, vz, yaw_rate in _SQUARE_LEGS:
                        for _ in range(2):
                            await _send_setpoint(ws, server, vx, vy, vz, yaw_rate)
                            server._bot.transport.step(0.1)
                            rec.capture_frame()
                else:
                    for _ in range(3):
                        server._bot.transport.step(0.1)
                        rec.capture_frame()
                await _land(ws, server)
                rec.capture_frame()
                rec.stop_episode()
    finally:
        await server.stop()

    dataset_path = rec.save(output_dir)
    return dataset_path, DatasetReader(dataset_path).stats()


def record_aerial_teleop_episodes(
    output_dir: str,
    num_episodes: int = 3,
    fps: float = 10.0,
    repo_id: str = "ohho-robotics/px4-sitl-sim-teleop",
    alt_m: float = 0.4,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Record a few ``px4-sitl`` episodes through sim-serve and drone-sim.

    The first episode is ``"sim takeoff hover land"``. Every second episode is
    a flat body-frame square (``"sim square"``). ``alt_m`` is positive up. This uses the
    in-process kinematic stub only: no Docker, no PX4, no hardware.

    Returns ``(dataset_path, stats)``.
    """
    if num_episodes < 1:
        raise ValueError("num_episodes must be >= 1")
    if alt_m <= 0.0:
        raise ValueError("alt_m must be positive up, in metres")
    return asyncio.run(
        _run_aerial_teleop_async(
            output_dir=output_dir,
            num_episodes=num_episodes,
            fps=fps,
            repo_id=repo_id,
            alt_m=alt_m,
        )
    )
