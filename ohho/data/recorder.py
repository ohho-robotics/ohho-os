"""Episode recorder — captures state/action frames from a Robot.

The recorder wraps the robot's ``drive`` and ``move_joints`` methods to intercept
commands (the "action"), and polls ``telemetry()`` for the "state". Frames are
captured either on a timer (live) or manually (deterministic tests / sim).

Frame schema (matches LeRobot v2.1 + the mobile-manipulation spec):
  - ``observation.state``: arm joints (6) + base velocity (3) = 9-D
  - ``action``: commanded arm joints (6) + base velocity (3) = 9-D
  - ``timestamp`` (``frame_index / fps``), ``frame_index``, ``episode_index``,
    ``task_index``, ``next.done``
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import capabilities as caps
from ..robot import Robot


@dataclass
class Frame:
    """One recorded timestep."""

    observation_state: List[float]
    action: List[float]
    timestamp: float
    frame_index: int
    episode_index: int
    task_index: int
    next_done: bool = False
    action_raw: Optional[List[float]] = None
    image: Optional[Any] = None


@dataclass
class Episode:
    """A recorded episode — a list of frames plus metadata."""

    task: str = ""
    frames: List[Frame] = field(default_factory=list)
    episode_index: int = 0
    task_index: int = 0
    fps: float = 10.0
    images: Dict[str, List[Any]] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.frames)

    @property
    def state_dim(self) -> int:
        return len(self.frames[0].observation_state) if self.frames else 0

    @property
    def action_dim(self) -> int:
        return len(self.frames[0].action) if self.frames else 0

    @property
    def has_videos(self) -> bool:
        return bool(self.images)


GO2_LEG_JOINTS = (
    "fl_hip",
    "fl_thigh",
    "fl_calf",
    "fr_hip",
    "fr_thigh",
    "fr_calf",
    "rl_hip",
    "rl_thigh",
    "rl_calf",
    "rr_hip",
    "rr_thigh",
    "rr_calf",
)

GO2_STATE_NAMES = [
    "base_vx",
    "base_vy",
    "base_omega",
    "roll",
    "pitch",
    "yaw",
    "fl_hip_q",
    "fl_thigh_q",
    "fl_calf_q",
    "fr_hip_q",
    "fr_thigh_q",
    "fr_calf_q",
    "rl_hip_q",
    "rl_thigh_q",
    "rl_calf_q",
    "rr_hip_q",
    "rr_thigh_q",
    "rr_calf_q",
    "fl_hip_dq",
    "fl_thigh_dq",
    "fl_calf_dq",
    "fr_hip_dq",
    "fr_thigh_dq",
    "fr_calf_dq",
    "rl_hip_dq",
    "rl_thigh_dq",
    "rl_calf_dq",
    "rr_hip_dq",
    "rr_thigh_dq",
    "rr_calf_dq",
    "posture_id",
]

GO2_ACTION_NAMES = [
    "base_vx",
    "base_vy",
    "base_omega",
    "mode_id",
]

OMNIBOT_STATE_NAMES = [
    "arm_shoulder_pan",
    "arm_shoulder_lift",
    "arm_elbow_flex",
    "arm_wrist_flex",
    "arm_wrist_roll",
    "arm_gripper",
    "base_vx",
    "base_vy",
    "base_omega",
]

OMNIBOT_ACTION_NAMES = [
    "arm_shoulder_pan",
    "arm_shoulder_lift",
    "arm_elbow_flex",
    "arm_wrist_flex",
    "arm_wrist_roll",
    "arm_gripper",
    "base_vx",
    "base_vy",
    "base_omega",
]

BASE_STATE_NAMES = [
    "base_vx",
    "base_vy",
    "base_omega",
]

BASE_ACTION_NAMES = [
    "base_vx",
    "base_vy",
    "base_omega",
]

MODE_TO_ID: Dict[str, float] = {
    "idle": 0.0,
    "stand": 1.0,
    "stand_up": 1.0,
    "balance_stand": 1.0,
    "recovery_stand": 1.0,
    "walk": 2.0,
    "trot": 3.0,
    "run": 4.0,
    "climb": 5.0,
    "sit": 6.0,
    "rise_sit": 6.0,
    "damp": 7.0,
}


def _is_legged(robot: Robot) -> bool:
    """Return True if the robot is legged (e.g. Unitree Go2)."""
    return (
        robot.has(caps.LEGGED_POSE)
        or getattr(robot.spec, "category", "") == "legged"
        or getattr(robot.spec, "id", "") == "unitree-go2"
    )


def _telemetry_to_state(robot: Robot) -> List[float]:
    """Extract the observation state vector from telemetry.

    - Legged 31-D: base velocity (3) + rpy (3) + 12 joint q + 12 joint dq + posture id
    - Mobile-manipulation 9-D: arm joints (6) + base velocity (3)
    - Base-only 3-D: base velocity (vx, vy, omega)
    """
    t = robot.telemetry()
    base_vel = (
        [float(t.odom.vx), float(t.odom.vy), float(t.odom.omega)]
        if t.odom
        else [0.0, 0.0, 0.0]
    )

    if _is_legged(robot):
        # 1. Base velocity (3)
        # 2. RPY (3)
        roll = float(t.custom.get("roll", 0.0))
        pitch = float(t.custom.get("pitch", 0.0))
        yaw = float(t.odom.theta if t.odom else t.custom.get("yaw", 0.0))
        rpy = [roll, pitch, yaw]

        # 3. 12 joint q + 4. 12 joint dq
        q = [0.0] * 12
        dq = [0.0] * 12
        if t.joints:
            j_map = {j.name.lower(): j for j in t.joints}
            if any(name in j_map for name in GO2_LEG_JOINTS):
                for idx, name in enumerate(GO2_LEG_JOINTS):
                    if name in j_map:
                        q[idx] = float(j_map[name].position)
                        if j_map[name].velocity is not None:
                            dq[idx] = float(j_map[name].velocity)
            else:
                for idx, j in enumerate(t.joints[:12]):
                    q[idx] = float(j.position)
                    if j.velocity is not None:
                        dq[idx] = float(j.velocity)
        elif "joint_q" in t.custom:
            custom_q = t.custom["joint_q"]
            for idx in range(min(12, len(custom_q))):
                q[idx] = float(custom_q[idx])
            if "joint_dq" in t.custom:
                custom_dq = t.custom["joint_dq"]
                for idx in range(min(12, len(custom_dq))):
                    dq[idx] = float(custom_dq[idx])

        # 5. Posture ID (1)
        posture_id = float(t.custom.get("posture_id", t.custom.get("mode", 0.0)))

        return base_vel + rpy + q + dq + [posture_id]

    arm = [j.position for j in t.joints] if t.joints else []
    if robot.has(caps.MANIPULATION):
        return arm + base_vel
    return base_vel


def _normalize_image(img: Any) -> Any:
    """Normalize raw camera frame to uint8 RGB numpy array (H, W, 3) or PIL Image."""
    if img is None:
        return None
    if isinstance(img, str):
        try:
            import base64
            import io
            from PIL import Image  # type: ignore

            raw = base64.b64decode(img)
            pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
            try:
                import numpy as np  # type: ignore

                return np.array(pil_img, dtype=np.uint8)
            except ImportError:
                return pil_img
        except Exception:
            return None
    if isinstance(img, bytes):
        try:
            import io
            from PIL import Image  # type: ignore

            pil_img = Image.open(io.BytesIO(img)).convert("RGB")
            try:
                import numpy as np  # type: ignore

                return np.array(pil_img, dtype=np.uint8)
            except ImportError:
                return pil_img
        except Exception:
            return None
    try:
        from PIL import Image  # type: ignore

        if isinstance(img, Image.Image):
            pil_img = img.convert("RGB")
            try:
                import numpy as np  # type: ignore

                return np.array(pil_img, dtype=np.uint8)
            except ImportError:
                return pil_img
    except ImportError:
        pass
    try:
        import numpy as np  # type: ignore

        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] != 3:
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            return img
    except ImportError:
        pass
    return img


class Recorder:
    """Records episodes from a :class:`Robot`.

    Usage (manual / sim)::

        rec = Recorder(bot, repo_id="local/demo")
        rec.start_episode(task="pick up the cup")
        for _ in range(50):
            bot.drive(vx=0.1)
            rec.capture_frame()
        rec.stop_episode()
        rec.save("~/datasets/demo")

    Usage (live / timer)::

        rec = Recorder(bot, repo_id="local/demo", fps=10.0)
        rec.record_episode(task="pick up the cup")  # background thread
        # ...teleop...
        rec.stop_episode()
        rec.save("~/datasets/demo")
    """

    def __init__(
        self,
        robot: Robot,
        repo_id: str = "local/episodes",
        fps: float = 10.0,
        camera_fn: Optional[Any] = None,
    ) -> None:
        self.robot = robot
        self.repo_id = repo_id
        self.fps = fps
        self._camera_fn = camera_fn
        self._current_mode: float = 0.0
        self._episodes: List[Episode] = []
        self._current: Optional[Episode] = None
        self._last_action: List[float] = self._zero_action()
        self._last_action_raw: List[float] = self._zero_action()
        self._frame_idx = 0
        self._task_map: Dict[str, int] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wrap_robot()

    def set_mode(self, mode: Any) -> None:
        """Set the active mode / posture ID for legged robots."""
        if isinstance(mode, str):
            mid = MODE_TO_ID.get(mode.lower(), 0.0)
        else:
            mid = float(mode)
        self._current_mode = mid
        if _is_legged(self.robot):
            if len(self._last_action) >= 4:
                self._last_action[3] = mid
            if len(self._last_action_raw) >= 4:
                self._last_action_raw[3] = mid

    def _zero_action(self) -> List[float]:
        """Zero action vector sized to the robot's DOF and category."""
        if _is_legged(self.robot):
            return [0.0, 0.0, 0.0, getattr(self, "_current_mode", 0.0)]
        n_arm = (
            len(self.robot.spec.joint_names) if self.robot.has(caps.MANIPULATION) else 0
        )
        return [0.0] * (n_arm + 3)

    def _wrap_robot(self) -> None:
        """Intercept drive/move_joints to capture both raw requests and clamped actions."""
        original_drive = self.robot.drive
        original_move_joints = self.robot.move_joints

        recorder = self

        def drive(vx: float = 0.0, vy: float = 0.0, w: float = 0.0):
            # 1. Capture raw action requested
            raw_action = recorder._last_action_raw[:]
            if _is_legged(recorder.robot):
                raw_action = [
                    float(vx),
                    float(vy),
                    float(w),
                    float(recorder._current_mode),
                ]
            elif recorder.robot.has(caps.MANIPULATION):
                n_arm = len(recorder.robot.spec.joint_names)
                raw_action[n_arm:] = [float(vx), float(vy), float(w)]
            else:
                raw_action = [float(vx), float(vy), float(w)]
            recorder._last_action_raw = raw_action

            # 2. Invoke Robot.drive (passes through Robot clamp and SafetyGate)
            res = original_drive(vx, vy, w)

            # 3. Determine clamped action that actually reached the motors
            tp = recorder.robot.transport
            if hasattr(tp, "_last_sent"):
                sent_vx, sent_vy, sent_w = tp._last_sent
            else:
                from ..schema import clamp as schema_clamp

                sent_vx = schema_clamp(
                    vx, -recorder.robot.spec.max_lin, recorder.robot.spec.max_lin
                )
                sent_vy = (
                    schema_clamp(
                        vy, -recorder.robot.spec.max_lin, recorder.robot.spec.max_lin
                    )
                    if recorder.robot.has(caps.BASE_HOLONOMIC)
                    else 0.0
                )
                sent_w = schema_clamp(
                    w, -recorder.robot.spec.max_ang, recorder.robot.spec.max_ang
                )

            clamped_action = recorder._last_action[:]
            if _is_legged(recorder.robot):
                clamped_action = [
                    float(sent_vx),
                    float(sent_vy),
                    float(sent_w),
                    float(recorder._current_mode),
                ]
            elif recorder.robot.has(caps.MANIPULATION):
                n_arm = len(recorder.robot.spec.joint_names)
                clamped_action[n_arm:] = [float(sent_vx), float(sent_vy), float(sent_w)]
            else:
                clamped_action = [float(sent_vx), float(sent_vy), float(sent_w)]
            recorder._last_action = clamped_action
            return res

        def move_joints(positions):
            positions = list(positions)
            raw_action = recorder._last_action_raw[:]
            n_arm = min(
                len(positions),
                len(raw_action) - 3 if not _is_legged(recorder.robot) else 0,
            )
            raw_action[:n_arm] = [float(p) for p in positions[:n_arm]]
            recorder._last_action_raw = raw_action

            res = original_move_joints(positions)

            clamped_action = recorder._last_action[:]
            tp = recorder.robot.transport
            if hasattr(tp, "_joint_cmds"):
                for i, name in enumerate(recorder.robot.spec.joint_names[:n_arm]):
                    if name in tp._joint_cmds:
                        clamped_action[i] = float(tp._joint_cmds[name])
            else:
                clamped_action[:n_arm] = [float(p) for p in positions[:n_arm]]
            recorder._last_action = clamped_action
            return res

        self.robot.drive = drive  # type: ignore[method-assign]
        self.robot.move_joints = move_joints  # type: ignore[method-assign]

    # ── manual recording ─────────────────────────────────────────────────────
    def start_episode(self, task: str = "") -> Episode:
        """Begin a new episode (manual mode)."""
        if self._current is not None:
            raise RuntimeError("an episode is already recording — stop it first")
        task_index = self._task_map.get(task, len(self._task_map))
        self._task_map[task] = task_index
        self._current = Episode(
            task=task,
            episode_index=len(self._episodes),
            task_index=task_index,
            fps=self.fps,
        )
        self._frame_idx = 0
        self._last_action = self._zero_action()
        return self._current

    def _capture_camera_frame(self) -> Optional[Any]:
        """Fetch camera frame from custom camera_fn or robot / transport."""
        if self._camera_fn is not None:
            try:
                frame = self._camera_fn()
                if frame is not None:
                    return _normalize_image(frame)
            except Exception:
                return None
        get_frame = getattr(self.robot, "get_camera_frame", None)
        if callable(get_frame):
            try:
                frame = get_frame()
                if frame is not None:
                    return _normalize_image(frame)
            except Exception:
                pass
        tp = getattr(self.robot, "transport", None)
        if tp is not None:
            get_frame_tp = getattr(tp, "get_camera_frame", None)
            if callable(get_frame_tp):
                try:
                    frame = get_frame_tp()
                    if frame is not None:
                        return _normalize_image(frame)
                except Exception:
                    pass
            inner = getattr(tp, "_inner", None)
            if inner is not None:
                get_frame_inner = getattr(inner, "get_camera_frame", None)
                if callable(get_frame_inner):
                    try:
                        frame = get_frame_inner()
                        if frame is not None:
                            return _normalize_image(frame)
                    except Exception:
                        pass
        return None

    def capture_frame(self) -> Frame:
        """Capture one frame (manual mode). Must be called after start_episode."""
        if self._current is None:
            raise RuntimeError("no active episode — call start_episode() first")
        state = _telemetry_to_state(self.robot)
        action = self._last_action[:]
        action_raw = self._last_action_raw[:]
        # Wall-clock deltas from a tight sim loop are ~0 and fail LeRobot's
        # consecutive-timestamp check. Space frames by 1/fps.
        ts = self._frame_idx / self.fps if self.fps else float(self._frame_idx)

        cam_frame = self._capture_camera_frame()
        if cam_frame is not None:
            if "observation.images.front" not in self._current.images:
                self._current.images["observation.images.front"] = []
            self._current.images["observation.images.front"].append(cam_frame)

        frame = Frame(
            observation_state=state,
            action=action,
            action_raw=action_raw,
            timestamp=ts,
            frame_index=self._frame_idx,
            episode_index=self._current.episode_index,
            task_index=self._current.task_index,
            image=cam_frame,
        )
        self._current.frames.append(frame)
        self._frame_idx += 1
        return frame

    def stop_episode(self) -> Episode:
        """Finish the current episode and store it."""
        if self._current is None:
            raise RuntimeError("no active episode")
        if self._current.frames:
            self._current.frames[-1].next_done = True
        ep = self._current
        self._episodes.append(ep)
        self._current = None
        return ep

    def discard_episode(self) -> None:
        """Discard the current episode without saving."""
        self._current = None

    # ── timer recording ──────────────────────────────────────────────────────
    def record_episode(self, task: str = "") -> Episode:
        """Start a background-thread episode at ``fps`` Hz."""
        self.start_episode(task)
        self._stop.clear()
        period = 1.0 / self.fps

        def loop():
            while not self._stop.is_set() and self._current is not None:
                self.capture_frame()
                time.sleep(period)

        self._thread = threading.Thread(target=loop, name="ohho-recorder", daemon=True)
        self._thread.start()
        return self._current

    def stop_recording(self) -> Episode:
        """Stop the background thread and finish the episode."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        return self.stop_episode()

    # ── output ───────────────────────────────────────────────────────────────
    @property
    def episodes(self) -> List[Episode]:
        return list(self._episodes)

    @property
    def total_frames(self) -> int:
        return sum(ep.length for ep in self._episodes)

    @property
    def tasks(self) -> List[str]:
        return sorted(self._task_map, key=lambda t: self._task_map[t])

    def save(self, path: str) -> str:
        """Write all episodes to ``path`` in LeRobot v2.1 format.

        Uses Parquet when ``pyarrow`` is available (the ``[data]`` extra),
        otherwise falls back to JSON Lines. Meta files (``info.json``,
        ``episodes.jsonl``, ``tasks.jsonl``) are always JSON.
        """
        from .writer import write_dataset

        robot_id = getattr(getattr(self.robot, "spec", None), "id", "omnibot")
        return write_dataset(
            self._episodes,
            path,
            self.repo_id,
            self.fps,
            self.tasks,
            robot_type=robot_id,
        )
