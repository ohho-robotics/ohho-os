"""Camera-to-action pipeline benchmark (OHH-474).

``ohho bench --pipeline`` replays a recorded camera stream, or steps the same
in-process simulator the sim loop uses, through seven stages: decode,
preprocess, h2d, inference, d2h, postprocess, publish. The result is an
``ohho.bench/v1`` document with an additive ``pipeline`` block versioned
``ohho.bench.pipeline/v1``.

H2D and D2H are host copies. A CUDA device is not required. Those stages are
listed on ``pipeline.gpu_follow_up`` until a GPU runner times real copies.
This module never opens a serial, DDS, or other hardware transport.
``live_robot`` is false on every result.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import statistics
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ohho.bench.clock import PerfClock
from ohho.bench.environment import (
    MemoryTrace,
    collect_environment,
    gpu_telemetry,
    host_block,
)
from ohho.bench.errors import BenchHarnessError
from ohho.bench.gates import check_gate_expressions
from ohho.bench.harness import finalize_result
from ohho.bench.schema import SCHEMA_ID
from ohho.bench.stats import summarize
from ohho.schema import clamp
from ohho.targets.errors import TargetProfileError
from ohho.targets.schema import get_profile

PIPELINE_SCHEMA_ID = "ohho.bench.pipeline/v1"
STREAM_SCHEMA_ID = "ohho.pipeline.stream/v1"
BAG_SCHEMA_ID = "ohho.pipeline.rosbag/v1"
PIPELINE_STAGE_NAMES: tuple[str, ...] = (
    "decode",
    "preprocess",
    "h2d",
    "inference",
    "d2h",
    "postprocess",
    "publish",
)
_SOURCES = frozenset({"sim", "lerobot", "rosbag", "fixture", "hardware"})
_SINKS = frozenset({"inprocess", "sim", "ros2"})
_WORLDS = frozenset({"sim", "recorded", "hardware"})
_DEVICE_SCHEMES = frozenset(
    {"serial", "dds", "feetech", "usb", "can", "mavlink", "px4", "ros2"}
)
_SHA = re.compile(r"^(sha256:)?[0-9a-fA-F]{7,128}\Z")
_MAX_EDGE = 4096
_EPS = 1e-9
_FEATURE_DIM = 8
_SIM_WIDTH = 8
_SIM_HEIGHT = 6
_GPU_FOLLOW_UP = ("h2d", "d2h")
_GPU_NOTE = (
    "H2D and D2H ran as host copies on CPU. CUDA event timing for those "
    "stages needs a CUDA device and is a follow-up."
)
_TINY_WEIGHTS: tuple[tuple[float, ...], ...] = tuple(
    tuple(0.001 * (row + 1) * (col + 1) for col in range(_FEATURE_DIM))
    for row in range(3)
)
_TINY_BIAS = (0.1, 0.0, 0.02)

__all__ = [
    "BAG_SCHEMA_ID",
    "PIPELINE_SCHEMA_ID",
    "PIPELINE_STAGE_NAMES",
    "STREAM_SCHEMA_ID",
    "InProcessSink",
    "PipelineConfig",
    "Ros2PublishSink",
    "SimSink",
    "TinyCpuPolicy",
    "execute_pipeline",
]


@dataclass(frozen=True)
class PipelineConfig:
    """What the pipeline bench measures. Gates live on :func:`execute_pipeline`."""

    source: str = "sim"
    target: str = "x86-cpu"
    hz: float = 30.0
    warmup: int = 1
    iters: int = 30
    frames: int | None = None
    episode: int = 0
    robot: str = "sim"
    dataset: str | None = None
    bag: str | None = None
    fixture: str | None = None
    sink: str = ""
    policy: str = "tiny-cpu"
    allow_hardware: bool = False
    workload: str = "pipeline"

    def __post_init__(self) -> None:
        source = self.source.strip() if isinstance(self.source, str) else ""
        if source not in _SOURCES:
            raise BenchHarnessError(
                "--source must be sim, lerobot, rosbag, fixture, or hardware"
            )
        if not isinstance(self.target, str) or not self.target.strip():
            raise BenchHarnessError("--target is required")
        if (
            isinstance(self.hz, bool)
            or not isinstance(self.hz, (int, float))
            or not math.isfinite(float(self.hz))
            or float(self.hz) <= 0
            or float(self.hz) > 10_000
        ):
            raise BenchHarnessError("--hz must be a positive number up to 10000")
        if (
            isinstance(self.warmup, bool)
            or not isinstance(self.warmup, int)
            or self.warmup < 0
        ):
            raise BenchHarnessError("--warmup must be a non-negative integer")
        if (
            isinstance(self.iters, bool)
            or not isinstance(self.iters, int)
            or self.iters < 1
        ):
            raise BenchHarnessError("--iters must be a positive integer")
        if self.frames is not None and (
            isinstance(self.frames, bool)
            or not isinstance(self.frames, int)
            or self.frames < 1
        ):
            raise BenchHarnessError("--frames must be a positive integer")
        if (
            isinstance(self.episode, bool)
            or not isinstance(self.episode, int)
            or self.episode < 0
        ):
            raise BenchHarnessError("--episode must be a non-negative integer")
        if not isinstance(self.robot, str) or not self.robot.strip():
            raise BenchHarnessError("--robot must name a robot id")
        sink = self.sink.strip() if isinstance(self.sink, str) else ""
        if sink and sink not in _SINKS:
            raise BenchHarnessError("--sink must be inprocess, sim, or ros2")
        policy = self.policy.strip() if isinstance(self.policy, str) else ""
        if not policy:
            raise BenchHarnessError("--policy must be a non-empty string")
        workload = self.workload.strip() if isinstance(self.workload, str) else ""
        if not workload:
            raise BenchHarnessError("--workload must be a non-empty string")
        for label, value in (
            ("--dataset", self.dataset),
            ("--bag", self.bag),
            ("--fixture", self.fixture),
        ):
            _reject_device_uri(value, label)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", self.target.strip())
        object.__setattr__(self, "hz", float(self.hz))
        object.__setattr__(self, "robot", self.robot.strip())
        object.__setattr__(self, "sink", sink)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "workload", workload)


@dataclass(frozen=True)
class EncodedFrame:
    """One encoded camera frame. ``payload`` is interpreted by ``encoding``."""

    index: int
    width: int
    height: int
    encoding: str
    payload: bytes
    timestamp: float = 0.0


@dataclass(frozen=True)
class RgbImage:
    """Decoded RGB8 image. ``rgb`` length is ``width * height * 3``."""

    width: int
    height: int
    rgb: bytes


@dataclass(frozen=True)
class CameraStream:
    """Frames the pipeline replays. ``world`` is sim, recorded, or hardware."""

    source: str
    world: str
    frames: tuple[EncodedFrame, ...]
    episode: int = 0
    fps: float = 30.0
    origin: str = "camera"
    label: str = ""
    recording: str = ""


@dataclass(frozen=True)
class ActionLimits:
    """Publish clamp. Matches ``Robot.drive`` on the named spec."""

    max_lin: float
    max_ang: float
    holonomic: bool


class TinyCpuPolicy:
    """Fixed linear map from eight image features to vx, vy, and yaw rate.

    No torch and no downloaded weights. The same features always produce the
    same action.
    """

    name = "tiny-cpu"

    def infer(self, features: Sequence[float]) -> list[float]:
        """Return an unclamped body velocity."""
        if len(features) != _FEATURE_DIM:
            raise BenchHarnessError(
                f"tiny-cpu expects {_FEATURE_DIM} features, got {len(features)}"
            )
        action: list[float] = []
        for row, bias in zip(_TINY_WEIGHTS, _TINY_BIAS, strict=True):
            total = bias
            for weight, feature in zip(row, features, strict=True):
                total += weight * float(feature)
            action.append(total)
        return action

    def digest(self) -> str:
        """sha256 of the canonical weight JSON."""
        payload = json.dumps(
            {
                "bias": list(_TINY_BIAS),
                "weights": [list(row) for row in _TINY_WEIGHTS],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class InProcessSink:
    """Records published actions. Tests and offline replays use this."""

    name = "inprocess"

    def __init__(self) -> None:
        self.actions: list[tuple[float, float, float]] = []

    def publish(self, action: tuple[float, float, float]) -> None:
        """Append one clamped body velocity."""
        self.actions.append((float(action[0]), float(action[1]), float(action[2])))

    def close(self) -> None:
        """Nothing to release."""


class SimSink:
    """Publish into an in-process :class:`~ohho.robot.Robot` on a sim transport."""

    name = "sim"

    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self.actions: list[tuple[float, float, float]] = []

    def publish(self, action: tuple[float, float, float]) -> None:
        """Drive the simulator with the clamped velocity."""
        vx, vy, yaw = float(action[0]), float(action[1]), float(action[2])
        self.actions.append((vx, vy, yaw))
        self.robot.drive(vx=vx, vy=vy, w=yaw)

    def close(self) -> None:
        """The caller closes the robot."""


class Ros2PublishSink:
    """Publish ``geometry_msgs/Twist`` on a topic. ``rclpy`` is imported here.

    ``factory`` returns ``(send, close)``. Tests pass a fake factory. The
    default factory imports ROS 2 and raises :class:`BenchHarnessError` when
    it is not installed.
    """

    name = "ros2"

    def __init__(
        self,
        topic: str = "/cmd_vel",
        *,
        factory: Callable[[str], tuple[Callable[..., None], Callable[[], None]]]
        | None = None,
    ) -> None:
        self.topic = topic
        self.actions: list[tuple[float, float, float]] = []
        build = factory if factory is not None else _rclpy_twist_publisher
        self._send, self._close = build(topic)

    def publish(self, action: tuple[float, float, float]) -> None:
        """Send one Twist and record it."""
        vx, vy, yaw = float(action[0]), float(action[1]), float(action[2])
        self.actions.append((vx, vy, yaw))
        self._send(vx, vy, yaw)

    def close(self) -> None:
        """Release the publisher."""
        self._close()


class PipelineRunner:
    """One frame through the seven pipeline stages."""

    def __init__(
        self,
        frame: EncodedFrame | None,
        camera: Callable[[], EncodedFrame] | None,
        policy: Any,
        sink: Any,
        limits: ActionLimits,
    ) -> None:
        self._frame = frame
        self._camera = camera
        self._policy = policy
        self._sink = sink
        self._limits = limits

    def decode(self) -> RgbImage:
        """Decode the recorded frame, or the sim camera, to RGB8."""
        frame = self._frame if self._frame is not None else self._capture()
        return decode_frame(frame)

    def preprocess(self, image: RgbImage) -> list[float]:
        """Reduce RGB8 to the eight features the tiny policy reads."""
        return features_from_rgb(image)

    def h2d(self, features: Sequence[float]) -> list[float]:
        """Host copy. CUDA copies are a follow-up and are not attempted."""
        return [float(value) for value in features]

    def inference(self, features: Sequence[float]) -> list[float]:
        """Run the CPU policy."""
        return [float(value) for value in self._policy.infer(features)]

    def d2h(self, action: Sequence[float]) -> list[float]:
        """Host copy of the action. CUDA copies are a follow-up."""
        return [float(value) for value in action]

    def postprocess(self, action: Sequence[float]) -> tuple[float, float, float]:
        """Clamp to the robot spec before anything is published."""
        return clamp_action(action, self._limits)

    def publish(self, action: tuple[float, float, float]) -> tuple[float, float, float]:
        """Hand the clamped velocity to the sink."""
        self._sink.publish(action)
        return action

    def _capture(self) -> EncodedFrame:
        if self._camera is None:
            raise BenchHarnessError("pipeline frame is missing")
        return self._camera()


def execute_pipeline(
    config: PipelineConfig,
    *,
    clock: Any = None,
    costs: Mapping[str, float] | None = None,
    policy: Any = None,
    sink: Any = None,
    ros_factory: Callable[[str], tuple[Callable[..., None], Callable[[], None]]]
    | None = None,
    baseline_path: str | None = None,
    slos: Sequence[str] = (),
    max_regressions: Sequence[str] = (),
    parity_path: str | None = None,
) -> dict[str, Any]:
    """Replay ``config`` and return a schema-valid ohho.bench/v1 document.

    ``clock`` defaults to :class:`~ohho.bench.clock.PerfClock`. Pass a
    :class:`~ohho.bench.clock.VirtualClock` with ``costs`` (seconds per stage)
    for an exact scripted run. ``costs`` on any other clock is rejected.
    A hardware source is refused unless ``allow_hardware`` is set, and even
    then this function only replays a recording.
    """
    check_gate_expressions(slos, max_regressions, has_baseline=bool(baseline_path))
    _reject_hardware(config)
    if clock is None:
        clock = PerfClock()
    if costs is not None and getattr(clock, "kind", "") != "virtual":
        raise BenchHarnessError("stage costs require a virtual clock")
    stage_costs = _stage_costs(costs)
    try:
        profile = get_profile(config.target)
    except TargetProfileError as exc:
        raise BenchHarnessError(str(exc)) from exc
    chosen = policy if policy is not None else _policy(config)
    digest = _require_digest(chosen)
    resolved_sink = _resolved_sink(config)
    stream = None if config.source == "sim" else _load_stream(config)
    wanted = _wanted_frames(config, stream)
    robot, transport = (None, None)
    owns_sink = sink is None
    opened: Any = None
    trace = MemoryTrace()
    sim_pose: tuple[float, float] | None = None
    try:
        if config.source == "sim" or (sink is None and resolved_sink == "sim"):
            robot, transport = _open_sim(config.robot)
            opened = robot
        if sink is None:
            sink = _make_sink(resolved_sink, robot, ros_factory)
            sink_name = resolved_sink
        else:
            sink_name = str(getattr(sink, "name", "") or resolved_sink)
        limits = _limits_for(robot, config.robot)
        trace.start()
        started = _stamp()
        samples = _replay(
            config,
            clock=clock,
            costs=stage_costs,
            stream=stream,
            wanted=wanted,
            robot=robot,
            transport=transport,
            policy=chosen,
            sink=sink,
            limits=limits,
        )
        if transport is not None:
            tele = transport.read()
            if tele.odom is not None:
                sim_pose = (float(tele.odom.x), float(tele.odom.y))
        memory = trace.snapshot()
        finished = _stamp()
    finally:
        trace.close()
        if owns_sink and sink is not None:
            close = getattr(sink, "close", None)
            if callable(close):
                close()
        if opened is not None:
            opened.stop()
            opened.disconnect()
    document = _document(
        config,
        profile=profile,
        clock=clock,
        samples=samples,
        policy=chosen,
        digest=digest,
        sink_name=getattr(sink, "name", sink_name),
        stream=stream,
        memory=memory,
        started=started,
        finished=finished,
        sim_pose=sim_pose,
    )
    return finalize_result(
        document,
        baseline_path=baseline_path,
        slos=slos,
        max_regressions=max_regressions,
        parity_path=parity_path,
    )


def decode_frame(frame: EncodedFrame) -> RgbImage:
    """Decode ``rgb8``, ``ppm``, or, when Pillow is installed, JPEG/PNG."""
    encoding = frame.encoding.lower()
    if encoding == "rgb8":
        _edge(frame.width, "width")
        _edge(frame.height, "height")
        expected = frame.width * frame.height * 3
        if len(frame.payload) != expected:
            raise BenchHarnessError(
                f"rgb8 frame {frame.index} has {len(frame.payload)} bytes, "
                f"expected {expected}"
            )
        return RgbImage(frame.width, frame.height, frame.payload)
    if encoding == "ppm":
        image = _decode_ppm(frame.payload)
        if frame.width and frame.height:
            if image.width != frame.width or image.height != frame.height:
                raise BenchHarnessError(
                    f"PPM frame {frame.index} is {image.width}x{image.height}, "
                    f"not {frame.width}x{frame.height}"
                )
        return image
    if encoding in {"jpeg", "jpg", "png"}:
        return _decode_pillow(frame.payload)
    raise BenchHarnessError(f"unsupported frame encoding {frame.encoding!r}")


def features_from_rgb(image: RgbImage) -> list[float]:
    """Eight features in ``[0, 1]``: mean RGB, four region brightnesses, center."""
    width, height, rgb = image.width, image.height, image.rgb
    count = width * height
    sum_r = sum_g = sum_b = 0
    left = right = top = bottom = 0
    left_n = right_n = top_n = bottom_n = 0
    mid_x = width / 2.0
    mid_y = height / 2.0
    for y in range(height):
        row = y * width
        for x in range(width):
            offset = (row + x) * 3
            red = rgb[offset]
            green = rgb[offset + 1]
            blue = rgb[offset + 2]
            sum_r += red
            sum_g += green
            sum_b += blue
            bright = red + green + blue
            if x < mid_x:
                left += bright
                left_n += 1
            else:
                right += bright
                right_n += 1
            if y < mid_y:
                top += bright
                top_n += 1
            else:
                bottom += bright
                bottom_n += 1
    center_i = ((height // 2) * width + (width // 2)) * 3
    center = (rgb[center_i] + rgb[center_i + 1] + rgb[center_i + 2]) / (3.0 * 255.0)
    return [
        sum_r / count / 255.0,
        sum_g / count / 255.0,
        sum_b / count / 255.0,
        _unit(left, left_n),
        _unit(right, right_n),
        _unit(top, top_n),
        _unit(bottom, bottom_n),
        center,
    ]


def clamp_action(
    action: Sequence[float], limits: ActionLimits
) -> tuple[float, float, float]:
    """Clamp vx, vy, and yaw rate. Non-holonomic vy is zero."""
    vx = clamp(_component(action, 0), -limits.max_lin, limits.max_lin)
    vy = clamp(_component(action, 1), -limits.max_lin, limits.max_lin)
    yaw = clamp(_component(action, 2), -limits.max_ang, limits.max_ang)
    if not limits.holonomic:
        vy = 0.0
    return (vx, vy, yaw)


def paint_sim_frame(
    robot_x: float,
    robot_y: float,
    *,
    width: int = _SIM_WIDTH,
    height: int = _SIM_HEIGHT,
) -> bytes:
    """Stdlib stand-in for the sim front camera: sky, floor, and a red target."""
    target_x, target_y = 2.0, 0.2
    dist = math.hypot(target_x - robot_x, target_y - robot_y)
    box = max(1, min(height // 2, int(6.0 / max(dist, 0.3))))
    center_x = width // 2
    center_y = height // 2
    horizon = height // 2
    out = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            offset = (y * width + x) * 3
            if y >= horizon:
                out[offset : offset + 3] = b"\x8c\x8c\x8c"
            else:
                out[offset : offset + 3] = b"\xc8\xc8\xc8"
            if abs(x - center_x) <= box and abs(y - center_y) <= box:
                out[offset : offset + 3] = b"\xdc\x1e\x1e"
    return bytes(out)


@dataclass
class _Samples:
    totals_s: list[float]
    stages: dict[str, list[float]]
    cold_s: float
    publish_times: list[float]
    presented: int
    published: int
    dropped: int
    late: bool
    origin: float
    finished: float


def _replay(
    config: PipelineConfig,
    *,
    clock: Any,
    costs: dict[str, float] | None,
    stream: CameraStream | None,
    wanted: int,
    robot: Any,
    transport: Any,
    policy: Any,
    sink: Any,
    limits: ActionLimits,
) -> _Samples:
    frames = None if stream is None else stream.frames[:wanted]
    presented = wanted if frames is None else len(frames)
    period = 1.0 / config.hz
    origin = float(clock.now())
    dropped = 0
    late = False
    processed = 0
    cold_s: float | None = None
    totals: list[float] = []
    stages = {name: [] for name in PIPELINE_STAGE_NAMES}
    publish_times: list[float] = []
    for index in range(presented):
        slot = origin + index * period
        deadline = slot + period
        now = float(clock.now())
        # A few microseconds past the slot start is still this frame. Drop only
        # after the whole control period has elapsed, otherwise the first
        # perf_counter read always looks late.
        if now > deadline + _EPS:
            dropped += 1
            _step(transport, period)
            continue
        if now < slot - _EPS:
            _sleep_until(clock, slot)
        frame = None if frames is None else frames[index]
        camera = None if frame is not None else _sim_camera(robot, index, config.hz)
        runner = PipelineRunner(frame, camera, policy, sink, limits)
        sample = _measure(runner, clock, costs)
        end = float(clock.now())
        if end > slot + period + _EPS:
            late = True
        elapsed = sum(sample[name] for name in PIPELINE_STAGE_NAMES)
        if cold_s is None:
            cold_s = elapsed
        if processed >= config.warmup:
            for name in PIPELINE_STAGE_NAMES:
                stages[name].append(sample[name])
            totals.append(elapsed)
        publish_times.append(end)
        processed += 1
        _step(transport, period)
    if not totals or cold_s is None:
        raise BenchHarnessError(
            "pipeline produced no measured frames "
            f"({dropped} dropped, warmup {config.warmup}). "
            "Lower --warmup or the stage cost, or raise --hz."
        )
    return _Samples(
        totals_s=totals,
        stages=stages,
        cold_s=cold_s,
        publish_times=publish_times,
        presented=presented,
        published=processed,
        dropped=dropped,
        late=late,
        origin=origin,
        finished=float(clock.now()),
    )


def _measure(
    runner: PipelineRunner, clock: Any, costs: dict[str, float] | None
) -> dict[str, float]:
    durations: dict[str, float] = {}
    value: Any = None
    for name in PIPELINE_STAGE_NAMES:
        started = float(clock.now())
        _apply_cost(clock, name, costs)
        method = getattr(runner, name)
        value = method() if name == "decode" else method(value)
        durations[name] = max(0.0, float(clock.now()) - started)
    return durations


def _document(
    config: PipelineConfig,
    *,
    profile: Any,
    clock: Any,
    samples: _Samples,
    policy: Any,
    digest: str,
    sink_name: str,
    stream: CameraStream | None,
    memory: Any,
    started: str,
    finished: str,
    sim_pose: tuple[float, float] | None = None,
) -> dict[str, Any]:
    latency = _latency_report(samples)
    achieved = _achieved_hz(samples.publish_times)
    window = max(0.0, samples.finished - samples.origin)
    world = "sim" if stream is None else stream.world
    origin = "sim" if stream is None else stream.origin
    recording = "" if stream is None else stream.recording
    source = config.source if stream is None else stream.source
    artifact = (
        digest
        if stream is None or not stream.label
        else _digest_path(Path(stream.label))
    )
    manifest = _manifest(config, stream)
    environment, gpu = collect_environment(
        profile,
        timing_source=str(getattr(clock, "kind", "perf_counter")),
        execution_provider="tiny-cpu",
        execution_providers=("tiny-cpu",),
        timing_note=_GPU_NOTE,
    )
    environment["world"] = world
    telemetry: dict[str, Any] = {
        "duration_s": window,
        "samples": len(samples.totals_s),
    }
    telemetry.update(memory.as_telemetry())
    telemetry.update(gpu_telemetry(gpu))
    pipeline: dict[str, Any] = {
        "version": PIPELINE_SCHEMA_ID,
        "world": world,
        "source": source,
        "target_hz": config.hz,
        "achieved_hz": achieved,
        "rate_sustained": samples.dropped == 0 and not samples.late,
        "dropped_frames": samples.dropped,
        "frames_presented": samples.presented,
        "frames_published": samples.published,
        "jitter_ms": latency["std"],
        "interval_jitter_ms": _interval_jitter_ms(samples.publish_times),
        "control_period_ms": 1000.0 / config.hz,
        "sink": sink_name,
        "device": "cpu",
        "live_robot": False,
        "gpu_follow_up": list(_GPU_FOLLOW_UP),
        "gpu_note": _GPU_NOTE,
        "policy": str(getattr(policy, "name", config.policy)),
        "episode": 0 if stream is None else stream.episode,
        "frame_origin": origin,
    }
    if recording:
        pipeline["recording"] = recording
    if sim_pose is not None:
        pipeline["sim_x"] = sim_pose[0]
        pipeline["sim_y"] = sim_pose[1]
    batch = {
        "batch_size": 1,
        "latency": latency,
        "throughput": {
            "value": achieved,
            "unit": "frames/s",
            "samples": len(samples.totals_s),
            "batch_size": 1,
            "duration_s": window,
        },
    }
    return {
        "schema": SCHEMA_ID,
        "run": {
            "id": f"pipeline-{uuid.uuid4().hex[:12]}",
            "model_sha": digest,
            "artifact_sha": artifact,
            "target": config.workload,
            "manifest_ref": manifest,
            "host": host_block(profile),
            "started_at": started,
            "finished_at": finished,
        },
        "latency": latency,
        "throughput": batch["throughput"],
        "accuracy": {
            "primary": {"name": "not_measured", "value": 0.0, "unit": "1"},
            "samples": 0,
        },
        "telemetry": telemetry,
        "environment": environment,
        "batches": [batch],
        "pipeline": pipeline,
    }


def _latency_report(samples: _Samples) -> dict[str, Any]:
    stats = summarize([item * 1000.0 for item in samples.totals_s])
    stages = []
    for name in PIPELINE_STAGE_NAMES:
        block = summarize([item * 1000.0 for item in samples.stages[name]])
        block["name"] = name
        block["unit"] = "ms"
        block["samples"] = len(samples.stages[name])
        stages.append(block)
    mean_s = stats["mean"] / 1000.0
    stats["unit"] = "ms"
    stats["samples"] = len(samples.totals_s)
    stats["stages"] = stages
    stats["cold_start_ms"] = samples.cold_s * 1000.0
    stats["achievable_hz"] = 0.0 if mean_s == 0.0 else 1.0 / mean_s
    return stats


def _wanted_frames(config: PipelineConfig, stream: CameraStream | None) -> int:
    wanted = config.warmup + config.iters
    if config.frames is not None:
        wanted = min(wanted, config.frames)
    if stream is not None:
        wanted = min(wanted, len(stream.frames))
    if wanted <= config.warmup:
        available = "the sim" if stream is None else f"{len(stream.frames)} frames"
        raise BenchHarnessError(
            f"pipeline needs at least one measured frame after warmup "
            f"({available}, warmup {config.warmup})"
        )
    return wanted


def _load_stream(config: PipelineConfig) -> CameraStream:
    if config.source == "fixture":
        if not config.fixture:
            raise BenchHarnessError("--source fixture needs --fixture PATH")
        return load_fixture(Path(config.fixture))
    if config.source == "lerobot":
        if not config.dataset:
            raise BenchHarnessError("--source lerobot needs --dataset PATH")
        return load_lerobot(Path(config.dataset), config.episode)
    if config.source == "rosbag":
        if not config.bag:
            raise BenchHarnessError("--source rosbag needs --bag PATH")
        return load_rosbag(Path(config.bag))
    kind, path = _hardware_recording(config)
    stream = _load_kind(kind, Path(path), config.episode)
    return replace(stream, world="hardware", source="hardware", recording=kind)


def load_fixture(path: Path) -> CameraStream:
    """Load an ``ohho.pipeline.stream/v1`` JSON file."""
    data = _read_json(path, "fixture")
    if data.get("schema") != STREAM_SCHEMA_ID:
        raise BenchHarnessError(f"fixture schema must be {STREAM_SCHEMA_ID} ({path})")
    return _stream_from_document(data, source="fixture", label=str(path))


def load_lerobot(path: Path, episode: int) -> CameraStream:
    """Load one LeRobot episode. A missing camera becomes a state raster."""
    if not path.is_dir():
        raise BenchHarnessError(f"dataset not found: {path}")
    from ohho.data.reader import DatasetReader

    try:
        reader = DatasetReader(str(path))
        rows = reader.load_episode(episode)
    except FileNotFoundError as exc:
        raise BenchHarnessError(str(exc)) from exc
    except OSError as exc:
        raise BenchHarnessError(f"cannot read dataset {path}: {exc}") from exc
    if not rows:
        raise BenchHarnessError(f"episode {episode} has no frames")
    images, origin = _episode_images(path, reader, episode, rows)
    frames = tuple(
        EncodedFrame(
            index=index,
            width=image.width,
            height=image.height,
            encoding="rgb8",
            payload=image.rgb,
            timestamp=float(rows[index].timestamp),
        )
        for index, image in enumerate(images)
    )
    fps = float(reader.info.fps or 30.0)
    return CameraStream(
        source="lerobot",
        world=_dataset_world(reader),
        frames=frames,
        episode=episode,
        fps=fps if fps > 0 else 30.0,
        origin=origin,
        label=str(path),
    )


def load_rosbag(path: Path) -> CameraStream:
    """Load an ``ohho.pipeline.rosbag/v1`` recording.

    A ROS 2 bag directory (``metadata.yaml``) or an ``.mcap`` file is refused
    with an install-or-convert message. This function does not download bags
    and does not decode ``sensor_msgs/Image``.
    """
    if path.is_file() and path.suffix.lower() == ".mcap":
        raise BenchHarnessError(
            "MCAP bags are not decoded in this CPU bench. Convert the "
            f"recording to {BAG_SCHEMA_ID}."
        )
    if path.is_file():
        data = _read_json(path, "rosbag")
        if data.get("schema") != BAG_SCHEMA_ID:
            raise BenchHarnessError(f"rosbag schema must be {BAG_SCHEMA_ID} ({path})")
        return _stream_from_document(data, source="rosbag", label=str(path))
    if not path.is_dir():
        raise BenchHarnessError(f"rosbag not found: {path}")
    meta_path = path / "metadata.json"
    if meta_path.is_file():
        meta = _read_json(meta_path, "rosbag metadata")
        if meta.get("schema") != BAG_SCHEMA_ID:
            raise BenchHarnessError(
                f"rosbag metadata schema must be {BAG_SCHEMA_ID} ({meta_path})"
            )
        frames = _frames_from_jsonl(path / "frames.jsonl")
        return CameraStream(
            source="rosbag",
            world=_world_name(meta.get("world", "recorded")),
            frames=frames,
            fps=_positive_fps(meta.get("fps", 30.0)),
            origin="camera",
            label=str(path),
        )
    if (path / "metadata.yaml").is_file() or (path / "metadata.yml").is_file():
        _refuse_rosbag2(path)
    raise BenchHarnessError(
        f"{path} is not an {BAG_SCHEMA_ID} directory (metadata.json + frames.jsonl)"
    )


def _stream_from_document(
    data: Mapping[str, Any], *, source: str, label: str
) -> CameraStream:
    raw_frames = data.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise BenchHarnessError(f"{label} has no frames")
    frames = tuple(
        _frame_from_json(index, item) for index, item in enumerate(raw_frames)
    )
    origin = data.get("frame_origin", "camera")
    if not isinstance(origin, str) or not origin.strip():
        raise BenchHarnessError(f"{label} frame_origin must be a string")
    return CameraStream(
        source=source,
        world=_world_name(data.get("world", "recorded")),
        frames=frames,
        episode=_nonnegative_int(data.get("episode", 0), "episode"),
        fps=_positive_fps(data.get("fps", 30.0)),
        origin=origin.strip(),
        label=label,
    )


def _episode_images(
    root: Path, reader: Any, episode: int, rows: Sequence[Any]
) -> tuple[list[RgbImage], str]:
    video = _video_frames(reader, episode, len(rows))
    if video is not None:
        if len(video) < len(rows):
            raise BenchHarnessError(
                f"episode {episode} video has {len(video)} frames for {len(rows)} rows"
            )
        return video[: len(rows)], "camera"
    inline = _jsonl_images(root, episode)
    if inline:
        if len(inline) != len(rows) or any(item is None for item in inline):
            raise BenchHarnessError(
                f"episode {episode} is missing observation.images.front on a frame"
            )
        return [item for item in inline if item is not None], "camera"
    return [
        _rasterize_state(list(row.observation_state)) for row in rows
    ], "state-raster"


def _video_frames(reader: Any, episode: int, limit: int) -> list[RgbImage] | None:
    path = reader.get_video_path(episode)
    if not path.is_file():
        return None
    decoded = _decode_video(path, limit)
    return decoded


def _decode_video(path: Path, limit: int) -> list[RgbImage]:
    try:
        import av
    except ImportError:
        av = None
    if av is not None:
        images: list[RgbImage] = []
        try:
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    array = frame.to_ndarray(format="rgb24")
                    height, width = int(array.shape[0]), int(array.shape[1])
                    _edge(width, "width")
                    _edge(height, "height")
                    images.append(RgbImage(width, height, bytes(array.tobytes())))
                    if len(images) >= limit:
                        break
        except Exception as exc:
            raise BenchHarnessError(f"cannot decode video {path}: {exc}") from exc
        if not images:
            raise BenchHarnessError(f"video {path} has no frames")
        return images
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise BenchHarnessError(
            f"{path} is an encoded camera video, and neither PyAV nor imageio "
            "is installed. This bench does not download them."
        ) from exc
    images = []
    try:
        for array in iio.imiter(path):
            height, width = int(array.shape[0]), int(array.shape[1])
            _edge(width, "width")
            _edge(height, "height")
            flat = array[:, :, :3].tobytes()
            images.append(RgbImage(width, height, flat))
            if len(images) >= limit:
                break
    except Exception as exc:
        raise BenchHarnessError(f"cannot decode video {path}: {exc}") from exc
    if not images:
        raise BenchHarnessError(f"video {path} has no frames")
    return images


def _jsonl_images(root: Path, episode: int) -> list[RgbImage | None]:
    chunk = episode // 1000
    file = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.jsonl"
    if not file.is_file():
        return []
    images: list[RgbImage | None] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        raw = row.get("observation.images.front")
        images.append(None if raw is None else _image_field(raw))
    return images


def _image_field(raw: Any) -> RgbImage:
    if isinstance(raw, dict):
        encoding = str(raw.get("encoding", "rgb8"))
        width = _edge(raw.get("width"), "width") if "width" in raw else 0
        height = _edge(raw.get("height"), "height") if "height" in raw else 0
        payload = _payload_bytes(raw.get("data"))
        return decode_frame(EncodedFrame(0, width, height, encoding, payload))
    if isinstance(raw, str):
        payload = _decode_b64(raw)
        if len(payload) % 3 != 0:
            raise BenchHarnessError("observation.images.front is not RGB8")
        side = int(math.sqrt(len(payload) // 3))
        if side * side * 3 != len(payload):
            raise BenchHarnessError(
                "observation.images.front base64 must be a square RGB8 image "
                "or an object with width, height, and data"
            )
        return RgbImage(side, side, payload)
    raise BenchHarnessError("observation.images.front must be a string or object")


def _rasterize_state(state: Sequence[float]) -> RgbImage:
    width, height = 4, 4
    out = bytearray(width * height * 3)
    values = list(state) or [0.0]
    for index in range(width * height):
        value = float(values[index % len(values)])
        byte = int(max(0.0, min(255.0, round((value + 1.0) * 80.0))))
        offset = index * 3
        out[offset] = byte
        out[offset + 1] = (byte * 3) & 255
        out[offset + 2] = (byte * 7) & 255
    return RgbImage(width, height, bytes(out))


def _dataset_world(reader: Any) -> str:
    repo = str(reader.info.repo_id or "").lower()
    robot = str(reader.info.robot_type or "").lower()
    if robot == "sim" or repo.startswith("local/sim"):
        return "sim"
    return "recorded"


def _refuse_rosbag2(path: Path) -> None:
    try:
        import rosbag2_py  # noqa: F401
    except ImportError as exc:
        raise BenchHarnessError(
            "This is a ROS 2 bag directory. rosbag2 is not installed, and this "
            "bench does not download it. Replay an "
            f"{BAG_SCHEMA_ID} directory instead ({path})."
        ) from exc
    raise BenchHarnessError(
        "rosbag2 is installed, but this CPU bench does not decode "
        "sensor_msgs/Image. Convert the recording to "
        f"{BAG_SCHEMA_ID} (metadata.json + frames.jsonl)."
    )


def _open_sim(robot_id: str) -> tuple[Any, Any]:
    from ohho.adapters.sim import SimTransport
    from ohho.registry import UnknownRobot, get_spec
    from ohho.robot import Robot
    from ohho.robot_spec import RobotClass
    from ohho.runtime import NativeRuntime

    try:
        spec = get_spec(robot_id)
    except UnknownRobot as exc:
        raise BenchHarnessError(f"unknown robot {robot_id!r}") from exc
    if spec.robot_class is RobotClass.AERIAL:
        raise BenchHarnessError(
            f"{spec.id} is an aerial spec. The pipeline bench does not fly it. "
            "Pass --robot sim."
        )
    transport = SimTransport(spec)
    transport.connect()
    robot = Robot(spec, transport, NativeRuntime())
    if spec.has("legged.pose"):
        transport.sport_command("stand_up")
    return robot, transport


def _sim_camera(robot: Any, index: int, hz: float) -> Callable[[], EncodedFrame]:
    def capture() -> EncodedFrame:
        tele = robot.telemetry()
        odom = tele.odom
        x = float(odom.x) if odom is not None else 0.0
        y = float(odom.y) if odom is not None else 0.0
        return EncodedFrame(
            index=index,
            width=_SIM_WIDTH,
            height=_SIM_HEIGHT,
            encoding="rgb8",
            payload=paint_sim_frame(x, y),
            timestamp=index / hz,
        )

    return capture


def _limits_for(robot: Any, robot_id: str) -> ActionLimits:
    from ohho import capabilities as caps
    from ohho.registry import UnknownRobot, get_spec

    spec = robot.spec if robot is not None else None
    if spec is None:
        try:
            spec = get_spec(robot_id)
        except UnknownRobot as exc:
            raise BenchHarnessError(f"unknown robot {robot_id!r}") from exc
    return ActionLimits(
        max_lin=float(spec.max_lin),
        max_ang=float(spec.max_ang),
        holonomic=bool(spec.has(caps.BASE_HOLONOMIC)),
    )


def _make_sink(name: str, robot: Any, ros_factory: Any) -> Any:
    if name == "inprocess":
        return InProcessSink()
    if name == "sim":
        if robot is None:
            raise BenchHarnessError("--sink sim needs the in-process simulator")
        return SimSink(robot)
    if name == "ros2":
        return Ros2PublishSink(factory=ros_factory)
    raise BenchHarnessError(f"unknown sink {name!r}")


def _policy(config: PipelineConfig) -> TinyCpuPolicy:
    if config.policy != TinyCpuPolicy.name:
        raise BenchHarnessError(
            f"unknown policy {config.policy!r}. The built-in CPU policy is "
            "tiny-cpu. Loading an exported model is not part of this bench."
        )
    return TinyCpuPolicy()


def _resolved_sink(config: PipelineConfig) -> str:
    if config.sink:
        return config.sink
    if config.source == "sim":
        return "sim"
    return "inprocess"


def _reject_hardware(config: PipelineConfig) -> None:
    if config.source != "hardware":
        return
    if not config.allow_hardware:
        raise BenchHarnessError(
            "real-robot pipeline bench is refused unless --allow-hardware is set. "
            "This process does not open a robot."
        )
    _hardware_recording(config)


def _hardware_recording(config: PipelineConfig) -> tuple[str, str]:
    found: list[tuple[str, str]] = []
    if config.fixture:
        found.append(("fixture", config.fixture))
    if config.dataset:
        found.append(("lerobot", config.dataset))
    if config.bag:
        found.append(("rosbag", config.bag))
    if len(found) != 1:
        raise BenchHarnessError(
            "refusing to open a robot. Pass one of --fixture, --dataset, or "
            "--bag to replay a recording with --allow-hardware."
        )
    return found[0]


def _load_kind(kind: str, path: Path, episode: int) -> CameraStream:
    if kind == "fixture":
        return load_fixture(path)
    if kind == "lerobot":
        return load_lerobot(path, episode)
    return load_rosbag(path)


def _rclpy_twist_publisher(
    topic: str,
) -> tuple[Callable[..., None], Callable[[], None]]:
    try:
        import rclpy
        from geometry_msgs.msg import Twist
    except ImportError as exc:
        raise BenchHarnessError(
            "ROS 2 publish needs rclpy and geometry_msgs, which are not "
            "installed. Use --sink inprocess or --sink sim."
        ) from exc
    if not rclpy.ok():
        rclpy.init(args=None)
    node = rclpy.create_node("ohho_bench_pipeline")
    publisher = node.create_publisher(Twist, topic, 10)

    def send(vx: float, vy: float, yaw: float) -> None:
        message = Twist()
        message.linear.x = float(vx)
        message.linear.y = float(vy)
        message.angular.z = float(yaw)
        publisher.publish(message)

    def close() -> None:
        node.destroy_node()

    return send, close


def _decode_ppm(payload: bytes) -> RgbImage:
    if not payload.startswith(b"P6"):
        raise BenchHarnessError("PPM frame must be a P6 image")
    index = 2
    values: list[int] = []
    size = len(payload)
    while len(values) < 3 and index < size:
        byte = payload[index]
        if byte in b" \t\r\n":
            index += 1
            continue
        if byte == ord("#"):
            while index < size and payload[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < size and payload[index] not in b" \t\r\n#":
            index += 1
        token = payload[start:index]
        if not token.isdigit():
            raise BenchHarnessError("PPM header is not numeric")
        values.append(int(token))
    if len(values) != 3:
        raise BenchHarnessError("PPM header is incomplete")
    if values[2] != 255:
        raise BenchHarnessError("PPM maxval must be 255")
    width, height = _edge(values[0], "width"), _edge(values[1], "height")
    if index < size and payload[index] in b" \t\r\n":
        index += 1
    expected = width * height * 3
    data = payload[index : index + expected]
    if len(data) != expected:
        raise BenchHarnessError("PPM frame is truncated")
    return RgbImage(width, height, data)


def _decode_pillow(payload: bytes) -> RgbImage:
    try:
        from PIL import Image
    except ImportError as exc:
        raise BenchHarnessError(
            "JPEG and PNG frames need Pillow, which is not installed. Use rgb8 or PPM."
        ) from exc
    import io

    try:
        with Image.open(io.BytesIO(payload)) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            _edge(width, "width")
            _edge(height, "height")
            return RgbImage(width, height, rgb.tobytes())
    except BenchHarnessError:
        raise
    except Exception as exc:
        raise BenchHarnessError(f"cannot decode image: {exc}") from exc


def _frame_from_json(index: int, item: Any) -> EncodedFrame:
    if not isinstance(item, dict):
        raise BenchHarnessError(f"frame {index} must be an object")
    encoding = str(item.get("encoding", "rgb8")).lower()
    payload = _payload_bytes(item.get("data"))
    width = _edge(item.get("width", 0), "width") if item.get("width") else 0
    height = _edge(item.get("height", 0), "height") if item.get("height") else 0
    if encoding == "rgb8" and (width < 1 or height < 1):
        raise BenchHarnessError(f"frame {index} needs width and height")
    timestamp = item.get("timestamp", index)
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise BenchHarnessError(f"frame {index} timestamp must be a number")
    return EncodedFrame(
        index=index,
        width=width,
        height=height,
        encoding=encoding,
        payload=payload,
        timestamp=float(timestamp),
    )


def _frames_from_jsonl(path: Path) -> tuple[EncodedFrame, ...]:
    if not path.is_file():
        raise BenchHarnessError(f"rosbag frames not found: {path}")
    frames: list[EncodedFrame] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchHarnessError(f"{path} is not JSON: {exc.msg}") from exc
        frames.append(_frame_from_json(len(frames), item))
    if not frames:
        raise BenchHarnessError(f"{path} has no frames")
    return tuple(frames)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BenchHarnessError(f"{label} not found: {path}") from exc
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise BenchHarnessError(f"cannot read {label}: {detail}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BenchHarnessError(
            f"{label} is not JSON: {exc.msg} (line {exc.lineno})"
        ) from exc
    if not isinstance(data, dict):
        raise BenchHarnessError(f"{label} must be a JSON object")
    return data


def _payload_bytes(value: Any) -> bytes:
    if isinstance(value, str):
        return _decode_b64(value)
    raise BenchHarnessError("frame data must be base64")


def _decode_b64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:
        raise BenchHarnessError("frame data is not valid base64") from exc


def _step(transport: Any, period: float) -> None:
    if transport is None:
        return
    transport.step(period)


def _sleep_until(clock: Any, target: float) -> None:
    fn = getattr(clock, "sleep_until", None)
    if callable(fn):
        fn(target)
        return
    now = float(clock.now())
    if now >= target:
        return
    advance = getattr(clock, "advance", None)
    if callable(advance):
        advance(target - now)


def _apply_cost(clock: Any, name: str, costs: dict[str, float] | None) -> None:
    if not costs:
        return
    advance = getattr(clock, "advance", None)
    if not callable(advance):
        return
    advance(costs.get(name, 0.0))


def _stage_costs(costs: Mapping[str, float] | None) -> dict[str, float] | None:
    if costs is None:
        return None
    parsed: dict[str, float] = {}
    for name, value in costs.items():
        if name not in PIPELINE_STAGE_NAMES:
            raise BenchHarnessError(f"unknown stage cost {name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BenchHarnessError(f"stage cost {name} must be a number")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise BenchHarnessError(f"stage cost {name} must be non-negative")
        parsed[name] = number
    return parsed


def _achieved_hz(times: Sequence[float]) -> float:
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    if span <= 0:
        return 0.0
    return (len(times) - 1) / span


def _interval_jitter_ms(times: Sequence[float]) -> float:
    if len(times) < 3:
        return 0.0
    deltas = [times[index + 1] - times[index] for index in range(len(times) - 1)]
    return statistics.pstdev(deltas) * 1000.0


def _manifest(config: PipelineConfig, stream: CameraStream | None) -> str:
    if stream is not None and stream.label:
        return stream.label
    if config.source == "sim":
        return f"sim://{config.robot}"
    return config.source


def _digest_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    elif path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        for file in files:
            digest.update(file.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(file.read_bytes())
            digest.update(b"\0")
    else:
        digest.update(str(path).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _require_digest(policy: Any) -> str:
    digest = policy.digest()
    if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
        raise BenchHarnessError("policy digest must be a hex sha256")
    return digest


def _reject_device_uri(value: str | None, label: str) -> None:
    if not isinstance(value, str) or "://" not in value:
        return
    scheme = value.split("://", 1)[0].strip().lower()
    if scheme in _DEVICE_SCHEMES:
        raise BenchHarnessError(
            f"{label} looks like a robot transport ({value!r}). "
            "The pipeline bench does not open a robot."
        )


def _world_name(value: Any) -> str:
    if not isinstance(value, str) or value not in _WORLDS:
        raise BenchHarnessError("world must be sim, recorded, or hardware")
    return value


def _positive_fps(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchHarnessError("fps must be a positive number")
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0:
        raise BenchHarnessError("fps must be a positive number")
    return fps


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchHarnessError(f"{label} must be a non-negative integer")
    return value


def _edge(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchHarnessError(f"{label} must be an integer")
    if value < 1 or value > _MAX_EDGE:
        raise BenchHarnessError(f"{label} must be from 1 to {_MAX_EDGE}")
    return value


def _component(action: Sequence[float], index: int) -> float:
    if index >= len(action):
        return 0.0
    value = float(action[index])
    if not math.isfinite(value):
        raise BenchHarnessError("policy action must be finite")
    return value


def _unit(total: int, count: int) -> float:
    if count <= 0:
        return 0.0
    return (total / count) / (3.0 * 255.0)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
