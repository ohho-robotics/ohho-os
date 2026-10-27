"""Perception — detections that feed the spatio-temporal memory.

Two perceptors behind one shape:

- :class:`SimPerceptor` — deterministic, dependency-free: "sees" the labeled
  objects of the sim world within a field of view and range (with range/bearing
  computed from the robot's pose), so the perceive→remember loop is fully
  testable without hardware.
- :class:`VlmPerceptor` — optional Claude-vision detector for real camera
  frames: give it JPEG/PNG bytes, get labeled detections back. ``anthropic`` is
  imported lazily; a fake client can be injected for tests.

``remember_detections`` writes any position-bearing detections into a
:class:`~ohho.memory.SpatialMemory`.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

from .memory import SpatialMemory

log = logging.getLogger("ohho.perception")


@dataclass
class Detection:
    label: str
    confidence: float = 1.0
    x: Optional[float] = None  # world frame, when known
    y: Optional[float] = None
    bearing: Optional[float] = None  # rad, relative to robot heading
    range: Optional[float] = None  # m
    image_x: Optional[float] = None  # normalized [0..1] horizontal position in image

    def __str__(self) -> str:
        where = ""
        if self.x is not None and self.y is not None:
            where = f" at ({self.x:.2f}, {self.y:.2f})"
        elif self.bearing is not None:
            where = f" bearing {self.bearing:+.2f} rad"
        elif self.image_x is not None:
            where = f" img_x {self.image_x:.2f}"
        return f"{self.label}{where} (conf {self.confidence:.2f})"


class Perceptor(Protocol):
    def look(self, *args, **kwargs) -> List[Detection]: ...


def _wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class SimPerceptor:
    """Detects the sim world's labeled objects within FOV + range."""

    def __init__(
        self, robot, *, fov: float = math.radians(120), max_range: float = 3.5
    ) -> None:
        self.robot = robot
        self.fov = fov
        self.max_range = max_range

    def look(self) -> List[Detection]:
        world_objects = getattr(self.robot.transport, "world_objects", None)
        if not callable(world_objects):
            return []
        t = self.robot.telemetry()
        if not (t and t.odom):
            return []
        px, py, th = t.odom.x, t.odom.y, t.odom.theta
        out: list[Detection] = []
        for ob in world_objects():
            dx, dy = ob.x - px, ob.y - py
            rng = math.hypot(dx, dy)
            if rng > self.max_range:
                continue
            bearing = _wrap(math.atan2(dy, dx) - th)
            if abs(bearing) > self.fov / 2.0:
                continue
            confidence = max(0.3, 1.0 - 0.5 * rng / self.max_range)
            out.append(
                Detection(
                    label=ob.label,
                    confidence=round(confidence, 3),
                    x=ob.x,
                    y=ob.y,
                    bearing=round(bearing, 4),
                    range=round(rng, 3),
                )
            )
        out.sort(key=lambda d: d.range if d.range is not None else 1e9)
        return out


_VLM_PROMPT = (
    "You are a robot's object detector. List the distinct physical objects "
    "visible in this image as a JSON array — nothing else. Each item: "
    '{"label": "<short noun>", "confidence": <0..1>, "x": <0..1 horizontal center>}. At most 12 items.'
)


def estimate_vlm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a Claude VLM inference call."""
    m = model.lower()
    if "haiku" in m:
        in_rate = 0.80 / 1_000_000
        out_rate = 4.00 / 1_000_000
    elif "opus" in m:
        in_rate = 15.00 / 1_000_000
        out_rate = 75.00 / 1_000_000
    else:  # default Sonnet
        in_rate = 3.00 / 1_000_000
        out_rate = 15.00 / 1_000_000
    return round(input_tokens * in_rate + output_tokens * out_rate, 6)


@dataclass
class VlmCallStats:
    duration_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    timestamp: float = 0.0


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]
        self.usage = type("Usage", (), {"input_tokens": 120, "output_tokens": 25})()


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_kwargs = None

    def create(self, **kwargs) -> _FakeMessage:
        self.last_kwargs = kwargs
        return _FakeMessage(self._text)


class _FakeClient:
    """Mock Anthropic client for tests and offline mock mode."""

    def __init__(self, text: str) -> None:
        self.messages = _FakeMessages(text)


class VlmPerceptor:
    """Claude-vision detector for real camera frames (optional dependency).

    ``client`` is injectable for tests; otherwise the ``anthropic`` package and
    ``ANTHROPIC_API_KEY`` are required at ``look()`` time.

    Enforces a rate limit of at most ``rate_limit_hz`` (default: 1.0 Hz).
    Logs latency and estimated token cost, and invokes ``stats_hook`` if set.
    """

    def __init__(
        self,
        client=None,
        *,
        model: str = "claude-sonnet-5",
        max_tokens: int = 512,
        rate_limit_hz: float = 1.0,
        stats_hook: Optional[Callable[[VlmCallStats], None]] = None,
    ):
        self._client = client
        self.model = model
        self.max_tokens = max_tokens
        self.rate_limit_hz = rate_limit_hz
        self.stats_hook = stats_hook
        self.last_stats: Optional[VlmCallStats] = None
        self._last_call_time = 0.0

    def _get_client(self):
        if self._client is not None:
            return self._client
        import anthropic  # lazy — the [agent] extra

        self._client = anthropic.Anthropic()
        return self._client

    def look(
        self, image_bytes: bytes, media_type: str = "image/jpeg"
    ) -> List[Detection]:
        # Enforce rate-limit (at most rate_limit_hz, e.g. 1 Hz)
        if self.rate_limit_hz > 0:
            min_interval = 1.0 / self.rate_limit_hz
            now = time.monotonic()
            elapsed = now - self._last_call_time
            if elapsed < min_interval:
                to_wait = min_interval - elapsed
                log.debug(
                    "VLM rate limit: sleeping %.3fs (max %.1f Hz)",
                    to_wait,
                    self.rate_limit_hz,
                )
                time.sleep(to_wait)

        client = self._get_client()
        t0 = time.monotonic()
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": _VLM_PROMPT},
                    ],
                }
            ],
        )
        t1 = time.monotonic()
        self._last_call_time = t1
        duration_s = t1 - t0

        usage = getattr(message, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        cost = estimate_vlm_cost(self.model, in_tok, out_tok)
        stats = VlmCallStats(
            duration_s=round(duration_s, 4),
            input_tokens=in_tok,
            output_tokens=out_tok,
            estimated_cost_usd=cost,
            timestamp=time.time(),
        )
        self.last_stats = stats
        log.info(
            "VLM call completed in %.1f ms | tokens: %d in, %d out | est. cost: $%.5f",
            duration_s * 1000.0,
            in_tok,
            out_tok,
            cost,
        )
        if self.stats_hook is not None:
            try:
                self.stats_hook(stats)
            except Exception as e:
                log.warning("VLM stats hook failed: %r", e)

        text = "".join(
            getattr(block, "text", "") for block in getattr(message, "content", [])
        )
        return parse_vlm_detections(text)


def parse_vlm_detections(text: str) -> List[Detection]:
    """Tolerantly parse the model's JSON array of {label, confidence, x}."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    out: list[Detection] = []
    for item in raw:
        if not isinstance(item, dict) or "label" not in item:
            continue
        try:
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        img_x = None
        for key in ("x", "image_x", "cx", "center_x"):
            if key in item:
                try:
                    val = float(item[key])
                    if 0.0 <= val <= 1.0:
                        img_x = val
                        break
                except (TypeError, ValueError):
                    pass
        if img_x is None and "box_2d" in item:
            try:
                box = item["box_2d"]
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    mid = float(box[1] + box[3]) / 2.0
                    if mid > 1.0:
                        mid /= 1000.0
                    img_x = max(0.0, min(1.0, mid))
            except (TypeError, ValueError):
                pass
        out.append(
            Detection(
                label=str(item["label"]),
                confidence=max(0.0, min(1.0, conf)),
                image_x=img_x,
            )
        )
    return out


def _range_from_scan(scan, bearing: float, fallback: float) -> float:
    """Distance on a planar ``Scan`` at ``bearing`` (robot frame), else ``fallback``.

    Rays stored at ``range_max`` are empty, so the camera fixed-range assumption
    stays. This is the one range source shared with lidar ``pointcloud_to_scan``.
    """
    ranges = getattr(scan, "ranges", None)
    if not ranges:
        return fallback
    inc = float(getattr(scan, "angle_increment", 0.0) or 0.0)
    if inc == 0.0:
        return fallback
    idx = int(round((bearing - float(getattr(scan, "angle_min", 0.0))) / inc))
    rng = float(ranges[idx % len(ranges)])
    range_max = float(getattr(scan, "range_max", rng))
    if rng >= range_max - 1e-3:
        return fallback
    return rng


def project_detection(
    detection: Detection,
    odom,
    *,
    fov: float = math.radians(90),
    fixed_range: float = 1.5,
    default_image_x: float = 0.5,
    scan=None,
) -> Detection:
    """Project a camera detection into world coordinates using robot odometry.

    Computes the horizontal bearing relative to robot heading from normalized
    image position ``image_x`` (0 = left, 0.5 = center, 1 = right):
        bearing = (0.5 - image_x) * fov
        world_heading = odom.theta + bearing
        world_x = odom.x + range * cos(world_heading)
        world_y = odom.y + range * sin(world_heading)

    If the detection already has a range, that is used. Otherwise a planar
    ``scan`` (from ``pointcloud_to_scan``) supplies the hit distance along
    ``bearing``, and ``fixed_range`` is the fallback when that ray is empty.
    """
    img_x = detection.image_x if detection.image_x is not None else default_image_x
    # Left of image (img_x < 0.5) is positive angle (counter-clockwise / +y)
    bearing = (0.5 - img_x) * fov
    bearing = _wrap(bearing)
    rng = detection.range if detection.range is not None else fixed_range
    if detection.range is None and scan is not None:
        rng = _range_from_scan(scan, bearing, rng)

    px = getattr(odom, "x", 0.0)
    py = getattr(odom, "y", 0.0)
    pth = getattr(odom, "theta", 0.0)

    world_heading = pth + bearing
    world_x = px + rng * math.cos(world_heading)
    world_y = py + rng * math.sin(world_heading)

    return Detection(
        label=detection.label,
        confidence=detection.confidence,
        x=round(world_x, 3),
        y=round(world_y, 3),
        bearing=round(bearing, 4),
        range=round(rng, 3),
        image_x=detection.image_x,
    )


def project_detections(
    detections: List[Detection],
    odom,
    *,
    fov: float = math.radians(90),
    fixed_range: float = 1.5,
    default_image_x: float = 0.5,
    scan=None,
) -> List[Detection]:
    """Project a list of camera detections into world coordinates."""
    return [
        project_detection(
            d,
            odom,
            fov=fov,
            fixed_range=fixed_range,
            default_image_x=default_image_x,
            scan=scan,
        )
        for d in detections
    ]


class CameraPerceptor:
    """Perceives objects from robot camera frames via VLM and projects to world frame.

    Pulls frames from ``robot.get_camera_frame()``, feeds them to :class:`VlmPerceptor`,
    computes bearing from normalized image x, and projects detections into world
    coordinates. Range comes from ``Telemetry.scan`` when the transport published
    one (lidar ``pointcloud_to_scan``), otherwise from ``fixed_range``.
    """

    def __init__(
        self,
        robot,
        *,
        vlm: Optional[VlmPerceptor] = None,
        fov: float = math.radians(90),
        fixed_range: float = 1.5,
    ) -> None:
        self.robot = robot
        self.vlm = vlm if vlm is not None else VlmPerceptor()
        self.fov = fov
        self.fixed_range = fixed_range

    def look(self) -> List[Detection]:
        frame = self.robot.get_camera_frame()
        if not frame:
            log.warning("No camera frame available from robot %s", self.robot.spec.id)
            return []
        raw = self.vlm.look(frame)
        if not raw:
            return []
        t = self.robot.telemetry()
        if t and t.odom:
            return project_detections(
                raw,
                t.odom,
                fov=self.fov,
                fixed_range=self.fixed_range,
                scan=t.scan,
            )
        return raw


def remember_detections(
    memory: SpatialMemory, detections: List[Detection], *, t: Optional[float] = None
) -> int:
    """Write position-bearing detections into memory. Returns how many stuck."""
    n = 0
    for d in detections:
        if d.x is None or d.y is None:
            continue
        memory.observe(d.label, d.x, d.y, t=t, attrs={"confidence": d.confidence})
        n += 1
    return n


__all__ = [
    "CameraPerceptor",
    "Detection",
    "Perceptor",
    "SimPerceptor",
    "VlmCallStats",
    "VlmPerceptor",
    "estimate_vlm_cost",
    "parse_vlm_detections",
    "project_detection",
    "project_detections",
    "remember_detections",
]
