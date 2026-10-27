"""Planar scan synthesis from 3-D lidar point clouds (stdlib + optional numpy).

Go2 and MuJoCo sim publish ``sensor_msgs/PointCloud2`` on DDS; native
``ohho nav`` consumes :class:`~ohho.schema.Scan` on ``Telemetry.scan``.
This module converts body-frame points into a 360-ray scan for mapping.
"""

from __future__ import annotations

import math
import os
import struct
from typing import Iterable, Iterator, Optional, Sequence, Union

from .schema import Scan

# Height band for obstacle slicing in the robot body frame (metres).
DEFAULT_Z_MIN = 0.05
DEFAULT_Z_MAX = 0.5
DEFAULT_NUM_RAYS = 360
DEFAULT_RANGE_MAX = 30.0

# DDS topic used by Go2 onboard lidar and ohho-quadruped MuJoCo sim (domain 1).
DEFAULT_LIDAR_DDS_TOPIC = "rt/utlidar/cloud"

PointLike = Union[Sequence[float], tuple[float, float, float]]


def lidar_dds_topic() -> str:
    """DDS topic for Go2 / sim point clouds (override with ``OHHO_LIDAR_DDS_TOPIC``)."""
    return os.environ.get("OHHO_LIDAR_DDS_TOPIC", DEFAULT_LIDAR_DDS_TOPIC)


def _iter_points(points: Iterable[PointLike]) -> Iterator[tuple[float, float, float]]:
    for p in points:
        if len(p) < 3:
            continue
        yield float(p[0]), float(p[1]), float(p[2])


def pointcloud_to_scan(
    points: Iterable[PointLike],
    *,
    z_min: float = DEFAULT_Z_MIN,
    z_max: float = DEFAULT_Z_MAX,
    num_rays: int = DEFAULT_NUM_RAYS,
    range_max: float = DEFAULT_RANGE_MAX,
    angle_min: float = -math.pi,
    use_numpy: bool = True,
) -> Scan:
    """Bin body-frame ``(x, y, z)`` points into a full-turn planar scan.

    * ``x`` forward, ``y`` left, ``z`` up (OhhO body / odom frame).
    * Keeps points with ``z_min <= z <= z_max``.
    * Each ray stores the **minimum** horizontal range in its bin; empty bins
      are set to ``range_max``.
    """
    if num_rays < 1:
        raise ValueError("num_rays must be >= 1")

    inc = 2.0 * math.pi / num_rays
    ranges = [range_max] * num_rays

    if use_numpy:
        try:
            import numpy as np

            arr = np.asarray(list(_iter_points(points)), dtype=np.float64)
            if arr.size == 0:
                return Scan(
                    angle_min=angle_min,
                    angle_increment=inc,
                    ranges=ranges,
                    range_max=range_max,
                )
            mask = (arr[:, 2] >= z_min) & (arr[:, 2] <= z_max)
            arr = arr[mask]
            if arr.size == 0:
                return Scan(
                    angle_min=angle_min,
                    angle_increment=inc,
                    ranges=ranges,
                    range_max=range_max,
                )
            xy = arr[:, :2]
            r = np.hypot(xy[:, 0], xy[:, 1])
            valid = r > 1e-6
            if not np.any(valid):
                return Scan(
                    angle_min=angle_min,
                    angle_increment=inc,
                    ranges=ranges,
                    range_max=range_max,
                )
            r = r[valid]
            ang = np.arctan2(arr[valid, 1], arr[valid, 0])
            bins = ((ang - angle_min) / inc).astype(np.int64)
            bins = np.mod(bins, num_rays)
            for b, dist in zip(bins.tolist(), r.tolist(), strict=False):
                if dist < ranges[b]:
                    ranges[b] = float(dist)
            return Scan(
                angle_min=angle_min,
                angle_increment=inc,
                ranges=ranges,
                range_max=range_max,
            )
        except ImportError:
            pass

    for x, y, z in _iter_points(points):
        if z < z_min or z > z_max:
            continue
        r = math.hypot(x, y)
        if r <= 1e-6:
            continue
        ang = math.atan2(y, x)
        k = int(math.floor((ang - angle_min) / inc)) % num_rays
        if r < ranges[k]:
            ranges[k] = r

    return Scan(
        angle_min=angle_min, angle_increment=inc, ranges=ranges, range_max=range_max
    )


def _field_offset(fields: object, name: str) -> Optional[int]:
    for f in fields:
        fname = getattr(f, "name", None) or (
            f.get("name") if isinstance(f, dict) else None
        )
        if fname != name:
            continue
        off = getattr(f, "offset", None)
        if off is None and isinstance(f, dict):
            off = f.get("offset")
        return int(off) if off is not None else None
    return None


def pointcloud2_to_points(msg: object) -> list[tuple[float, float, float]]:
    """Decode a ``sensor_msgs/PointCloud2``-like message to ``(x, y, z)`` tuples.

    Accepts SDK DDS objects or plain dicts (for tests). Only ``x``, ``y``, ``z``
    float32 fields are required.
    """
    data = getattr(msg, "data", None)
    if data is None and isinstance(msg, dict):
        data = msg.get("data")
    if data is None:
        return []
    raw = bytes(data)
    fields = getattr(msg, "fields", None)
    if fields is None and isinstance(msg, dict):
        fields = msg.get("fields", [])
    point_step = int(
        getattr(msg, "point_step", 0)
        or (msg.get("point_step") if isinstance(msg, dict) else 0)
    )
    if point_step <= 0:
        point_step = 12
    ox = _field_offset(fields, "x")
    oy = _field_offset(fields, "y")
    oz = _field_offset(fields, "z")
    if ox is None:
        ox, oy, oz = 0, 4, 8

    width = int(
        getattr(msg, "width", 0) or (msg.get("width") if isinstance(msg, dict) else 0)
    )
    height = int(
        getattr(msg, "height", 0) or (msg.get("height") if isinstance(msg, dict) else 0)
    )
    n = width * height if width and height else len(raw) // point_step

    out: list[tuple[float, float, float]] = []
    for i in range(n):
        base = i * point_step
        if base + oz + 4 > len(raw):
            break
        x = struct.unpack_from("<f", raw, base + ox)[0]
        y = struct.unpack_from("<f", raw, base + oy)[0]
        z = struct.unpack_from("<f", raw, base + oz)[0]
        if not all(math.isfinite(v) for v in (x, y, z)):
            continue
        out.append((x, y, z))
    return out


def pointcloud2_to_scan(
    msg: object,
    *,
    z_min: float = DEFAULT_Z_MIN,
    z_max: float = DEFAULT_Z_MAX,
    num_rays: int = DEFAULT_NUM_RAYS,
    range_max: float = DEFAULT_RANGE_MAX,
) -> Scan:
    """Decode PointCloud2 and synthesize a :class:`Scan`."""
    return pointcloud_to_scan(
        pointcloud2_to_points(msg),
        z_min=z_min,
        z_max=z_max,
        num_rays=num_rays,
        range_max=range_max,
    )
