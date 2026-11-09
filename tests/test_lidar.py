import math
import struct
import unittest

from ohho.lidar import (
    DEFAULT_NUM_RAYS,
    pointcloud2_to_points,
    pointcloud2_to_scan,
    pointcloud_to_scan,
)
from ohho.nav import Mapper, OccupancyGrid
from ohho.schema import Odometry


def _wall_points(
    x: float, ys: list[float], z: float = 0.2
) -> list[tuple[float, float, float]]:
    return [(x, y, z) for y in ys]


class TestPointcloudToScan(unittest.TestCase):
    def test_empty_cloud_is_all_max_range(self):
        scan = pointcloud_to_scan([], use_numpy=False)
        self.assertEqual(len(scan.ranges), DEFAULT_NUM_RAYS)
        self.assertTrue(all(r == scan.range_max for r in scan.ranges))

    def test_height_band_filters_points(self):
        hit = pointcloud_to_scan([(2.0, 0.0, 0.2)], use_numpy=False)
        miss = pointcloud_to_scan([(2.0, 0.0, 0.01)], use_numpy=False)
        forward = DEFAULT_NUM_RAYS // 2
        self.assertLess(hit.ranges[forward], hit.range_max)
        self.assertEqual(miss.ranges[forward], miss.range_max)

    def test_wall_ahead_bins_to_forward_ray(self):
        pts = _wall_points(2.0, [0.0, 0.1, -0.1])
        scan = pointcloud_to_scan(pts, use_numpy=False)
        forward = DEFAULT_NUM_RAYS // 2
        self.assertAlmostEqual(scan.ranges[forward], 2.0, places=3)

    def test_keeps_nearest_hit_per_bin(self):
        pts = [(1.0, 0.0, 0.2), (3.0, 0.0, 0.2)]
        scan = pointcloud_to_scan(pts, use_numpy=False)
        forward = DEFAULT_NUM_RAYS // 2
        self.assertAlmostEqual(scan.ranges[forward], 1.0, places=3)

    def test_side_wall_bins_by_angle(self):
        pts = _wall_points(0.0, [2.0], z=0.25)
        scan = pointcloud_to_scan(pts, use_numpy=False)
        # +y is left; atan2(2, 0) = pi/2
        inc = 2.0 * math.pi / DEFAULT_NUM_RAYS
        k = int(math.floor((math.pi / 2 - (-math.pi)) / inc)) % DEFAULT_NUM_RAYS
        self.assertAlmostEqual(scan.ranges[k], 2.0, places=3)

    def test_numpy_path_matches_python(self):
        pts = _wall_points(1.5, [0.0, 0.2, -0.2])
        py_scan = pointcloud_to_scan(pts, use_numpy=False)
        try:
            import numpy as np  # noqa: F401

            np_scan = pointcloud_to_scan(pts, use_numpy=True)
        except ImportError:
            self.skipTest("numpy not installed")
        self.assertEqual(py_scan.ranges, np_scan.ranges)


class TestPointCloud2Decode(unittest.TestCase):
    def _pack_xyz(self, points: list[tuple[float, float, float]]) -> dict:
        raw = bytearray()
        for x, y, z in points:
            raw.extend(struct.pack("<fff", x, y, z))
        return {
            "width": len(points),
            "height": 1,
            "point_step": 12,
            "fields": [
                {"name": "x", "offset": 0},
                {"name": "y", "offset": 4},
                {"name": "z", "offset": 8},
            ],
            "data": bytes(raw),
        }

    def test_decode_and_scan(self):
        msg = self._pack_xyz([(2.0, 0.0, 0.2), (2.0, 0.5, 0.2)])
        decoded = pointcloud2_to_points(msg)
        self.assertEqual(len(decoded), 2)
        scan = pointcloud2_to_scan(msg)
        forward = DEFAULT_NUM_RAYS // 2
        self.assertAlmostEqual(scan.ranges[forward], 2.0, places=3)


class TestLidarNavIntegration(unittest.TestCase):
    def test_mapper_accepts_synthetic_scan(self):
        scan = pointcloud_to_scan(_wall_points(2.0, [0.0]), use_numpy=False)
        g = OccupancyGrid(size_m=8.0, resolution=0.08)
        Mapper(g).update(Odometry(), scan)
        self.assertGreater(g.counts()["occupied"], 0)


if __name__ == "__main__":
    unittest.main()
