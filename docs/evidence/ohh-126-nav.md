# OHH-126 — Go2 lidar nav evidence

## Automated (this repo)

- ``tests/test_lidar.py`` — ``pointcloud_to_scan`` / PointCloud2 decode on synthetic clouds.
- ``tests/test_unitree.py::TestUnitreeLidarScan`` — DDS transport + ``SafetyGate`` → ``Navigator.update_map``.

## MuJoCo / office eval — TODO (human)

Not run on this workstation (Windows; MuJoCo sim not executed here).

- [ ] Start **ohho-quadruped** MuJoCo sim on ``dds://lo?domain=1`` with body-frame ``rt/utlidar/cloud``.
- [ ] ``ohho nav map unitree-go2 --transport "dds://lo?domain=1"`` — map non-empty after spin-scan.
- [ ] Office scene: **5/5** ``goto`` successes — record log path below when done.

```
# paste command output / log path here after eval
```
