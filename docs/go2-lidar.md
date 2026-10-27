# Go2 lidar → planar scan (OHH-126)

Native ``ohho nav goto|explore|map`` needs a planar :class:`~ohho.schema.Scan` on
``Telemetry.scan``. The Unitree Go2 (and the **ohho-quadruped** MuJoCo sim) expose
3-D lidar as DDS ``sensor_msgs/PointCloud2``.

## Conversion

``ohho.lidar.pointcloud_to_scan()`` slices a **body-frame** cloud:

- Height band: **0.05–0.5 m** (``z`` up).
- **360** rays, ``angle_min = -π``, increment ``2π/360``.
- Each bin keeps the nearest horizontal range; empty bins → ``range_max`` (30 m).

Points must already be in the OhhO body frame (**x** forward, **y** left, **z**
up). The onboard ``utlidar_lidar`` frame is close but not identical; the
**ohho-quadruped** sim should publish clouds already transformed to body frame
on DDS domain **1** (loopback ``lo``).

## DDS wiring (hardware or sim)

``UnitreeDdsTransport`` subscribes after ``connect()``:

| Item | Default |
|------|---------|
| Topic | ``rt/utlidar/cloud`` |
| Override | ``OHHO_LIDAR_DDS_TOPIC`` |
| Transport URI | ``dds://lo?domain=1`` (MuJoCo sim) |

```bash
ohho nav map unitree-go2 --transport "dds://lo?domain=1"
ohho nav goto unitree-go2 --transport "dds://lo?domain=1" --x 1 --y 0
```

Arm the robot at a TTY before motion (``SafetyGate``; see ``docs/safety.md``).

## Test / CI hooks (no DDS)

Inject scans without CycloneDDS:

1. **State dict** — ``state_factory`` returns ``{"pointcloud": [(x, y, z), ...], ...}``.
2. **`pointcloud_factory`** — callable returning a list of ``(x,y,z)`` or a
   PointCloud2-like object (see ``ohho.lidar.pointcloud2_to_scan``).

## ohho-quadruped interface

The sim should publish body-frame points on ``rt/utlidar/cloud`` at **10 Hz** on
**domain 1**, or document an alternate topic via ``OHHO_LIDAR_DDS_TOPIC``. Ray
data may come from MJCF rangefinders; conversion to ``Scan`` stays in **ohho-os**.
