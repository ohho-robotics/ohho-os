# HIL pre-flight checklist (OHH-35)

Do this on the bench before any hardware HIL command. `ohho hil` with no
flags only prints this reminder. It does not open a port.

The harness does not measure batteries, wiring, the e-stop, lidar, cameras,
or the Pi image. `--preflight-ok` means you did this list.

## One command

Dry run:

```bash
ohho hil
# or: python scripts/hil_run.py
```

Fake transports (CI, no devices):

```bash
ohho hil --mock
```

Hardware, from an interactive terminal, after every box below is checked.
Replace the device names with the nodes you confirmed (`ls /dev/ttyUSB*`,
`ls /dev/ttyACM*`, `ip link`). Example values are typical on the Pi, not
defaults:

```bash
OHHO_HIL=1 \
OHHO_OMNIBOT_PORT=/dev/ttyUSB0 \
OHHO_OMNIBOT_ARM=/dev/ttyACM0 \
OHHO_GO2_IFACE=eth0 \
ohho hil --hardware --preflight-ok
```

That runs `python -m unittest discover -s tests/hil -p test_hil.py -v` and
writes `docs/hil-results.md`. Omit a variable to skip that device. Set
`OHHO_HIL_OPERATOR` and `OHHO_HIL_BAG` if you want them copied into the
results file. `OHHO_HIL=1` on its own does not open a port.

Exit codes: `0` passed, `1` a test failed, `2` the harness refused to start,
`3` every test was skipped.

## These tests are outside SafetyGate

`tests/hil` builds the Yahboom, Feetech, and Unitree adapters directly. The
300 ms command watchdog and the speed clamp are not on this path. The
physical e-stop has to be in your hand.

What the four tests command:

- OmniBot base: 0.05 m/s forward for about 1 second, then zero
- OmniBot arm: gripper to 0.2 rad, then back to 0
- Composite: both of those
- Go2: read telemetry, then zero velocity

If the process hangs, hit the physical e-stop. The harness kills a run that
is still going after 180 seconds, and a hard kill may skip the adapter's
`disconnect`.

## Checklist

### Batteries

- OmniBot base pack is seated, charged, and in its normal voltage range.
- The arm supply is on. STS3215 servos need that supply, not only USB.
- Go2 battery is installed and charged enough that the robot will not sit
  down during the read.
- Chargers are unplugged before motion.

### Wiring

- The base USB serial node is the string you will pass as
  `OHHO_OMNIBOT_PORT`.
- The Feetech arm bus is the string you will pass as `OHHO_OMNIBOT_ARM`.
- Base and arm cables are not swapped.
- Go2 Ethernet is the interface in `OHHO_GO2_IFACE`, link is up, and it is
  this robot (domain 0 reaches every Go2 on that network).
- E-stop wiring is intact.

### E-stop

- You can reach a physical e-stop, or the Go2 remote, without stepping into
  the robot's path.
- Press it once before the run and confirm motion stops. Reset it only when
  the area is clear.
- The area in front of the OmniBot is clear for a short 0.05 m/s pulse. The
  base is on the ground or on a stand it cannot roll off.

### Lidar

- Lens and cable are seated, cap is off.
- After the robot stack is up on the Pi, `ros2 topic list` shows a scan
  topic and `ros2 topic hz` on it is non-zero (often `/scan`).

### Cameras

- Caps are off and the image is not black.
- `ros2 topic list` shows a camera image topic and `ros2 topic hz` on it is
  non-zero.
- Camera cables are seated.

### Pi 5 Jazzy image

- The Raspberry Pi 5 boots the image you intend to record from.
- `source /opt/ros/jazzy/setup.bash` then `echo $ROS_DISTRO` prints `jazzy`.
- `ros2` runs.
- `ros-jazzy-rosbag2-storage-mcap` is installed if the bag will be recorded
  on the Pi (see `docs/hil-bag.md`).
- The clock is close to UTC so the bag stamp and `docs/hil-results.md` match.

## After the list

Record the 2-minute MCAP (`docs/hil-bag.md`), then run the hardware command
above. Keep the e-stop in hand while it runs.
