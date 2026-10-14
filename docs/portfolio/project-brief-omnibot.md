# DRAFT — OmniBot project brief (about two pages)

**Status:** draft for a friend to review. Not a paper, not a datasheet, and not a claim that the physical robot has passed bring-up.

**Project:** OmniBot, OhhO's mecanum-wheel mobile manipulator, and the engine that drives it.
**Author:** Varun Vaidhiya
**Licence:** Apache-2.0 on OhhO code in these repositories. Vendored Meta XR packages keep the Oculus SDK licence.
**Date of sources:** tree versions and READMEs as checked out with OhhO OS at **1.1.3**. PyPI currently documented as **ohho-os 1.1.2** (tag `v1.1.2`).

## What it is

OmniBot is a holonomic base plus a 6-DOF SO-101 arm (Feetech STS3215 bus) and onboard sensing (RGB, depth, and scan in the robot spec; lidar, front RGB camera, and IMU in the Gazebo launch). OhhO OS is the Python engine in front of that hardware and in front of a Unitree Go2: one `Robot` API, a runtime that is either in-process threads or ROS 2, and a transport per link. The website product is separate. This brief covers the four public repositories a profile should pin.

## Architecture

```
operator clients
  Quest (OhhO-VR, Unity)  ·  Android (ohho.apk, Kotlin)
  both speak ROSBridge WebSocket, default port 9090
  camera: MJPEG from web_video_server, port 8080
        │
        ▼
OmniBot ROS 2 workspace (Jazzy)          OhhO OS  (Python, ROS optional)
  bringup, arm, Nav2, Gazebo Harmonic      Robot  →  Runtime (native | ros2)
  Yahboom codec, mecanum kinematics        Transport: sim | serial | feetech | dds | ros2
  LeRobot record, VLA / RL engines         SafetyGate on every non-sim transport
        │                                  data → tiny ACT → FastAPI serve
        ▼
hardware: Yahboom base (serial), SO-101 arm (Feetech), or Go2 (DDS)
simulation: in-process sim://   or   Gazebo Harmonic via ros_gz_bridge
```

Joint names shared by the registry and the arm driver: `arm_shoulder_pan`, `arm_shoulder_lift`, `arm_elbow_flex`, `arm_wrist_flex`, `arm_wrist_roll`, `arm_gripper`. OmniBot `max_lin` in the spec is 0.2 m/s. The Feetech conversion in code is 4096 ticks per revolution, home tick 2048.

`serial://<base_port>,<arm_port>` builds one transport from the Yahboom base and the Feetech arm. `ros2://` publishes `geometry_msgs/Twist` on `/cmd_vel` and subscribes to `/odom` and arm joint state. Gazebo is launched from the OmniBot repo, not from `ohho sim-serve`. That server can sit on the in-process sim, or on rosbridge; its deadman window is 300 ms.

## Sim versus real

| Path | Robot in the loop? | What the sources actually show |
|---|---|---|
| `pip install ohho-os` then `ohho doctor` and `ohho sim --robot omnibot --seconds 2` | No | README and the OmniBot docs-test job. No serial port, camera, or GPU. |
| OhhO OS unit tests and CI matrix | No | README: 197 tests, 18 skipped, Python 3.12.3 Linux, base install. CI: 3 OS × Python 3.10–3.13, plus a 95% coverage gate on `ohho/safety.py`. `AGENTS.md` separately notes 383 tests, 7 skipped, on Windows Python 3.12. |
| `examples/demo_drive.py --transport sim://` and `ros2://` | No | Fake ROS 2 integrator, not Gazebo. Both calls in the evidence note use `--seconds 2 --vx 0.2 --min-dx 0.15`. Gazebo GIF for that issue was not produced. |
| Gazebo `sim.launch.py world:=flat` | Simulated only | `sim-smoke` on `ros:jazzy`. Headless. Topics `/odom`, `/scan`, `/imu`, `/camera/front/image_raw` must publish within 30 s. `/cmd_vel` must move the model at least 0.2 m. `world:=apartment` is the furnished room (kitchen, desk, door) and is not what that smoke job asserts. |
| `ohho sim-loop` | Simulated only | At least 5 episodes, 200 CPU steps of tiny ACT, at least 50 closed-loop steps, latency percentiles computed. Nightly workflow's first run is **pending**. No latency number is printed in the README. |
| Android screenshots in `ohho.apk` | No | API 35 emulator. README: Home cards are demo data. Camera badge stays NO FEED until a JPEG arrives. |
| Quest app | Needs the robot PC for teleop | Unity client only. 39 edit-mode tests exist; CI skips them until Unity licence secrets are set. No on-headset pass is recorded in the README. |
| `ohho selftest` on hardware, `tests/hil/` | Real robot | Sim mode is in CI. Hardware mode is implemented and **not yet run** on the bench (`docs/selftest.md`: pending). HIL skips unless `OHHO_HIL=1`. |
| Xbox teleop, physical cameras, physical arm, LeRobot recording on the robot, Nav2/SLAM | Real robot | Launch files exist. The OmniBot README lists them under "Needs hardware". This brief does not treat those launches as completed field trials. |
| OpenVLA / SmolVLA, Isaac Lab | GPU workstation | Requirement files. No reported training score in the READMEs used here. |

The OmniBot README embeds photos and GIFs from `assets/` (including `PXL_20260505_121303728.jpg` and `Omnibot_demo1.gif`). Those files are not a substitute for the missing bench self-test or HIL log.

## Results that are specified

Design limits and CI assertions only. Nothing in this table is a measured tracking error, battery life, or policy success rate.

| Item | Figure | Kind |
|---|---|---|
| OmniBot SafetyGate caps | 0.2 m/s, 0.2 m/s, 1.0 rad/s | Design profile (`docs/safety.md`) |
| Go2 SafetyGate caps | 0.5 m/s forward, 0.3 m/s lateral, 1.0 rad/s | Design profile. Spec `max_lin` is 1.5 m/s; the gate is tighter. |
| Hard ceiling | 1.5 m/s, 2.0 rad/s | Design. A config file can raise it. A caller cannot. |
| Watchdogs | 300 ms command, 500 ms state | Design |
| Arming phrase | Required on hardware, refused in CI | Design |
| SafetyGate CI | ≥ 95% line and branch on `ohho/safety.py` | CI gate |
| Base-install tests | 197 run, 18 skipped, exit OK | README, one Linux environment |
| Demo-drive displacement check | ≥ 0.15 m in 2 s at vx 0.2 | CI / evidence vs fake robot and `sim://` |
| Gazebo smoke displacement | ≥ 0.2 m, sensors within 30 s | CI vs Gazebo Harmonic |
| Wheel-spin self-test pulse | 0.25 s, 5% PWM, off unless `--allow-spin` | Design. Not a recorded hardware pass. |
| Go2 state poll | 20 Hz | Adapter default |
| Quest IK | 20 Hz, ≤ 50 iterations, 1 mm tolerance; reach 0.33 m (0.117 + 0.133 + 0.080) | Design constants in the VR README |
| Quest recording | 30 Hz JSONL on the headset | Client behaviour. Upload target is a bridge that is not in that repo. |
| Android CI | `assembleDebug` and `test` | CI. Emulator screenshots, not a robot. |
| PyPI | `ohho-os==1.1.2` installable from the documented release | Release note. 1.1.3 is the tree version. |

## Links

| | |
|---|---|
| OmniBot | https://github.com/ohho-robotics/OmniBot |
| OhhO OS (git remote of this checkout) | https://github.com/ohho-robotics/ohho-os |
| OhhO OS Source URL still in README / `pyproject.toml` | https://github.com/ohho-robotics/ohho-sdk |
| OhhO VR | https://github.com/ohho-robotics/OhhO-VR |
| Android controller | https://github.com/ohho-robotics/ohho.apk |
| Package | https://pypi.org/project/ohho-os/ |
| Site | https://ohho-robotics.com |
| YouTube | https://www.youtube.com/@varun.vaidhiya/videos |
| Safety write-up | `docs/safety.md` in OhhO OS |
| Self-test write-up | `docs/selftest.md` |
| Demo-drive evidence | `docs/evidence/ohh-32-demo-drive.md` |

## Still open (human, hardware)

1. Run `ohho selftest --robot omnibot --allow-spin` on the bench and attach the JSON report.
2. Run `OHHO_HIL=1` against the OmniBot base, arm, composite, and Go2. Ports are environment variables, not guesses in this brief.
3. Let `.github/workflows/sim-loop.yml` complete once, then paste the real p50 and p95 into a revision of this page.
4. Set the Unity licence secrets if the 39 Quest edit-mode tests should run in CI.
5. Decide whether the public engine URL is `ohho-os` or `ohho-sdk` and make the README match the remote.
