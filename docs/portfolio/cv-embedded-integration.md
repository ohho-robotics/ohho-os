# DRAFT — CV, embedded and integration

**Status:** draft for review. Do not send, export to PDF, or publish until the contact block and any employment or education lines are filled in by hand.

**Varun Vaidhiya**

Embedded integration for a mecanum mobile manipulator: serial base, servo bus, safety interlock, and the headset and phone clients that speak ROSBridge to it.

| | |
|---|---|
| GitHub org | https://github.com/ohho-robotics |
| Site | https://ohho-robotics.com |
| YouTube | https://www.youtube.com/@varun.vaidhiya/videos |
| X | https://x.com/varunvaidhiya |
| Email / phone / location | _not in the repositories — fill in_ |
| Employment / education | _not in the repositories — fill in_ |

Public handles above are the ones already printed on the OmniBot README. The git author name on this checkout is Varun Vaidhiya.

## Summary

Integrates the OmniBot stack across a Yahboom base board, an SO-101 arm on a Feetech bus, a Unitree Go2 DDS link, and two operator clients (Meta Quest and Android). Motion on a real link is refused until a human arms a fail-closed software gate. Simulated and mock-byte tests exist; a recorded pass on the physical bench does not.

## OmniBot links (https://github.com/ohho-robotics/OmniBot)

OmniBot is OhhO's mecanum-wheel mobile manipulator. The repository holds the ROS 2 packages, wheel kinematics, the Yahboom board protocol, a Quest project, and an Android app. Licence for OhhO code: Apache-2.0. ROS 2 distro in CI and the README: **Jazzy**. Photos and GIFs are the files in `assets/`.

- **Base.** Yahboom serial protocol, vendored codec, ROS 2 driver path under `omnibot-ros2/`. In OhhO OS the same board is `serial://` (`ohho/adapters/yahboom.py`): reader thread, `0xFB` packets, odometry integration, IMU yaw fusion. `AdapterUnavailable` if the `[serial]` extra (pyserial) is missing.
- **Arm.** SO-101, six joints: `arm_shoulder_pan`, `arm_shoulder_lift`, `arm_elbow_flex`, `arm_wrist_flex`, `arm_wrist_roll`, `arm_gripper`. Feetech bus in OhhO OS (`feetech://`): 4096 ticks/rev, home tick 2048, radians ↔ ticks (`ohho/adapters/feetech.py`). Self-test diagnostics for STS3215 servos cover ID, voltage, temperature, and position (`docs/selftest.md`).
- **One robot, two links.** `serial://<base>,<arm>` returns a `CompositeTransport`: velocity to the base, joint commands to the arm, merged telemetry, e-stop to both.
- **Go2 (second robot, same API).** `dds://<iface>` polls `SportClient` state at 20 Hz (`ohho/adapters/unitree.py`, default `hz=20.0`) and maps velocity to `Move`. Extra: `[unitree]` (cyclonedds). SDK import is lazy.
- **CI on the robot repo** (`.github/workflows/ci.yml`): `cpu` (pytest/unittest for kinematics, Yahboom codec, BEV homography, arm tick math, learning engine, agent engine, data engine, plus ruff), `docs` (README fences in a fresh venv), `ros` (colcon build and test in `ros:jazzy-ros-base-noble`), `sim-smoke` (headless Gazebo Harmonic, software rendering).

## Safety interlock (software, on every non-sim transport)

`Robot.connect` wraps every transport whose protocol is not `simulated` in `SafetyGate` (`ohho/safety.py`, `docs/safety.md`). There is no bypass flag.

| Rule | Value in the profile / docs |
|---|---|
| OmniBot velocity caps | 0.2 m/s forward, 0.2 m/s lateral, 1.0 rad/s yaw; acceleration 0.5 / 0.5 / 2.0 |
| Go2 velocity caps | 0.5 m/s forward, 0.3 m/s lateral, 1.0 rad/s yaw |
| Hard ceiling | 1.5 m/s linear, 2.0 rad/s angular, unless a safety config file raises `hard_ceiling` |
| Command watchdog | 300 ms without a command while moving → zero velocity |
| State watchdog | 500 ms without state while armed → stop and latch |
| Tilt / fault | \|roll\| or \|pitch\| > 0.6 rad, or non-zero `error_code` → stop and latch |
| Arming | Never automatic. Non-loopback hardware needs `OHHO_ARM_HARDWARE=1`, an interactive TTY, and the typed phrase `I am physically present, the area is clear, the remote is in my hand`. CI and a non-TTY refuse. |
| Audit | `~/.ohho/safety/<UTC date>.jsonl` |
| CI | `ohho/safety.py` line and branch coverage at least 95% |

These are design limits and CI gates, not measured tracking error on a physical robot.

## Bench self-test (`ohho selftest`)

Five checks: serial link, four-wheel spin with encoder direction, IMU rate, STS3215 servo diagnostics, camera frames. Dated JSON report. Exit codes 0–3 are specified in `docs/selftest.md`.

- `ohho selftest --sim` runs all five checks in-process and is a CI smoke step. It does not open a serial port.
- On hardware, wheel spin is skipped unless `--allow-spin` or a TTY confirmation. The documented pulse is 0.25 s at 5% PWM.
- The IMU check's pass floor is `min_rate_hz` 10 Hz (`ohho/selftest.py`). The in-sim sample reports a nominal 20.0 Hz. `AGENTS.md` summarizes the check as "20+ Hz". A physical run is not in the repo.
- `docs/selftest.md` and `AGENTS.md`: real-robot bench run is **pending**.

HIL tests (`tests/hil/test_hil.py`) cover OmniBot base, OmniBot arm, OmniBot composite, and Go2 DDS. They skip unless `OHHO_HIL=1`. Default ports: `OHHO_OMNIBOT_PORT=/dev/ttyUSB0`, `OHHO_OMNIBOT_ARM=/dev/ttyACM0`, `OHHO_GO2_IFACE=eth0`.

## Operator clients

**Android** — https://github.com/ohho-robotics/ohho.apk

Kotlin app, package `com.varunvaidhiya.robotcontrol`. Min SDK 24, compile and target SDK 35. Kotlin 2.1.0, Android Gradle Plugin 8.13.2, JDK 17 in CI. CI runs `./gradlew assembleDebug test` and uploads `app-debug.apk`. The app opens cleartext `ws://<ip>:<port>` (default `192.168.1.100:9090`) and, after the socket connects, MJPEG from `http://<robot-ip>:8080/stream?topic=/camera/front/image_raw`. Joystick and arm sliders publish `/cmd_vel/teleop`, `/control_mode`, and `/arm/joint_commands` only while connected. README is explicit that Home battery, Pi, workstation, and model cards are **demo data on a timer**, and that the checked-in screenshots are an API 35 emulator with no robot. TLS, a checked-in `robot.glb`, and the robot-side nodes are listed as not in this repository.

**Quest** — https://github.com/ohho-robotics/OhhO-VR

Unity **6000.5.2f1**, Meta Quest 3 / 3S, OpenXR 1.17.1, Meta XR SDK Core and Interaction 203.0.0. Single world-space screen. Left stick drives, right stick yaws, both grips are the e-stop, turbo is 2× and clamped to 0.2 m/s. Hand tracking drives a CCD IK solver for the SO-101: 20 Hz, at most 50 iterations, 1 mm tolerance; documented link lengths L1 0.117 m, L2 0.133 m, L3 0.080 m (reach 0.33 m). Camera in this repo is MJPEG. WebRTC is marked roadmap. 39 edit-mode tests live under `Assets/Scripts/Editor/Tests/`. CI (`.github/workflows/unity-tests.yml`) skips the Unity steps until `UNITY_LICENSE`, `UNITY_EMAIL`, and `UNITY_PASSWORD` are set. Apple Vision Pro is described as not implemented.

## Do not claim

- A measured bring-up on the physical OmniBot or Go2 (self-test and HIL both still gated on a bench run).
- That Gazebo, the emulator, or the fake ROS 2 node is the real robot. See `docs/portfolio/project-brief-omnibot.md`.
- That Home-screen metrics in the Android app are diagnostics.
- That Quest CI has executed the 39 edit-mode tests. The README says the workflow skips them until the licence secrets exist.
