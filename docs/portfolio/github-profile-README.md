# DRAFT — GitHub profile README

**Status:** draft text for the GitHub profile repository (`<user>/<user>`). Do not paste this onto the live profile until someone has checked the wording. This file does not publish anything.

Suggested profile name line: **Varun Vaidhiya**. Suggested one-line bio: robotics software for a mecanum mobile manipulator and a quadruped, with ROS 2 optional.

Handles already printed on the OmniBot README:

- https://www.youtube.com/@varun.vaidhiya/videos
- https://x.com/varunvaidhiya
- https://ohho-robotics.com
- https://github.com/ohho-robotics

## Pin these four

GitHub shows six pins. These four are the ones this draft asks for. Leave the other two empty until there is a fifth repository with a README that is just as careful.

| Pin order | Repository | One line that matches its README |
|---|---|---|
| 1 | https://github.com/ohho-robotics/OmniBot | Mecanum mobile manipulator. ROS 2 Jazzy workspace, Yahboom base protocol, SO-101 arm, Gazebo Harmonic, Quest project, Android app. Apache-2.0. |
| 2 | https://github.com/ohho-robotics/ohho-os | Python engine `ohho-os` (import `ohho`). Tree **1.1.3**. One `Robot` API over an in-process runtime or ROS 2, and over sim, serial, Feetech, Unitree DDS, or `ros2://`. Base install has no third-party dependencies. |
| 3 | https://github.com/ohho-robotics/OhhO-VR | Meta Quest 3 / 3S client only. Unity **6000.5.2f1**. ROSBridge teleop, MJPEG camera, hand-tracking IK for the SO-101. WebRTC and visionOS are roadmap, not this build. |
| 4 | https://github.com/ohho-robotics/ohho.apk | Kotlin controller, package `com.varunvaidhiya.robotcontrol`. Min SDK 24, target SDK 35. ROSBridge plus MJPEG. CI builds a debug APK and runs unit tests. |

`README.md` in OhhO OS still lists Source as https://github.com/ohho-robotics/ohho-sdk. The git remote of the checkout this draft was written from is `ohho-os`. Pin **ohho-os** unless those two URLs are later confirmed to be the same repository.

## Short README body (paste after review)

```markdown
# Varun Vaidhiya

I build the open-source stack behind [OhhO](https://ohho-robotics.com): a mecanum mobile manipulator, the Python engine that drives it with or without ROS 2, and the Quest and Android clients.

The engine's base install is dependency-free. Non-simulated robots start disarmed behind a software safety gate. A lot of the automated checks run in simulation or on a fake ROS 2 node. Hardware-in-the-loop on the real robots is still open.

- [OmniBot](https://github.com/ohho-robotics/OmniBot) — ROS 2 Jazzy reference platform (Apache-2.0)
- [ohho-os](https://github.com/ohho-robotics/ohho-os) — robot-agnostic engine, PyPI package `ohho-os` (1.1.2 is the published release; the tree is 1.1.3)
- [OhhO-VR](https://github.com/ohho-robotics/OhhO-VR) — Quest teleop client
- [ohho.apk](https://github.com/ohho-robotics/ohho.apk) — Android teleop client

[YouTube](https://www.youtube.com/@varun.vaidhiya/videos) · [X](https://x.com/varunvaidhiya)
```

## Do not write on the profile

- Star counts, download counts, or "used in production".
- A latency, accuracy, or success-rate number. The sim loop computes p50 and p95 and does not publish a figure; its nightly workflow's first run is still described as pending.
- "CI green" as a permanent badge claim inside the prose. The OhhO OS README says the CI workflow is green on `main` at the time that README was written. Prefer the workflow badge that GitHub renders, which can go red.
- That Android Home metrics are live diagnostics. That repository's README says they are demo data.
- That the 39 Quest edit-mode tests run in GitHub Actions today. They are skipped until Unity licence secrets are set.
- Employment, degree, or location. Those are not in these repositories.

## Longer drafts in this folder

- `cv-robotics-software.md`
- `cv-embedded-integration.md`
- `cv-ml-robotics.md`
- `project-brief-omnibot.md`
