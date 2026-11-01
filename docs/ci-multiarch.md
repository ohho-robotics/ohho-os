# Multi-arch and multi-distro CI

OHH-374. Workflow: [`.github/workflows/multiarch.yml`](../.github/workflows/multiarch.yml).

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) remains the
operating-system and Python-version matrix (Ubuntu, macOS, Windows; Python
3.10–3.13), including the safety coverage gate and CLI smoke. This workflow
adds architecture, ROS 2 distro, and a Jetson L4T image. Hardware arming stays off. ROS rows keep DDS on loopback.

## Time

The five jobs run in parallel (`fail-fast: false`). A later push to the same
ref cancels the in-progress run.

| Job | Where | What | Estimated runner time |
| --- | --- | --- | --- |
| py3.12 / x86_64 | `ubuntu-24.04` | full unit tests, then `runtime/` if present | 5 min |
| py3.12 / aarch64 | `ubuntu-24.04-arm` | same suite, native arm64 | 6 min |
| ros2 jazzy | `ubuntu-24.04` + `ros:jazzy-ros-base-noble` | `tests/test_ros2.py`, then a sourced RMW smoke | 8 min |
| ros2 humble | `ubuntu-24.04` + `ros:humble-ros-base-jammy` | same | 8 min |
| l4t r36.2.0 | `ubuntu-24.04-arm` + `nvcr.io/nvidia/l4t-base:r36.2.0` | import, `runtime/` if present, full unit tests | 12 min |
| **Total added runner time** | | | **about 40 minutes** |
| **Added wall clock** | longest job | | **about 12–15 minutes** |

`python -m unittest discover -s tests` is the same command as the unit-test
step in `ci.yml`. A local run of that command finished in about 80 seconds,
so the test body is the small part of each job. The rest of the estimate is
checkout, pip, `apt-get`, and a cold image pull (about 300 MB for each ROS
base image, about 260 MB for `l4t-base`). These are planning figures for a
cold runner, not a measured Actions bill.

## How the matrix is built

Every row is one `strategy.matrix.include` entry with its own runner and, for
ROS, its own image and RMW. Copy an entry to add an architecture, a distro,
or a DDS implementation. Rows do not multiply each other.

### Unit tests (x86_64 and aarch64)

| arch | runner | Python | install |
| --- | --- | --- | --- |
| x86_64 | `ubuntu-24.04` | 3.12 | `pip install -e '.[serve]'` |
| aarch64 | `ubuntu-24.04-arm` | 3.12 | same |

Both rows then run `python -m unittest discover -s tests`. The aarch64 runner
is native arm64.

`scripts/ci/run-ohho-runtime-tests.sh` runs next. The onboard runtime is the
`ohho-runtime` distribution (`import ohho_runtime`) in the top-level
`runtime/` directory, laid out in
[`docs/architecture/onboard-runtime.md`](architecture/onboard-runtime.md).
`ohho/runtime/` is the executor port and is already part of the unit tests
above. When `runtime/` is absent the script prints a skip line and exits 0,
so the package's own `runtime/tests` start running on a later checkout with
no workflow edit. The install uses `--no-deps` so pip keeps this checkout's
`ohho-os`.

### ROS 2 (Jazzy and Humble)

The container shape follows
[`.github/workflows/demo-drive-ros2.yml`](../.github/workflows/demo-drive-ros2.yml):
the job runs inside the official ROS image and sources
`/opt/ros/<distro>/setup.bash` for the steps that talk to ROS. Both rows
use the `ubuntu-24.04` runner. The image supplies the distro userspace
(Noble for Jazzy, Jammy for Humble).

| distro | image | runner | `RMW_IMPLEMENTATION` | apt package |
| --- | --- | --- | --- | --- |
| jazzy | `ros:jazzy-ros-base-noble` | `ubuntu-24.04` | `rmw_cyclonedds_cpp` | `ros-jazzy-rmw-cyclonedds-cpp` |
| humble | `ros:humble-ros-base-jammy` | `ubuntu-24.04` | `rmw_cyclonedds_cpp` | `ros-humble-rmw-cyclonedds-cpp` |

Each row sets `ROS_DOMAIN_ID=199`. DDS stays on loopback.
`scripts/ci/ros2-loopback.sh` sets `ROS_LOCALHOST_ONLY=1` on Humble and
`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` on Jazzy. The RMW value is a
matrix field, so pinning a different DDS implementation on one distro is a
new `include` row (`rmw` plus `rmw_apt`).

Steps, in order:

1. **Install.** `scripts/ci/ros2-install.sh` installs the row's RMW apt
   package and `ohho-os` into `/opt/ohho-ci`. The virtualenv can see the
   image's apt Python modules, which `rclpy` imports. Humble's Debian pip
   is left in place; the virtualenv's pip does the build.
2. **Fake-node tests.** `scripts/ci/ros2-test.sh` runs `tests/test_ros2.py`
   (`Ros2Runtime` and `Ros2Transport`). That module injects a fake node and
   expects `rclpy` to be absent, so this step strips `/opt/ros` from
   `PYTHONPATH` and does not source `setup.bash`. It exits if `rclpy` is
   still importable.
3. **Pinned RMW smoke.** `scripts/ci/ros2-rmw-smoke.sh` sources the distro,
   exports the row's `RMW_IMPLEMENTATION`, applies `ros2-loopback.sh`,
   requires `librmw_cyclonedds_cpp.so` under `/opt/ros/<distro>`, and
   creates one rclpy node (`ohho_ci_rmw_pin`). The node is destroyed and
   rclpy is shut down. The step is limited to 60 seconds.
4. **QoS catalogue hook (RT-08).** The step sources the distro, applies
   the same loopback settings, re-exports `RMW_IMPLEMENTATION`, and records
   the row. QoS catalogue tests are not in this checkout. Add the command
   in that step, below the source, so it runs on the sourced graph with the
   row's DDS pin. Leave `tests/test_ros2.py` as the fake-node suite.

### L4T (no GPU)

| image | runner | device |
| --- | --- | --- |
| `nvcr.io/nvidia/l4t-base:r36.2.0` | `ubuntu-24.04-arm` | none |

`r36.2.0` is the newest tag published for `nvcr.io/nvidia/l4t-base`. The
image is Ubuntu 22.04, `linux/arm64`, and public (anonymous pull). The job
checks out on the arm64 runner, then `docker run`s the image with
`--platform linux/arm64`, `--entrypoint bash`, an empty
`NVIDIA_VISIBLE_DEVICES`, and no `--gpus`.

`scripts/ci/l4t-inner.sh` (only when `OHHO_L4T_INNER=1`) checks `uname -m`
is `aarch64`, installs `python3-venv`, creates `/opt/ohho-ci`, installs
`.[serve]`, imports `ohho` with the source directory off `sys.path`, runs
the same `runtime/` step as the unit jobs, and runs
`python -m unittest discover -s tests`. Python in this image is 3.10, which
is the floor in `pyproject.toml`. If `apt-get update` fails on the image's
NVIDIA apt entries, the script comments those lines out and retries the
Ubuntu archives. The rootfs is still the L4T base image.

## Leaving the other workflows alone

Safety coverage, CLI smoke, and Python 3.10–3.13 stay in `ci.yml`. This
file is the architecture, ROS distro, and L4T coverage.
