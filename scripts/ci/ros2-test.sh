#!/usr/bin/env bash
# Run the fake-node Ros2Runtime / Ros2Transport tests.
# tests/test_ros2.py expects rclpy to be absent. This step does not source
# setup.bash, and it drops /opt/ros entries from PYTHONPATH.
set -eo pipefail

script_dir=$(
  CDPATH=''
  cd -- "$(dirname -- "$0")" && pwd
)
root=$(
  CDPATH=''
  cd -- "${script_dir}/../.." && pwd
)
cd "${root}"

# shellcheck disable=SC1091
source "${script_dir}/ros2-venv.sh"

if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH=$(
    python3 -c 'import os; print(":".join(p for p in os.environ.get("PYTHONPATH", "").split(":") if p and "/opt/ros/" not in p))'
  )
  export PYTHONPATH
fi
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH || true

python3 - <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("rclpy")
if spec is not None:
    origin = getattr(spec, "origin", None)
    sys.exit(
        "rclpy is importable (%s). tests/test_ros2.py expects it to be "
        "absent (test_without_rclpy_raises, "
        "test_auto_prefers_native_without_rclpy)." % (origin,)
    )
print("rclpy is not importable; running tests/test_ros2.py")
PY

python3 -m unittest discover -s tests -p 'test_ros2.py' -v
