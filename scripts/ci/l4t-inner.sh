#!/usr/bin/env bash
# Install ohho-os inside the public L4T base image and run import plus unit tests.
# The workflow sets OHHO_L4T_INNER=1 and does not pass a GPU to docker run.
set -euo pipefail

if [[ "${OHHO_L4T_INNER:-}" != "1" ]]; then
  echo "This script installs packages into the container rootfs." >&2
  echo "The L4T job sets OHHO_L4T_INNER=1." >&2
  exit 1
fi

arch=$(uname -m)
if [[ "${arch}" != "aarch64" ]]; then
  echo "L4T base is arm64. This container reports ${arch}." >&2
  exit 1
fi

script_dir=$(
  CDPATH=''
  cd -- "$(dirname -- "$0")" && pwd
)
root=$(
  CDPATH=''
  cd -- "${script_dir}/../.." && pwd
)
cd "${root}"

export DEBIAN_FRONTEND=noninteractive

if ! apt-get update -qq; then
  echo "apt-get update failed. Commenting NVIDIA apt entries and retrying Ubuntu archives."
  shopt -s nullglob
  for f in /etc/apt/sources.list /etc/apt/sources.list.d/*; do
    [[ -f "${f}" ]] || continue
    sed -i -E '/nvidia|repo\.download\.nvidia|jetson/s/^/# ohho-ci /' "${f}"
  done
  shopt -u nullglob
  apt-get update -qq
fi

apt-get install -y -qq --no-install-recommends \
  python3 \
  python3-pip \
  python3-venv \
  ca-certificates

python3 - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "ohho-os requires Python >= 3.10; this image has %s" % sys.version
    )
print("container python", sys.version.split()[0])
PY

python3 -m venv /opt/ohho-ci
/opt/ohho-ci/bin/pip install -U pip
/opt/ohho-ci/bin/pip install -e '.[serve]'

cd /tmp
/opt/ohho-ci/bin/python - <<'PY'
import platform

import ohho

print("import ohho", ohho.__version__, platform.machine(), platform.python_version())
PY

export PATH="/opt/ohho-ci/bin:${PATH}"
hash -r
cd "${root}"
bash "${root}/scripts/ci/run-ohho-runtime-tests.sh"
python -m unittest discover -s tests
