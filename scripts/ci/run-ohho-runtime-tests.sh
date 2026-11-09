#!/usr/bin/env bash
# Test the ohho-runtime distribution when top-level runtime/ is in the checkout.
# ohho/runtime/ is the executor port and is already covered by the ohho-os tests.
set -euo pipefail

script_dir=$(
  CDPATH=''
  cd -- "$(dirname -- "$0")" && pwd
)
root=$(
  CDPATH=''
  cd -- "${script_dir}/../.." && pwd
)
cd "${root}"

if [[ ! -d runtime ]]; then
  echo "ohho-runtime: runtime/ is not in this checkout; skipping."
  exit 0
fi

echo "ohho-runtime: installing ${root}/runtime"
# The checkout's ohho-os is already installed. --no-deps keeps pip from
# replacing it with a released ohho-os.
python -m pip install -e runtime --no-deps
python -c 'import ohho_runtime; print("import ohho_runtime", getattr(ohho_runtime, "__version__", "ok"))'

if [[ ! -d runtime/tests ]]; then
  echo "ohho-runtime: runtime/tests is not present; skipping tests."
  exit 0
fi

python -m unittest discover -s runtime/tests
