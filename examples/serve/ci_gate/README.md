# Add the Serve example gate to a repo

This example is the CPU gate. It does two things that exist on main:

1. `ohho bench validate` on the three schema fixtures
   (`ohho.bench.validate_path`). Those JSON files are schema samples. This
   script does not print their latency or accuracy fields and does not treat
   them as measured Serve results.
2. It checks that `.github/workflows/release.yml` runs
   `python examples/serve/run_cpu.py` and that the publish job
   `needs: [serve-examples]`. A non-zero example then fails the release.
   It also checks that the nightly GPU workflow still probes for a
   self-hosted runner before it starts.

Copy [`release-job.yml`](release-job.yml) into a release workflow and set the
publish job to `needs: [serve-examples]`. Run the suite locally with:

```bash
python examples/serve/run_cpu.py
```

## Hardware

CPU. GitHub-hosted `ubuntu-latest` in CI, or any machine with Python 3.10+.
No GPU and no network.

## Runtime

Under 5 seconds.

## Expected output

```text
example: ci-gate
status: ok
hardware: cpu
check: ohho bench validate
validated: ohho/bench/examples/vision.json
validated: ohho/bench/examples/vla.json
validated: ohho/bench/examples/llm.json
gate: .github/workflows/release.yml
gpu-workflow: .github/workflows/serve-examples-gpu.yml
runtime: under-5s
result: fixtures-only
```

`result: fixtures-only` means the schema files parsed. It is not a benchmark.
