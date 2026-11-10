"""Check the release gate and validate the bench schema fixtures.

This is the CPU example that runs a command which exists on main:
``ohho.bench.validate_path``, the same check as ``ohho bench validate``.
The schema fixtures are not measured results. The script does not print
their latency or accuracy fields.
"""

from __future__ import annotations


def _support() -> object:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "support.py"
    spec = importlib.util.spec_from_file_location("ohho_serve_example_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    """Validate fixtures and the release / GPU workflow wiring."""
    support = _support()
    root = support.REPO_ROOT
    release_path = root / ".github" / "workflows" / "release.yml"
    gpu_path = root / ".github" / "workflows" / "serve-examples-gpu.yml"
    problems = support.release_text_problems(release_path.read_text(encoding="utf-8"))
    if gpu_path.is_file():
        problems.extend(
            support.gpu_workflow_problems(gpu_path.read_text(encoding="utf-8"))
        )
    else:
        problems.append("missing .github/workflows/serve-examples-gpu.yml")
    validated: list[str] = []
    if not problems:
        from ohho.bench import BenchValidationError, validate_path

        try:
            for rel in support.FIXTURES:
                validate_path(root / rel)
                validated.append(rel)
        except BenchValidationError as exc:
            first = str(exc).splitlines()[0]
            problems.append(first)
    if problems:
        print("example: ci-gate")
        print("status: fail")
        print("hardware: cpu")
        for item in problems:
            print(f"problem: {item}")
        print("runtime: under-5s")
        print("result: none")
        return 1
    print("example: ci-gate")
    print("status: ok")
    print("hardware: cpu")
    print("check: ohho bench validate")
    for rel in validated:
        print(f"validated: {rel}")
    print("gate: .github/workflows/release.yml")
    print("gpu-workflow: .github/workflows/serve-examples-gpu.yml")
    print("runtime: under-5s")
    print("result: fixtures-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
