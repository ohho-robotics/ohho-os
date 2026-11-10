"""ohho.job/v1 schema, generated types, and local job commands."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib.resources import files
from pathlib import Path

from ohho.cli import main
from ohho.serve.job import (
    SCHEMA_VERSION,
    JobSpecInvalid,
    format_error,
    load_schema,
    run_job,
    validate_path,
    validate_text,
)
from ohho.serve.job.errors import JobSpecError
from ohho.serve.job.generated import Job
from ohho.serve.job.run import BUILTIN_ENGINES, StepResult, exit_code
from ohho.serve.job.validate import HANDLED_KEYWORDS, collect_errors
from ohho.serve.job.yaml_subset import YamlParseError, load_yaml_subset

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "ohho" / "serve" / "job" / "examples"

_LINE = re.compile(
    r"^(?P<path>\$(?:\.[A-Za-z_][A-Za-z0-9_]*"
    r'|\[\"(?:\\.|[^"\\])*\"\]|\[\d+\])*)'
    r": (?P<code>[a-z][a-z0-9_]*): (?P<message>.+)$"
)

_BASE = """\
schema_version: ohho.job/v1
model:
  version: repo/model@v1
targets:
  - cpu-local
steps:
  - bench
"""


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def _generator():
    path = ROOT / "scripts" / "generate_job_types.py"
    spec = importlib.util.spec_from_file_location("generate_job_types", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load generate_job_types.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _objects(node: object):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _objects(value)


def _keywords(node: object) -> set[str]:
    """Schema keywords only. Property names and ``$defs`` names are not keywords."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(str(key))
            if key in ("properties", "$defs") and isinstance(value, dict):
                for child in value.values():
                    found |= _keywords(child)
            else:
                found |= _keywords(value)
    elif isinstance(node, list):
        for value in node:
            found |= _keywords(value)
    return found


class TestYamlSubset(unittest.TestCase):
    def test_full_example_scalars_and_nesting(self):
        data = load_yaml_subset((EXAMPLES / "full.yaml").read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)
        assert isinstance(data, dict)
        self.assertEqual(data["schema_version"], "ohho.job/v1")
        self.assertEqual(data["model"], {"version": "repo/model@v1"})
        self.assertEqual(data["targets"], ["edge-cpu", "workstation"])
        self.assertEqual(data["steps"], ["optimize", "parity", "bench"])
        self.assertEqual(data["precision"], "fp16")
        self.assertEqual(data["parity"]["samples"], 8)
        self.assertEqual(data["parity"]["max_abs_error"], 0.001)
        self.assertEqual(data["budget"]["max_usd"], 2.5)
        self.assertIs(data["outputs"]["wandb"]["enabled"], True)
        self.assertEqual(data["outputs"]["prometheus"]["port"], 9090)

    def test_flow_and_sequence_of_mappings(self):
        data = load_yaml_subset(
            "steps: [optimize, parity]\n"
            "items:\n"
            "  - name: a\n"
            "    value: 1\n"
            "  - name: b\n"
            "    value: 2\n"
        )
        self.assertEqual(data["steps"], ["optimize", "parity"])
        self.assertEqual(
            data["items"],
            [{"name": "a", "value": 1}, {"name": "b", "value": 2}],
        )

    def test_quotes_comments_and_null(self):
        data = load_yaml_subset(
            "note: \"hash # 1\"\nflag: false\nempty: null\nquoted: 'a''b'\n"
        )
        self.assertEqual(data["note"], "hash # 1")
        self.assertIs(data["flag"], False)
        self.assertIsNone(data["empty"])
        self.assertEqual(data["quoted"], "a'b")

    def test_version_string_with_colon(self):
        data = load_yaml_subset('version: "sha256:0123456789abcdef"\n')
        self.assertEqual(data["version"], "sha256:0123456789abcdef")

    def test_rejects_tabs_anchors_multiline_and_duplicate_keys(self):
        cases = (
            ("\tkey: 1\n", "tabs are not allowed in indentation"),
            ("key: &anchor value\n", "anchors, aliases and tags are not supported"),
            ("key: |\n  hello\n", "multiline strings are not supported"),
            ("key: 1\nkey: 2\n", 'duplicate key "key"'),
            ("", "document is empty"),
        )
        for text, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(YamlParseError) as ctx:
                    load_yaml_subset(text)
                self.assertIn(message, ctx.exception.message)


class TestJobSchema(unittest.TestCase):
    def test_schema_identity_and_closed_objects(self):
        schema = load_schema()
        self.assertEqual(
            schema["$id"], "https://ohho-robotics.com/schemas/ohho.job/v1.json"
        )
        self.assertEqual(schema["properties"]["schema_version"]["const"], "ohho.job/v1")
        self.assertIn("schema_version", schema["required"])
        self.assertEqual(
            schema["properties"]["steps"]["items"]["enum"],
            ["optimize", "parity", "bench"],
        )
        self.assertEqual(_keywords(schema) - HANDLED_KEYWORDS, set())
        for obj in _objects(schema):
            self.assertIs(obj["additionalProperties"], False)

    def test_packaged_schema_and_example(self):
        packaged = (
            files("ohho.serve.job").joinpath("schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(json.loads(packaged)["title"], "Job")
        example = (
            files("ohho.serve.job")
            .joinpath("examples/full.yaml")
            .read_text(encoding="utf-8")
        )
        job = validate_text(example)
        self.assertEqual(job.schema_version, SCHEMA_VERSION)

    def test_bounds_keywords(self):
        errors = collect_errors(
            {
                "type": "object",
                "additionalProperties": {"type": "integer", "minimum": 1},
                "properties": {
                    "name": {"type": "string", "maxLength": 2},
                    "nums": {
                        "type": "array",
                        "maxItems": 1,
                        "items": {"type": "integer"},
                    },
                },
            },
            {"name": "abcd", "nums": [1, 2], "n": 0},
        )
        self.assertEqual(
            [str(error) for error in errors],
            [
                "$.name: max_length: string length must be <= 2",
                "$.nums: max_items: array length must be <= 1",
                "$.n: minimum: value must be >= 1",
            ],
        )


class TestExamples(unittest.TestCase):
    def test_each_example_records_schema_version(self):
        paths = sorted(EXAMPLES.glob("*.yaml"))
        self.assertGreaterEqual(len(paths), 3)
        for path in paths:
            with self.subTest(path=path.name):
                job = validate_path(path)
                self.assertEqual(job.schema_version, "ohho.job/v1")
                self.assertEqual(job.to_dict()["schema_version"], "ohho.job/v1")
                again = validate_text(
                    json.dumps(job.to_dict()),
                    suffix=".json",
                )
                self.assertEqual(again, job)

    def test_full_example_fields(self):
        job = validate_path(EXAMPLES / "full.yaml")
        self.assertEqual(job.steps, ("optimize", "parity", "bench"))
        self.assertEqual(job.targets, ("edge-cpu", "workstation"))
        self.assertEqual(job.precision, "fp16")
        self.assertEqual(job.backend, "onnx")
        assert job.parity is not None
        self.assertEqual(job.parity.samples, 8)
        assert job.outputs is not None and job.outputs.prometheus is not None
        self.assertEqual(job.outputs.prometheus.port, 9090)


class TestInvalidSpecs(unittest.TestCase):
    def test_several_specs_report_path_and_stable_message(self):
        cases = {
            "missing steps": (
                "schema_version: ohho.job/v1\n"
                "model:\n"
                "  version: repo/model@v1\n"
                "targets:\n"
                "  - cpu-local\n",
                ['$.steps: required: missing required property "steps"'],
            ),
            "missing model version": (
                "schema_version: ohho.job/v1\n"
                "model: {}\n"
                "targets:\n"
                "  - cpu-local\n"
                "steps:\n"
                "  - bench\n",
                ['$.model.version: required: missing required property "version"'],
            ),
            "bad step": (
                _BASE.replace("  - bench\n", "  - train\n"),
                [
                    "$.steps[0]: enum: value must be one of "
                    '"optimize", "parity", "bench"'
                ],
            ),
            "wrong schema version": (
                _BASE.replace("ohho.job/v1", "ohho.job/v0", 1),
                ['$.schema_version: const: value must be "ohho.job/v1"'],
            ),
            "schema version type": (
                _BASE.replace("schema_version: ohho.job/v1\n", "schema_version: 1\n"),
                ["$.schema_version: type: expected string, got integer"],
            ),
            "empty targets": (
                _BASE.replace("targets:\n  - cpu-local\n", "targets: []\n"),
                ["$.targets: min_items: array length must be >= 1"],
            ),
            "target type": (
                _BASE.replace("  - cpu-local\n", "  - 3\n"),
                ["$.targets[0]: type: expected string, got integer"],
            ),
            "duplicate steps": (
                _BASE.replace(
                    "  - bench\n",
                    "  - optimize\n  - optimize\n",
                ),
                ["$.steps[1]: unique_items: duplicate item"],
            ),
            "negative budget": (
                _BASE + "budget:\n  max_usd: -1\n",
                ["$.budget.max_usd: minimum: value must be >= 0"],
            ),
            "zero iterations": (
                _BASE + "bench:\n  iterations: 0\n",
                ["$.bench.iterations: minimum: value must be >= 1"],
            ),
            "zero timeout": (
                _BASE + "bench:\n  timeout_s: 0\n",
                ["$.bench.timeout_s: exclusive_minimum: value must be > 0"],
            ),
            "empty version": (
                _BASE.replace("version: repo/model@v1\n", 'version: ""\n'),
                ["$.model.version: min_length: string length must be >= 1"],
            ),
            "unknown property": (
                _BASE + "robot: sim\n",
                ['$.robot: additional_property: unknown property "robot"'],
            ),
            "unknown nested property": (
                _BASE + "parity:\n  tolerance: 1\n",
                [
                    '$.parity.tolerance: additional_property: unknown property "tolerance"'
                ],
            ),
            "dotted key": (
                _BASE + '"a.b": 1\n',
                ['$["a.b"]: additional_property: unknown property "a.b"'],
            ),
            "cosine above one": (
                _BASE + "parity:\n  min_cosine: 1.1\n",
                ["$.parity.min_cosine: maximum: value must be <= 1"],
            ),
            "port above range": (
                _BASE + "outputs:\n  prometheus:\n    enabled: true\n    port: 70000\n",
                ["$.outputs.prometheus.port: maximum: value must be <= 65535"],
            ),
            "port zero": (
                _BASE + "outputs:\n  prometheus:\n    enabled: true\n    port: 0\n",
                ["$.outputs.prometheus.port: minimum: value must be >= 1"],
            ),
            "samples float": (
                _BASE + "parity:\n  samples: 1.5\n",
                ["$.parity.samples: type: expected integer, got number"],
            ),
            "samples bool": (
                _BASE + "parity:\n  samples: true\n",
                ["$.parity.samples: type: expected integer, got boolean"],
            ),
            "steps not array": (
                _BASE.replace("steps:\n  - bench\n", "steps: bench\n"),
                ["$.steps: type: expected array, got string"],
            ),
            "root array": (
                "- bench\n",
                ["$: type: expected object, got array"],
            ),
            "bad precision and missing steps and extra key": (
                "schema_version: ohho.job/v1\n"
                "model:\n"
                "  version: repo/model@v1\n"
                "targets:\n"
                "  - cpu-local\n"
                "precision: fp64\n"
                "extra: 1\n",
                [
                    '$.steps: required: missing required property "steps"',
                    '$.precision: enum: value must be one of "fp32", "fp16", "bf16", "int8"',
                    '$.extra: additional_property: unknown property "extra"',
                ],
            ),
        }
        self.assertGreaterEqual(len(cases), 8)
        for name, (text, expected) in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(JobSpecInvalid) as ctx:
                    validate_text(text)
                lines = [str(error) for error in ctx.exception.errors]
                self.assertEqual(lines, expected)
                for line, error in zip(lines, ctx.exception.errors):
                    match = _LINE.fullmatch(line)
                    self.assertIsNotNone(match, line)
                    assert match is not None
                    path, code, message = line.split(": ", 2)
                    self.assertEqual(path, error.path)
                    self.assertEqual(code, error.code)
                    self.assertEqual(message, error.message)
                    self.assertEqual(line, format_error(path, code, message))

    def test_parse_and_io_use_the_same_line(self):
        with self.assertRaises(JobSpecInvalid) as ctx:
            validate_text("key: |\n  text\n")
        self.assertEqual(
            str(ctx.exception.errors[0]),
            "$: parse: line 1: multiline strings are not supported",
        )
        with self.assertRaises(JobSpecInvalid) as ctx:
            validate_text("{", suffix=".json")
        self.assertTrue(str(ctx.exception.errors[0]).startswith("$: parse: line "))
        missing = ROOT / "ohho" / "serve" / "job" / "examples" / "missing.yaml"
        with self.assertRaises(JobSpecInvalid) as ctx:
            validate_path(missing)
        line = str(ctx.exception.errors[0])
        self.assertTrue(line.startswith("$: io: cannot read "))
        self.assertIn("not a file", line)
        self.assertIsNotNone(_LINE.fullmatch(line))


class TestGeneratedTypes(unittest.TestCase):
    def test_regenerating_produces_no_diff(self):
        self.assertEqual(_generator().check(), [])

    def test_generator_script_check(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "generate_job_types.py"),
                "--check",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout, "")

    def test_typescript_names_the_schema(self):
        text = (ROOT / "schemas" / "ts" / "job.v1.d.ts").read_text(encoding="utf-8")
        self.assertIn('export const JOB_SCHEMA_VERSION = "ohho.job/v1" as const;', text)
        self.assertIn('export type JobStep = "optimize" | "parity" | "bench";', text)
        self.assertIn("schema_version: JobSchemaVersion;", text)


class TestJobRun(unittest.TestCase):
    def test_builtin_engines_are_empty(self):
        self.assertEqual(dict(BUILTIN_ENGINES), {})

    def test_default_run_reports_each_step_unavailable(self):
        job = validate_path(EXAMPLES / "full.yaml")
        result = run_job(job)
        self.assertFalse(result.ok)
        self.assertEqual(result.schema_version, "ohho.job/v1")
        self.assertEqual(exit_code(result), 2)
        payload = result.to_dict()
        self.assertEqual(set(payload), {"schema_version", "ok", "steps"})
        steps = payload["steps"]
        assert isinstance(steps, list)
        self.assertEqual(
            steps,
            [
                {
                    "step": "optimize",
                    "status": "not_available",
                    "message": "optimize is not available in this build",
                },
                {
                    "step": "parity",
                    "status": "not_available",
                    "message": "parity is not available in this build",
                },
                {
                    "step": "bench",
                    "status": "not_available",
                    "message": "bench is not available in this build",
                },
            ],
        )

    def test_run_executes_only_registered_steps(self):
        job = validate_text(
            "schema_version: ohho.job/v1\n"
            "model:\n"
            "  version: repo/model@v1\n"
            "targets: [cpu-local]\n"
            "steps: [optimize, bench]\n"
        )
        seen: list[str] = []

        def bench(spec: Job) -> StepResult:
            seen.append(spec.schema_version)
            return StepResult("ignored", "ok", "bench engine ran")

        result = run_job(job, {"bench": bench})
        self.assertEqual(seen, ["ohho.job/v1"])
        self.assertEqual(result.steps[0].status, "not_available")
        self.assertEqual(
            result.steps[0].message, "optimize is not available in this build"
        )
        self.assertEqual(result.steps[1].step, "bench")
        self.assertEqual(result.steps[1].status, "ok")
        self.assertEqual(result.steps[1].message, "bench engine ran")
        self.assertFalse(result.ok)
        self.assertEqual(exit_code(result), 2)

    def test_engine_exception_is_an_error_not_a_fake_result(self):
        job = validate_text(_BASE)

        def boom(spec: Job) -> StepResult:
            del spec
            raise RuntimeError("backend missing")

        result = run_job(job, {"bench": boom})
        self.assertEqual(result.steps[0].status, "error")
        self.assertEqual(result.steps[0].message, "bench failed: backend missing")
        self.assertNotIn("not available", result.steps[0].message)
        self.assertEqual(set(result.steps[0].to_dict()), {"step", "status", "message"})
        self.assertEqual(exit_code(result), 1)


class TestJobCli(unittest.TestCase):
    def test_validate_and_run_example(self):
        spec = str(EXAMPLES / "full.yaml")
        code, out, err = _run_cli(["job", "validate", spec])
        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        self.assertEqual(out, "ok\nschema_version: ohho.job/v1\n")
        code, out, err = _run_cli(["job", "run", spec])
        self.assertEqual(code, 2, err + out)
        self.assertEqual(err, "")
        self.assertEqual(
            out,
            "schema_version: ohho.job/v1\n"
            "optimize is not available in this build\n"
            "parity is not available in this build\n"
            "bench is not available in this build\n",
        )
        for word in ("latency", "accuracy", "metric", "wandb_url"):
            self.assertNotIn(word, out)

    def test_validate_json_round_trip_line(self):
        spec = str(EXAMPLES / "bench.yaml")
        code, out, err = _run_cli(["job", "validate", spec, "--json"])
        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["schema_version"], "ohho.job/v1")
        self.assertEqual(payload["job"]["schema_version"], "ohho.job/v1")
        self.assertEqual(payload["job"]["steps"], ["bench"])

    def test_invalid_json_and_text(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "bad-step.yaml"
        path.write_text(
            _BASE.replace("  - bench\n", "  - train\n"),
            encoding="utf-8",
            newline="\n",
        )
        expected = (
            '$.steps[0]: enum: value must be one of "optimize", "parity", "bench"'
        )
        code, out, err = _run_cli(["job", "validate", str(path)])
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertEqual(err, expected + "\n")
        code, out, err = _run_cli(["job", "run", str(path), "--json"])
        self.assertEqual(code, 1)
        self.assertEqual(err, "")
        self.assertNotIn("not available", out)
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["schema_version"], "ohho.job/v1")
        error = payload["errors"][0]
        self.assertEqual(
            format_error(error["path"], error["code"], error["message"]),
            expected,
        )

    def test_missing_file(self):
        code, out, err = _run_cli(["job", "validate", str(EXAMPLES / "nope.yaml")])
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("$: io: cannot read ", err)
        self.assertIn("not a file", err)

    def test_job_requires_a_subcommand(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            main(["job"])
        self.assertEqual(ctx.exception.code, 2)

    def test_doc_states_the_message_contract(self):
        text = (ROOT / "docs" / "serve" / "job-spec.md").read_text(encoding="utf-8")
        self.assertIn("<path>: <code>: <message>", text)
        self.assertIn("not available in this build", text)
        self.assertIn("ohho.job/v1", text)


class TestErrorShape(unittest.TestCase):
    def test_empty_invalid_is_refused(self):
        with self.assertRaises(ValueError):
            JobSpecInvalid(())

    def test_format_error_is_one_line(self):
        self.assertEqual(
            format_error("$.steps", "required", 'missing required property "steps"'),
            '$.steps: required: missing required property "steps"',
        )
        self.assertEqual(
            str(JobSpecError("$.steps", "required", "line one\nline two")),
            "$.steps: required: line one line two",
        )


if __name__ == "__main__":
    unittest.main()
