"""ohho.bench/v1 schema, validator, generated types, and golden results."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib.resources import files
from io import StringIO
from pathlib import Path
from typing import Any

from ohho.bench import SCHEMA_ID, BenchConversionError, BenchValidationError, convert
from ohho.bench.schema import load_schema, validate, validate_path
from ohho.cli import main

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = ("vision", "vla", "llm")


def _packaged(name: str) -> str:
    resource = files("ohho.bench").joinpath(f"examples/{name}.json")
    return resource.read_bytes().decode("utf-8")


def _example(name: str) -> dict[str, Any]:
    document = json.loads(_packaged(name))
    if not isinstance(document, dict):
        raise AssertionError(f"{name} example must be an object")
    return document


def _errors(document: dict[str, Any]) -> list[str]:
    try:
        validate(document)
    except BenchValidationError as exc:
        return [str(issue) for issue in exc.issues]
    return []


def _object_schemas(schema: object, found: list[dict[str, Any]]) -> None:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            found.append(schema)
        for value in schema.values():
            _object_schemas(value, found)
    elif isinstance(schema, list):
        for item in schema:
            _object_schemas(item, found)


class TestBenchSchemaPackage(unittest.TestCase):
    def test_schema_id_and_package_data(self) -> None:
        raw = files("ohho.bench").joinpath("schemas/bench.v1.json").read_bytes()
        schema = json.loads(raw.decode("utf-8"))
        self.assertEqual(
            schema["$id"], "https://ohho-robotics.com/schemas/ohho.bench/v1.json"
        )
        self.assertEqual(schema["properties"]["schema"]["const"], SCHEMA_ID)
        self.assertEqual(load_schema()["title"], "BenchResult")

    def test_required_blocks(self) -> None:
        schema = load_schema()
        self.assertEqual(
            schema["required"],
            [
                "schema",
                "run",
                "latency",
                "throughput",
                "accuracy",
                "telemetry",
                "environment",
            ],
        )
        run = schema["$defs"]["run"]
        self.assertEqual(
            run["required"],
            ["model_sha", "artifact_sha", "target", "manifest_ref", "host"],
        )
        stats = schema["$defs"]["latency_stats"]["required"]
        self.assertEqual(
            stats,
            ["mean", "median", "p50", "p90", "p95", "p99", "min", "max", "std", "unit"],
        )

    def test_object_schemas_keep_additional_properties_open(self) -> None:
        found: list[dict[str, Any]] = []
        _object_schemas(load_schema(), found)
        self.assertGreaterEqual(len(found), 10)
        for schema in found:
            self.assertIn("additionalProperties", schema)
            self.assertIsNot(schema["additionalProperties"], False)

    def test_examples_are_package_data(self) -> None:
        seen: set[str] = set()
        for name in _EXAMPLES:
            text = _packaged(name)
            seen.add(text)
            document = json.loads(text)
            self.assertEqual(document["run"]["target"], name)
            self.assertEqual(document["schema"], SCHEMA_ID)
        self.assertEqual(len(seen), 3)


class TestBenchGoldenResults(unittest.TestCase):
    def test_vision_vla_and_llm_examples_validate(self) -> None:
        for name in _EXAMPLES:
            with self.subTest(name=name):
                self.assertEqual(_errors(_example(name)), [])

    def test_targets_cover_the_three_workloads(self) -> None:
        targets = {_example(name)["run"]["target"] for name in _EXAMPLES}
        self.assertEqual(targets, {"vision", "vla", "llm"})


class TestBenchValidator(unittest.TestCase):
    def test_missing_required_field_reports_json_path(self) -> None:
        document = _example("vision")
        del document["run"]["model_sha"]
        self.assertIn(
            "$.run.model_sha: required property is missing",
            _errors(document),
        )

    def test_missing_nested_and_stage_paths(self) -> None:
        document = _example("vla")
        del document["run"]["host"]["name"]
        del document["latency"]["stages"][0]["p99"]
        errors = _errors(document)
        self.assertIn("$.run.host.name: required property is missing", errors)
        self.assertIn("$.latency.stages[0].p99: required property is missing", errors)

    def test_reports_every_missing_field(self) -> None:
        document = _example("llm")
        del document["throughput"]
        del document["accuracy"]
        errors = _errors(document)
        self.assertIn("$.throughput: required property is missing", errors)
        self.assertIn("$.accuracy: required property is missing", errors)

    def test_wrong_type_and_bool_are_not_numbers(self) -> None:
        document = _example("vision")
        document["latency"]["mean"] = True
        document["run"]["target"] = 1
        errors = _errors(document)
        self.assertIn("$.latency.mean: expected number, got boolean", errors)
        self.assertIn("$.run.target: expected string, got integer", errors)

    def test_whole_number_float_is_an_integer(self) -> None:
        document = _example("vision")
        document["telemetry"]["samples"] = 1.0
        self.assertEqual(_errors(document), [])
        document["telemetry"]["samples"] = 1.5
        self.assertIn(
            "$.telemetry.samples: expected integer, got number",
            _errors(document),
        )

    def test_non_finite_number_is_rejected(self) -> None:
        document = _example("vision")
        document["latency"]["mean"] = float("nan")
        errors = _errors(document)
        self.assertTrue(
            any(
                item.startswith("$.latency.mean:") and "non-finite" in item
                for item in errors
            )
        )

    def test_digest_pattern(self) -> None:
        document = _example("vision")
        document["run"]["model_sha"] = "abcdef0"
        self.assertEqual(_errors(document), [])
        short = _example("vision")
        short["run"]["model_sha"] = "abc"
        self.assertIn(
            "$.run.model_sha: does not match ^(sha256:)?[0-9a-fA-F]{7,128}$",
            _errors(short),
        )
        prefixed = _example("vision")
        prefixed["run"]["artifact_sha"] = "sha256:" + ("ab" * 32)
        self.assertEqual(_errors(prefixed), [])

    def test_blank_manifest_ref(self) -> None:
        document = _example("llm")
        document["run"]["manifest_ref"] = ""
        self.assertIn(
            "$.run.manifest_ref: expected length >= 1",
            _errors(document),
        )

    def test_unknown_schema_id(self) -> None:
        document = _example("vision")
        document["schema"] = "ohho.bench/v2"
        self.assertEqual(
            _errors(document),
            ["$.schema: expected 'ohho.bench/v1'"],
        )

    def test_additive_field_is_allowed(self) -> None:
        document = _example("vla")
        document["labels"] = {"suite": "serve"}
        document["run"]["note"] = "closed loop"
        self.assertEqual(_errors(document), [])

    def test_optional_fields_may_be_omitted(self) -> None:
        document = _example("llm")
        del document["run"]["id"]
        del document["latency"]["samples"]
        del document["accuracy"]["metrics"]
        document["latency"]["stages"] = []
        self.assertEqual(_errors(document), [])

    def test_stages_key_is_required(self) -> None:
        document = _example("vision")
        del document["latency"]["stages"]
        self.assertIn(
            "$.latency.stages: required property is missing",
            _errors(document),
        )

    def test_extra_version_must_be_a_string(self) -> None:
        document = _example("vision")
        document["environment"]["versions"]["numpy"] = 1
        self.assertIn(
            "$.environment.versions.numpy: expected string, got integer",
            _errors(document),
        )

    def test_negative_latency_is_rejected(self) -> None:
        document = _example("vision")
        document["latency"]["min"] = -0.1
        self.assertIn("$.latency.min: expected >= 0", _errors(document))

    def test_string_latency_is_one_object_error(self) -> None:
        document = _example("vision")
        document["latency"] = "slow"
        errors = _errors(document)
        self.assertEqual(errors, ["$.latency: expected object, got string"])

    def test_additional_property_can_be_forbidden(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["a"],
            "properties": {"a": {"type": "string"}},
        }
        with self.assertRaises(BenchValidationError) as caught:
            validate({"a": "x", "b": 1}, schema)
        self.assertEqual(
            [str(issue) for issue in caught.exception.issues],
            ["$.b: additional property is not allowed"],
        )

    def test_odd_property_name_uses_bracket_path(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": True,
            "required": ["a-b"],
            "properties": {"a-b": {"type": "number", "minimum": 0}},
        }
        with self.assertRaises(BenchValidationError) as caught:
            validate({}, schema)
        self.assertEqual(
            str(caught.exception),
            "$['a-b']: required property is missing",
        )

    def test_unsupported_keyword_names_the_schema_path(self) -> None:
        with self.assertRaises(BenchValidationError) as caught:
            validate({}, {"oneOf": []})
        self.assertIn("unsupported schema keyword 'oneOf'", str(caught.exception))
        self.assertTrue(str(caught.exception).startswith("#"))

    def test_cyclic_ref_is_rejected(self) -> None:
        schema = {
            "$defs": {"loop": {"$ref": "#/$defs/loop"}},
            "$ref": "#/$defs/loop",
        }
        with self.assertRaises(BenchValidationError) as caught:
            validate({}, schema)
        self.assertIn("cyclic $ref", str(caught.exception))

    def test_validate_path_rejects_bad_json_and_missing_file(self) -> None:
        missing = _ROOT / "ohho" / "bench" / "examples" / "missing-result.json"
        with self.assertRaises(BenchValidationError) as caught:
            validate_path(missing)
        self.assertIn("file not found", str(caught.exception))
        self.assertTrue(str(caught.exception).startswith("$:"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(BenchValidationError) as caught_json:
                validate_path(path)
            self.assertIn("invalid JSON", str(caught_json.exception))
            path.write_bytes(b"\xff")
            with self.assertRaises(BenchValidationError) as caught_utf:
                validate_path(path)
            self.assertIn("invalid UTF-8", str(caught_utf.exception))


class TestBenchConvert(unittest.TestCase):
    def test_identity_keeps_additive_fields_and_copies(self) -> None:
        document = _example("vision")
        document["future"] = {"keep": True}
        converted = convert(document, to=SCHEMA_ID)
        self.assertEqual(converted["future"], {"keep": True})
        converted["run"]["target"] = "changed"
        self.assertEqual(document["run"]["target"], "vision")

    def test_missing_schema_and_unknown_target(self) -> None:
        with self.assertRaises(BenchConversionError) as missing:
            convert({}, to=SCHEMA_ID)
        self.assertIn("$.schema: required property is missing", str(missing.exception))
        with self.assertRaises(BenchConversionError) as unknown:
            convert({"schema": SCHEMA_ID}, to="ohho.bench/v2")
        self.assertIn("no converter", str(unknown.exception))
        self.assertIn("ohho.bench/v2", str(unknown.exception))


class TestBenchCli(unittest.TestCase):
    def test_validate_golden_file(self) -> None:
        path = _ROOT / "ohho" / "bench" / "examples" / "vision.json"
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["bench", "validate", str(path)])
        self.assertEqual(code, 0)
        self.assertIn("valid ohho.bench/v1:", stdout.getvalue())
        self.assertIn("vision.json", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_validate_missing_field(self) -> None:
        document = _example("vla")
        del document["run"]["artifact_sha"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vla.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["bench", "validate", str(path)])
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn(
            "$.run.artifact_sha: required property is missing",
            stderr.getvalue(),
        )

    def test_validate_rejects_invalid_json_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("[]", encoding="utf-8")
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["bench", "validate", str(path)])
            self.assertEqual(code, 1)
            self.assertIn("expected object", stderr.getvalue())
            missing = Path(tmp) / "nope.json"
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["bench", "validate", str(missing)])
            self.assertEqual(code, 1)
            self.assertIn("file not found", stderr.getvalue())

    def test_llm_block_is_optional_and_versioned(self) -> None:
        schema = load_schema()
        self.assertNotIn("llm", schema["required"])
        block = schema["$defs"]["llm"]
        self.assertEqual(block["title"], "LlmBench")
        self.assertEqual(block["properties"]["schema"]["const"], "ohho.bench.llm/v1")
        self.assertTrue(block["additionalProperties"])
        document = _example("llm")
        self.assertNotIn("llm", document)
        self.assertEqual(_errors(document), [])
        document["llm"] = {"schema": "ohho.bench.llm/v2"}
        errors = _errors(document)
        self.assertTrue(any(item.startswith("$.llm.schema:") for item in errors))

    def test_bench_requires_a_subcommand(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main(["bench"])
        self.assertEqual(caught.exception.code, 2)


class TestBenchGeneratedTypes(unittest.TestCase):
    def test_titles_are_generated_in_python_and_typescript(self) -> None:
        schema = load_schema()
        titles = [schema["title"]]
        for sub in schema["$defs"].values():
            title = sub.get("title")
            if isinstance(title, str):
                titles.append(title)
        import ohho.bench.generated.types as types

        typescript = (_ROOT / "ohho" / "bench" / "generated" / "bench.v1.ts").read_text(
            encoding="utf-8"
        )
        for title in titles:
            self.assertTrue(hasattr(types, title), title)
            self.assertIn(f"export interface {title}", typescript)
        self.assertEqual(types.BENCH_SCHEMA_ID, SCHEMA_ID)
        self.assertIn("[key: string]: string", typescript)

    def test_typed_dict_required_keys(self) -> None:
        from ohho.bench.generated.types import (
            BenchResult,
            BenchRun,
            EnvironmentVersions,
            LatencyStage,
        )

        self.assertIn("schema", BenchResult.__required_keys__)
        self.assertIn("model_sha", BenchRun.__required_keys__)
        self.assertIn("id", BenchRun.__optional_keys__)
        self.assertIn("name", LatencyStage.__required_keys__)
        self.assertIn("p99", LatencyStage.__required_keys__)
        self.assertIn("samples", LatencyStage.__optional_keys__)
        self.assertIn("python", EnvironmentVersions.__required_keys__)
        self.assertIn("ohho", EnvironmentVersions.__required_keys__)

    def test_regenerating_types_produces_no_diff(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(_ROOT / "scripts" / "generate_bench_types.py"),
                "--check",
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class TestBenchPolicyDoc(unittest.TestCase):
    def test_policy_requires_a_v2_converter(self) -> None:
        text = (_ROOT / "docs" / "serve" / "bench-schema.md").read_text(
            encoding="utf-8"
        )
        lowered = text.lower()
        self.assertIn("ohho.bench/v2", text)
        self.assertIn("converter", lowered)
        self.assertIn("additional propert", lowered)
        self.assertIn("ohho bench validate", text)


if __name__ == "__main__":
    unittest.main()
