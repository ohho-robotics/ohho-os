"""Pinned serve worker images (OHH-484).

These tests do not build container images, import torch, or call git.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from importlib.resources import files
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ohho import __version__
from ohho.adapters.sim import SimTransport
from ohho.agent import Agent
from ohho.cli import main
from ohho.optimize.toolchain import jetpack_from_tegra_release
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime
from ohho.worker_images.dockerfiles import dockerfile_path, render_dockerfile
from ohho.worker_images.errors import WorkerImageError
from ohho.worker_images.matrix import (
    SCHEMA_ID,
    get_variant,
    matrix_text,
    parse_image_ref,
    variants,
)
from ohho.worker_images.policy import (
    allowlist_from_json,
    default_allowlist,
    refusal_for_environment,
    refusal_reason,
)
from ohho.worker_images.probe import WorkerProbe, collect_probe
from ohho.worker_images.results import RESULT_SCHEMA, build_job_result, write_job_result
from ohho.worker_images.scan import critical_failures, load_exceptions
from ohho.worker_images.stamp import read_stamp, write_image_stamp
from ohho.worker_images.supply import release_plan
from ohho.worker_images.verify import verify_worker, versions_match

ROOT = Path(__file__).resolve().parents[1]
_DIGEST = "sha256:" + ("ab" * 32)
_OTHER_DIGEST = "sha256:" + ("cd" * 32)
_BANNED = ("unitree", "omnibot", "yahboom", "booster", "go2")
_ENV_KEYS = (
    "OHHO_WORKER_IMAGE",
    "OHHO_WORKER_IMAGE_DIGEST",
    "OHHO_WORKER_ALLOWLIST",
    "OHHO_WORKER_STAMP",
    "OHHO_WORKER_REQUIRE_STACK",
    "OHHO_WORKER_STACK",
    "CUDA_VERSION",
    "NVIDIA_CUDA_VERSION",
)


@contextmanager
def _env(**values: str | None):
    saved = {key: os.environ.get(key) for key in _ENV_KEYS}
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in values.items():
        if value is not None:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _probe(variant: str, **overrides: object) -> WorkerProbe:
    row = get_variant(variant)
    l4t = f"R{row.l4t}" if row.l4t else None
    cuda = row.cuda if row.cuda_identity == "required" else None
    fields: dict[str, object] = {
        "arch": row.arches[0],
        "torch": None,
        "ort": None,
        "tensorrt": None,
        "cuda": cuda,
        "l4t": l4t,
        "image": row.tag,
        "digest": None,
        "stack": "platform",
        "require_stack": False,
        "base_image": row.base_image,
    }
    fields.update(overrides)
    return WorkerProbe(**fields)  # type: ignore[arg-type]


def _run(argv: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class _FlagBrain:
    def __init__(self) -> None:
        self.called = False

    def run(self, robot: object, goal: str, max_steps: int) -> list[str]:
        self.called = True
        return [f"done: {goal}"]


class MatrixTest(unittest.TestCase):
    def test_three_pinned_tags_match_the_package_version(self) -> None:
        rows = variants()
        self.assertEqual(tuple(row.variant for row in rows), ("cpu", "cuda", "jetson"))
        self.assertEqual(rows[0].version, __version__)
        for row in rows:
            self.assertEqual(row.tag, f"ohho-serve-worker:{row.variant}-{__version__}")
            parsed = parse_image_ref(row.tag)
            self.assertEqual(parsed.variant, row.variant)
            self.assertIsNone(parsed.digest)

    def test_package_data_matches_the_schema(self) -> None:
        text = matrix_text()
        self.assertIn(SCHEMA_ID, text)
        resource = files("ohho.worker_images").joinpath("worker_matrix.json")
        self.assertEqual(resource.read_text(encoding="utf-8"), text)

    def test_doc_lists_every_pin(self) -> None:
        doc = (ROOT / "docs" / "serve" / "worker-images.md").read_text(encoding="utf-8")
        for row in variants():
            self.assertIn(row.tag, doc)
            self.assertIn(row.base_image, doc)
            self.assertIn(row.torch, doc)
            self.assertIn(row.ort, doc)
            if row.tensorrt:
                self.assertIn(row.tensorrt, doc)
            if row.cuda:
                self.assertIn(row.cuda, doc)
            if row.jetpack:
                self.assertIn(row.jetpack, doc)
            if row.l4t:
                self.assertIn(row.l4t, doc)

    def test_no_robot_vendor_names(self) -> None:
        paths = list((ROOT / "ohho" / "worker_images").rglob("*"))
        paths.extend((ROOT / "docker" / "serve-worker").rglob("*"))
        paths.append(ROOT / "docs" / "serve" / "worker-images.md")
        paths.append(ROOT / ".github" / "workflows" / "serve-worker-images.yml")
        for path in paths:
            if not path.is_file():
                continue
            if path.suffix not in {
                ".py",
                ".json",
                ".md",
                ".yml",
                ".cpu",
                ".cuda",
                ".jetson",
            } and not path.name.startswith("Dockerfile"):
                continue
            text = path.read_text(encoding="utf-8").lower()
            for name in _BANNED:
                self.assertNotIn(name, text, f"{name} in {path}")


class DockerfileTest(unittest.TestCase):
    def test_committed_files_match_the_matrix(self) -> None:
        for row in variants():
            rendered = render_dockerfile(row.variant).replace("\r\n", "\n")
            on_disk = dockerfile_path(row.variant).read_text(encoding="utf-8")
            on_disk = on_disk.replace("\r\n", "\n")
            self.assertEqual(on_disk, rendered)
            self.assertIn(f"ARG BASE_IMAGE={row.base_image}", rendered)
            self.assertIn(row.tag, rendered)
            self.assertIn(f"torch=={row.torch}", rendered)
            self.assertIn(f"{row.ort_package}=={row.ort}", rendered)
            self.assertIn('CMD ["targets", "verify"]', rendered)
            self.assertIsNone(re.search(r"\bgit\b", rendered))
        jetson = render_dockerfile("jetson")
        self.assertIn("nvcr.io/nvidia/l4t-base:r36.2.0", jetson)
        cuda = render_dockerfile("cuda")
        self.assertIn("tensorrt==10.0.1", cuda)


class WorkflowTest(unittest.TestCase):
    def test_ci_builds_verifies_scans_and_signs(self) -> None:
        text = (ROOT / ".github" / "workflows" / "serve-worker-images.yml").read_text(
            encoding="utf-8"
        )
        for name in (
            "docker/serve-worker/Dockerfile.${{ matrix.variant }}",
            "variant: cpu",
            "variant: cuda",
            "variant: jetson",
            "ubuntu-24.04-arm",
            "OHHO_WORKER_STACK=platform",
            "targets verify",
            "anchore/sbom-action@v0",
            "anchore/scan-action@v7",
            "cosign sign-blob",
            "refs/tags/v",
            "worker-job-result.json",
            "image_digest",
            "scripts/worker_vuln_gate.py",
            "--python-only",
            "scripts/record_worker_job.py",
            "sbom-${{ matrix.variant }}.cdx.json",
            "sbom-${{ matrix.variant }}.spdx.json",
            "${{ matrix.variant }}.image.sig.bundle",
        ):
            self.assertIn(name, text)
        self.assertNotIn("docker push", text)
        for item in release_plan():
            self.assertIn("cosign sign-blob", item["sign_command"])
            self.assertEqual(item["cyclonedx"], f"sbom-{item['variant']}.cdx.json")
            self.assertEqual(item["signature"], f"{item['variant']}.image.sig.bundle")
            self.assertEqual(item["job_result"], "worker-job-result.json")


class VerifyTest(unittest.TestCase):
    def test_platform_images_pass_on_their_arch(self) -> None:
        for name in ("cpu", "cuda", "jetson"):
            report = verify_worker(probe=_probe(name))
            self.assertTrue(report.text().startswith(f"ok: {get_variant(name).tag}"))
            self.assertEqual(report.to_dict()["ok"], True)
            self.assertIsNone(report.image_digest)

    def test_digest_is_recorded_when_the_probe_has_one(self) -> None:
        report = verify_worker(probe=_probe("cpu", digest=_DIGEST))
        self.assertEqual(report.image_digest, _DIGEST)
        self.assertIn(f"image_digest: {_DIGEST}", report.text())

    def test_cpu_rejects_the_wrong_arch_cuda_or_l4t(self) -> None:
        with self.assertRaises(WorkerImageError) as caught:
            verify_worker(probe=_probe("cpu", arch="aarch64"))
        self.assertIn("requires arch", str(caught.exception))
        with self.assertRaises(WorkerImageError) as caught:
            verify_worker(probe=_probe("cpu", cuda="12.4.1"))
        self.assertIn("does not pin CUDA", str(caught.exception))
        with self.assertRaises(WorkerImageError) as caught:
            verify_worker(probe=_probe("cpu", l4t="R36.2.0"))
        self.assertIn("does not use an L4T base", str(caught.exception))

    def test_cuda_requires_the_pinned_cuda_version(self) -> None:
        with self.assertRaises(WorkerImageError) as caught:
            verify_worker(probe=_probe("cuda", cuda=None))
        self.assertIn("CUDA version is not set", str(caught.exception))
        with self.assertRaises(WorkerImageError) as caught:
            verify_worker(probe=_probe("cuda", cuda="12.0.0"))
        self.assertIn("pin is 12.4.1", str(caught.exception))

    def test_jetson_requires_the_l4t_release(self) -> None:
        sample = (
            "# R36 (release), REVISION: 2.0, GCID: 1, BOARD: generic, "
            "EABI: aarch64, DATE: Thu Jan 1 00:00:00 UTC 2024\n"
        )
        release = jetpack_from_tegra_release(sample)
        self.assertEqual(release, "R36.2.0")
        report = verify_worker(probe=_probe("jetson", l4t=release))
        self.assertEqual(report.variant, "jetson")
        with self.assertRaises(WorkerImageError) as caught:
            verify_worker(probe=_probe("jetson", l4t=None))
        self.assertIn("L4T release is not set", str(caught.exception))
        with self.assertRaises(WorkerImageError) as caught:
            verify_worker(probe=_probe("jetson", l4t="R36.4.0"))
        self.assertIn("pin is 36.2.0", str(caught.exception))

    def test_installed_packages_must_match_and_full_stack_must_be_present(self) -> None:
        verify_worker(probe=_probe("cpu", torch="2.5.1+cpu", ort="1.20.1"))
        with self.assertRaises(WorkerImageError):
            verify_worker(probe=_probe("cpu", torch="2.6.0"))
        with self.assertRaises(WorkerImageError) as caught:
            verify_worker(probe=_probe("cpu", require_stack=True))
        self.assertIn("torch 2.5.1 is not installed", str(caught.exception))
        verify_worker(
            probe=_probe(
                "cpu",
                require_stack=True,
                torch="2.5.1",
                ort="1.20.1",
            )
        )
        with self.assertRaises(WorkerImageError) as caught:
            verify_worker(probe=_probe("cpu", tensorrt="8.6.2"))
        self.assertIn("pins none", str(caught.exception))
        verify_worker(
            probe=_probe(
                "jetson",
                require_stack=True,
                torch="2.5.1",
                ort="1.20.1",
            )
        )
        with self.assertRaises(WorkerImageError) as caught:
            verify_worker(
                probe=_probe("cuda", require_stack=True, torch="2.5.1", ort="1.20.1")
            )
        self.assertIn("TensorRT 10.0.1 is not installed", str(caught.exception))
        verify_worker(
            probe=_probe(
                "cuda",
                require_stack=True,
                torch="2.5.1+cu124",
                ort="1.20.1",
                tensorrt="10.0.1.26",
            )
        )

    def test_unknown_image_and_variant_mismatch(self) -> None:
        with self.assertRaises(WorkerImageError):
            verify_worker(probe=_probe("cpu", image=None))
        with self.assertRaises(WorkerImageError):
            verify_worker(probe=_probe("cpu", image="ohho-serve-worker:cpu-9.9.9"))
        with self.assertRaises(WorkerImageError) as caught:
            verify_worker(probe=_probe("cpu"), variant="cuda")
        self.assertIn("not cuda", str(caught.exception))

    def test_version_prefix_match(self) -> None:
        self.assertTrue(versions_match(None, None))
        self.assertTrue(versions_match("2.5.1+cpu", "2.5.1"))
        self.assertTrue(versions_match("10.0.1.26", "10.0.1"))
        self.assertFalse(versions_match("2.5", "2.5.1"))
        self.assertFalse(versions_match("2.6.0", "2.5.1"))

    def test_stamp_round_trip(self) -> None:
        row = get_variant("cuda")
        env = {
            "OHHO_WORKER_VARIANT": row.variant,
            "OHHO_WORKER_IMAGE": row.tag,
            "OHHO_WORKER_STACK": "platform",
            "OHHO_WORKER_TORCH": row.torch,
            "OHHO_WORKER_ORT": row.ort,
            "OHHO_WORKER_TENSORRT": row.tensorrt or "",
            "OHHO_WORKER_CUDA": row.cuda or "",
            "OHHO_WORKER_JETPACK": row.jetpack or "",
            "OHHO_WORKER_L4T": row.l4t or "",
            "OHHO_WORKER_BASE": row.base_image,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "worker-image.json"
            write_image_stamp(path, env)
            document = read_stamp(path)
            self.assertEqual(document["tag"], row.tag)
            self.assertFalse(document["require_stack"])
            probe = collect_probe(
                environ={"OHHO_WORKER_IMAGE": row.tag, "CUDA_VERSION": "12.4.1"},
                arch="x86_64",
                l4t=None,
                packages={"torch": None, "ort": None, "tensorrt": None},
                stamp_path=path,
            )
            report = verify_worker(probe=probe)
            self.assertEqual(report.tag, row.tag)
        bad = dict(env)
        bad["OHHO_WORKER_IMAGE"] = "ohho-serve-worker:cuda-9.9.9"
        with self.assertRaises(WorkerImageError):
            write_image_stamp(Path(tempfile.gettempdir()) / "unused-stamp.json", bad)

    def test_live_probe_does_not_import_torch(self) -> None:
        probe = collect_probe(environ={}, arch="x86_64", l4t=None, packages={})
        self.assertEqual(probe.arch, "x86_64")
        self.assertIsNone(probe.image)


class PolicyTest(unittest.TestCase):
    def _bot(self) -> Robot:
        spec = get_spec("sim")
        transport = SimTransport(spec)
        transport.connect()
        return Robot(spec, transport, NativeRuntime())

    def test_unset_image_runs_and_a_pinned_tag_is_allowed(self) -> None:
        brain = _FlagBrain()
        with _env():
            log = Agent(self._bot(), brain=brain).run("look")
        self.assertTrue(brain.called)
        self.assertTrue(any(line.startswith("done") for line in log))
        self.assertIsNone(refusal_for_environment({}))
        allowed = default_allowlist()
        for row in variants():
            self.assertIsNone(refusal_reason(row.tag, allowed))

    def test_agent_refuses_an_unpinned_or_omitted_image_without_acting(self) -> None:
        brain = _FlagBrain()
        with _env(OHHO_WORKER_IMAGE="ohho-serve-worker:cpu-9.9.9"):
            log = Agent(self._bot(), brain=brain).run("look")
        self.assertFalse(brain.called)
        self.assertIn("refused:", log[1])
        self.assertIn("not a pinned serve worker image", log[1])
        cpu = get_variant("cpu").tag
        cuda = get_variant("cuda").tag
        allow = allowlist_from_json(
            json.dumps({"schema": "ohho.worker_allowlist.v1", "images": [{"tag": cpu}]})
        )
        self.assertIsNone(refusal_reason(cpu, allow))
        reason = refusal_reason(cuda, allow)
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("not allowed by the console", reason)
        brain = _FlagBrain()
        with _env(
            OHHO_WORKER_IMAGE=cuda,
            OHHO_WORKER_ALLOWLIST=json.dumps(
                {"schema": "ohho.worker_allowlist.v1", "images": [cpu]}
            ),
        ):
            log = Agent(self._bot(), brain=brain).run("look")
        self.assertFalse(brain.called)
        self.assertIn("not allowed by the console", log[1])

    def test_digest_must_match_when_the_console_pins_one(self) -> None:
        tag = get_variant("cuda").tag
        payload = {
            "schema": "ohho.worker_allowlist.v1",
            "images": [{"tag": tag, "digest": _DIGEST}],
        }
        allow = allowlist_from_json(json.dumps(payload))
        self.assertIn("missing the digest", refusal_reason(tag, allow) or "")
        self.assertIsNone(refusal_reason(tag, allow, digest=_DIGEST))
        mismatch = refusal_reason(tag, allow, digest=_OTHER_DIGEST)
        self.assertIn("does not match the console allowlist", mismatch or "")
        brain = _FlagBrain()
        with _env(OHHO_WORKER_IMAGE=tag, OHHO_WORKER_IMAGE_DIGEST=_DIGEST):
            log = Agent(self._bot(), brain=brain).run("look")
        self.assertTrue(brain.called)
        self.assertEqual(log[0], f"worker_image: {tag}")
        self.assertEqual(log[1], f"image_digest: {_DIGEST}")

    def test_bad_allowlist_refuses_every_image(self) -> None:
        with self.assertRaises(WorkerImageError):
            allowlist_from_json(
                json.dumps(
                    {
                        "schema": "ohho.worker_allowlist.v1",
                        "images": ["ohho-serve-worker:cpu-9.9.9"],
                    }
                )
            )
        with _env(OHHO_WORKER_IMAGE=get_variant("cpu").tag, OHHO_WORKER_ALLOWLIST="{}"):
            reason = refusal_for_environment()
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("unreadable", reason)


class JobResultTest(unittest.TestCase):
    def test_result_records_the_digest_and_pins(self) -> None:
        row = get_variant("jetson")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "worker-job-result.json"
            document = write_job_result(
                path,
                job_id="run-1-jetson",
                status="verified",
                image=row.tag,
                image_digest=_DIGEST,
            )
            self.assertEqual(document["schema"], RESULT_SCHEMA)
            self.assertEqual(document["image_digest"], _DIGEST)
            self.assertEqual(document["image"], row.tag)
            self.assertEqual(document["jetpack"], row.jetpack)
            self.assertEqual(document["l4t"], row.l4t)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["image_digest"], _DIGEST)
        with self.assertRaises(WorkerImageError):
            build_job_result(
                job_id="run",
                status="verified",
                image=row.tag,
                image_digest="nope",
            )
        with self.assertRaises(WorkerImageError):
            build_job_result(
                job_id="run",
                status="verified",
                image="python:3.12",
                image_digest=_DIGEST,
            )


class ScanTest(unittest.TestCase):
    def test_gate_fails_only_fixed_critical_findings(self) -> None:
        fixed = {
            "matches": [
                {
                    "vulnerability": {
                        "id": "CVE-2024-0001",
                        "severity": "Critical",
                        "fix": {"state": "fixed", "versions": ["2"]},
                    },
                    "artifact": {"name": "openssl", "version": "1"},
                }
            ]
        }
        unfixed = {
            "matches": [
                {
                    "vulnerability": {
                        "id": "CVE-2024-0002",
                        "severity": "Critical",
                        "fix": {"state": "not-fixed", "versions": []},
                    },
                    "artifact": {"name": "libc", "version": "1"},
                }
            ]
        }
        self.assertEqual(
            critical_failures(fixed),
            ["CVE-2024-0001: openssl@1"],
        )
        self.assertEqual(critical_failures(unfixed), [])
        self.assertEqual(critical_failures(fixed, frozenset({"CVE-2024-0001"})), [])
        self.assertEqual(critical_failures(fixed, types=frozenset({"python"})), [])
        python_fixed = {
            "matches": [
                {
                    "vulnerability": fixed["matches"][0]["vulnerability"],
                    "artifact": {"name": "torch", "version": "2.5.1", "type": "python"},
                }
            ]
        }
        self.assertEqual(
            critical_failures(python_fixed, types=frozenset({"python"})),
            ["CVE-2024-0001: torch@2.5.1"],
        )
        with self.assertRaises(WorkerImageError):
            critical_failures({"results": []})
        ignored = load_exceptions(
            ROOT / "docker" / "serve-worker" / "scan-exceptions.json"
        )
        self.assertEqual(ignored, frozenset())

    def test_gate_script_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "clean.json"
            clean.write_text('{"matches": []}\n', encoding="utf-8")
            bad = Path(tmp) / "bad.json"
            bad.write_text("{}\n", encoding="utf-8")
            script = str(ROOT / "scripts" / "worker_vuln_gate.py")
            ok = subprocess.run(
                [sys.executable, script, str(clean)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(ok.returncode, 0, ok.stderr)
            failed = subprocess.run(
                [sys.executable, script, str(bad)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(failed.returncode, 2)


class CliTest(unittest.TestCase):
    def test_verify_exit_codes(self) -> None:
        with _env():
            code, out, err = _run(["targets", "verify", "--json"])
        self.assertEqual(code, 2)
        self.assertIn("no worker image", err)
        self.assertIn('"ok": false', out)
        probe = _probe("cpu", digest=_DIGEST)
        with patch("ohho.worker_images.verify.collect_probe", return_value=probe):
            code, out, err = _run(["targets", "verify", "--json"])
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual(payload["image"], get_variant("cpu").tag)
        self.assertEqual(payload["image_digest"], _DIGEST)

    def test_agent_and_serve_refuse_before_work(self) -> None:
        with _env(OHHO_WORKER_IMAGE="ohho-serve-worker:tpu-1.2.0"):
            code, _out, err = _run(["agent", "sim", "look"])
            serve_code, _serve_out, serve_err = _run(["serve", "--mock"])
        self.assertEqual(code, 2)
        self.assertIn("refused:", err)
        self.assertIn("not a pinned", err)
        self.assertEqual(serve_code, 2)
        self.assertIn("refused:", serve_err)

    def test_record_script_writes_the_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "worker-job-result.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "record_worker_job.py"),
                    "--job-id",
                    "local-1",
                    "--status",
                    "verified",
                    "--image",
                    get_variant("cpu").tag,
                    "--digest",
                    _DIGEST,
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            document = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(document["image_digest"], _DIGEST)


if __name__ == "__main__":
    unittest.main()
