"""Versioned parity sets (OHH-471).

Capture, the registry hook, and report labelling run without torch. Comparing
a captured set across two ``ohho optimize`` runs skips unless torch, onnx, and
onnxruntime are installed.
"""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho.cli import main
from ohho.optimize.errors import OptimizeError
from ohho.optimize.models import REFERENCE_ACT, ExportInput
from ohho.optimize.plan import OptimizeRequest, build_plan
from ohho.parity.errors import ParityError
from ohho.parity.gate import evaluate
from ohho.parity.inputs import load_observation_frames
from ohho.parity.registry import (
    ParitySetAttachment,
    ParitySetPublisher,
    attach_parity_set,
    resolve_parity_set,
)
from ohho.parity.report import report_problems
from ohho.parity.runners import PreparedModel
from ohho.parity.sets import (
    SET_SCHEMA_ID,
    load_parity_set,
    set_schema_text,
    write_parity_set,
)

ROOT = Path(__file__).resolve().parents[1]


def _has_export_stack() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("torch", "onnx", "onnxruntime")
    )


def _run(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def _write_dataset(root: Path, frames: list[list[float]]) -> None:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "total_episodes": 1,
                "total_frames": len(frames),
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [len(frames[0])],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": len(frames)}) + "\n",
        encoding="utf-8",
    )
    lines = []
    for index, state in enumerate(frames):
        lines.append(
            json.dumps(
                {
                    "observation.state": state,
                    "action": [0.0, 0.0],
                    "timestamp": index / 10,
                    "frame_index": index,
                    "episode_index": 0,
                    "task_index": 0,
                    "next.done": index == len(frames) - 1,
                }
            )
        )
    (data / "episode_000000.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _tiny_set(root: Path, *, outputs: list[list[float]] | None = None) -> object:
    chosen = outputs if outputs is not None else [[0.0, 1.0], [0.5, -0.5]]
    return write_parity_set(
        root,
        set_id="act-golden",
        version=1,
        model="reference-act",
        submodule=None,
        source="dataset",
        source_path="dataset",
        seed=0,
        n_requested=2,
        indices=[0, 1],
        batches={"state": [[0.25, -0.5], [1.0, 0.125]]},
        outputs={"actions": chosen},
    )


def _prepared() -> PreparedModel:
    return PreparedModel(
        ref="reference-act",
        kind="act",
        module=object(),
        onnx_path=None,
        inputs=(ExportInput("state", (1, 2)),),
        outputs=("actions",),
    )


class _RecordingPublisher(ParitySetPublisher):
    def __init__(self) -> None:
        self.seen: list[object] = []

    def publish(self, attachment: ParitySetAttachment) -> str:
        self.seen.append(attachment)
        return f"registry://models/{attachment.model}/parity-sets/{attachment.set_id}"


class TestParitySetFiles(unittest.TestCase):
    def test_schema_is_package_data(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("parity/*.json", text)
        document = json.loads(set_schema_text())
        self.assertEqual(document["properties"]["schema"]["const"], SET_SCHEMA_ID)

    def test_rewrite_keeps_the_same_sha_and_a_tamper_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _tiny_set(root / "a")
            second = _tiny_set(root / "b")
            self.assertEqual(first.sha256, second.sha256)
            loaded = load_parity_set(root / "a")
            self.assertEqual(loaded.sha256, first.sha256)
            self.assertEqual(loaded.inputs["state"][0], [0.25, -0.5])
            changed = _tiny_set(root / "c", outputs=[[0.0, 1.0], [9.0, 9.0]])
            self.assertNotEqual(changed.sha256, first.sha256)
            samples = root / "a" / "samples.json"
            text = samples.read_text(encoding="utf-8")
            self.assertIn("0.25", text)
            samples.write_bytes(text.replace("0.25", "0.26", 1).encode("utf-8"))
            with self.assertRaises(ParityError) as caught:
                load_parity_set(root / "a")
            self.assertIn("sha256", str(caught.exception))

    def test_archive_roundtrip_matches_the_directory_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = _tiny_set(root / "dir")
            archive = _tiny_set(root / "set.tar.gz")
            self.assertEqual(directory.sha256, archive.sha256)
            loaded = load_parity_set(root / "set.tar.gz")
            self.assertEqual(loaded.sha256, directory.sha256)
            self.assertEqual(loaded.n, 2)
            self.assertFalse(loaded.synthetic)

    def test_camera_log_frames_and_media_are_stored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "cam"
            log.mkdir()
            image = log / "000.jpg"
            image.write_bytes(b"\xff\xd8\xff")
            (log / "frames.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"observation.state": [0.1, 0.2], "image": "000.jpg"}
                        ),
                        json.dumps({"observation.state": [-0.3, 0.4]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            frames = load_observation_frames(str(log), n=8, seed=0)
            self.assertEqual(frames.source, "camera")
            self.assertEqual(frames.frames, [[0.1, 0.2], [-0.3, 0.4]])
            self.assertTrue(str(frames.media[0]).endswith("000.jpg"))
            self.assertIsNone(frames.media[1])
            prefixed = load_observation_frames(
                f"camera:{log / 'frames.jsonl'}", n=8, seed=1
            )
            self.assertEqual(prefixed.source, "camera")
            self.assertEqual(prefixed.frames, frames.frames)
            stored = write_parity_set(
                root / "set",
                set_id="act-camera",
                version=1,
                model="reference-act",
                submodule=None,
                source="camera",
                source_path=str(log),
                seed=0,
                n_requested=8,
                indices=list(frames.indices),
                batches={"state": frames.frames},
                outputs={"actions": [[0.0, 0.0], [1.0, 1.0]]},
                media_files=frames.media,
            )
            self.assertEqual(stored.media[0], "media/0000.jpg")
            self.assertIsNone(stored.media[1])
            copied = root / "set" / "media" / "0000.jpg"
            self.assertEqual(copied.read_bytes(), image.read_bytes())
            bare = write_parity_set(
                root / "bare",
                set_id="act-camera-b",
                version=1,
                model="reference-act",
                submodule=None,
                source="camera",
                source_path=str(log),
                seed=0,
                n_requested=8,
                indices=list(frames.indices),
                batches={"state": frames.frames},
                outputs={"actions": [[0.0, 0.0], [1.0, 1.0]]},
            )
            self.assertEqual(stored.sha256, bare.sha256)

    def test_reserved_id_and_random_capture_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ParityError):
                write_parity_set(
                    Path(tmp) / "set",
                    set_id="synthetic",
                    version=1,
                    model="reference-act",
                    submodule=None,
                    source="dataset",
                    source_path=None,
                    seed=0,
                    n_requested=1,
                    indices=[0],
                    batches={"state": [[0.0]]},
                    outputs={"actions": [[0.0]]},
                )
            code, text, err = _run(
                [
                    "parity",
                    "capture",
                    "--reference",
                    "reference-act",
                    "--inputs",
                    "random",
                    "--out",
                    str(Path(tmp) / "out"),
                    "--id",
                    "act-golden",
                    "--json",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("camera", text + err)

    def test_registry_attach_calls_the_publisher_and_resolves_the_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stored = _tiny_set(root / "set")
            publisher = _RecordingPublisher()
            record = attach_parity_set(
                root / "set",
                model_ref="reference-act",
                model_version="v3",
                publisher=publisher,
                root=root / "registry",
            )
            self.assertEqual(len(publisher.seen), 1)
            seen = publisher.seen[0]
            self.assertEqual(seen.sha256, stored.sha256)
            self.assertEqual(seen.model, "reference-act")
            self.assertEqual(seen.model_version, "v3")
            self.assertTrue((Path(seen.path) / "samples.json").is_file())
            self.assertEqual(
                record["registry_uri"],
                "registry://models/reference-act/parity-sets/act-golden",
            )
            by_id = resolve_parity_set("act-golden", root=root / "registry")
            by_version = resolve_parity_set("act-golden@1", root=root / "registry")
            self.assertEqual(by_id.sha256, stored.sha256)
            self.assertEqual(by_version.sha256, stored.sha256)
            again = attach_parity_set(
                root / "set",
                model_ref="reference-act",
                model_version="v3",
                publisher=ParitySetPublisher(),
                root=root / "registry",
            )
            self.assertTrue(again["registry_uri"].startswith("local://act-golden@1"))


class TestGateUsesTheSet(unittest.TestCase):
    def test_stored_outputs_are_the_reference_and_the_result_names_the_set(
        self,
    ) -> None:
        outputs = [[0.0, 1.0], [0.5, -0.5]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stored = _tiny_set(root / "set", outputs=outputs)
            candidate = root / "model.pt"
            candidate.write_bytes(b"not-a-module")
            with (
                patch("ohho.parity.gate.prepare_reference", return_value=_prepared()),
                patch(
                    "ohho.parity.gate.run_prepared",
                    side_effect=AssertionError("reference was re-run"),
                ),
                patch(
                    "ohho.parity.gate.run_candidate", return_value={"actions": outputs}
                ),
            ):
                report = evaluate(
                    "reference-act",
                    str(candidate),
                    inputs="random",
                    n=99,
                    seed=5,
                    parity_set=str(root / "set"),
                )
        self.assertEqual(report["result"], "pass")
        self.assertEqual(report["parity_set"]["id"], "act-golden")
        self.assertEqual(report["parity_set"]["version"], 1)
        self.assertEqual(report["parity_set"]["sha256"], stored.sha256)
        self.assertFalse(report["parity_set"]["synthetic"])
        self.assertEqual(report["parity_set"]["source"], "dataset")
        self.assertEqual(report["inputs"]["source"], "dataset")
        self.assertEqual(report["inputs"]["seed"], 0)
        self.assertEqual(report["inputs"]["n"], 2)
        self.assertEqual(report_problems(report), [])

    def test_a_set_for_another_model_is_refused(self) -> None:
        other = _prepared()
        other = PreparedModel(
            ref="other-model",
            kind=other.kind,
            module=other.module,
            onnx_path=other.onnx_path,
            inputs=other.inputs,
            outputs=other.outputs,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tiny_set(root / "set")
            candidate = root / "model.pt"
            candidate.write_bytes(b"not-a-module")
            with (
                patch("ohho.parity.gate.prepare_reference", return_value=other),
                patch("ohho.parity.gate.run_candidate", return_value={"actions": []}),
            ):
                with self.assertRaises(ParityError) as caught:
                    evaluate(
                        "other-model",
                        str(candidate),
                        inputs="random",
                        n=2,
                        parity_set=str(root / "set"),
                    )
        self.assertIn("reference-act", str(caught.exception))

    def test_missing_set_is_labelled_synthetic_and_the_sha_is_stable(self) -> None:
        outputs = {"actions": [[0.0, 0.0], [0.0, 0.0]]}
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "model.pt"
            candidate.write_bytes(b"not-a-module")
            with (
                patch("ohho.parity.gate.prepare_reference", return_value=_prepared()),
                patch("ohho.parity.gate.run_prepared", return_value=outputs),
                patch("ohho.parity.gate.run_candidate", return_value=outputs),
            ):
                first = evaluate(
                    str(candidate), str(candidate), inputs="random", n=2, seed=0
                )
                second = evaluate(
                    str(candidate), str(candidate), inputs="random", n=2, seed=0
                )
                shifted = evaluate(
                    str(candidate), str(candidate), inputs="random", n=2, seed=1
                )
        for report in (first, second):
            self.assertTrue(report["parity_set"]["synthetic"])
            self.assertEqual(report["parity_set"]["id"], "synthetic")
            self.assertEqual(report["parity_set"]["source"], "synthetic")
            self.assertEqual(len(report["parity_set"]["sha256"]), 64)
            self.assertEqual(report["inputs"]["source"], "random")
        self.assertEqual(first["parity_set"]["sha256"], second["parity_set"]["sha256"])
        self.assertNotEqual(
            first["parity_set"]["sha256"], shifted["parity_set"]["sha256"]
        )

    def test_real_inputs_without_a_set_are_not_synthetic(self) -> None:
        frames = [[0.2, -0.2], [0.4, 0.1]]
        outputs = {"actions": [[0.0, 0.0], [0.0, 0.0]]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset, frames)
            candidate = root / "model.pt"
            candidate.write_bytes(b"not-a-module")
            with (
                patch("ohho.parity.gate.prepare_reference", return_value=_prepared()),
                patch("ohho.parity.gate.run_prepared", return_value=outputs),
                patch("ohho.parity.gate.run_candidate", return_value=outputs),
            ):
                report = evaluate(
                    str(candidate), str(candidate), inputs=str(dataset), n=8, seed=0
                )
        self.assertFalse(report["parity_set"]["synthetic"])
        self.assertEqual(report["parity_set"]["id"], "ephemeral")
        self.assertEqual(report["parity_set"]["source"], "dataset")
        self.assertEqual(len(report["parity_set"]["sha256"]), 64)

    def test_a_report_without_parity_set_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "model.pt"
            candidate.write_bytes(b"not-a-module")
            outputs = {"actions": [[0.0]]}
            prepared = PreparedModel(
                ref="reference-act",
                kind="act",
                module=object(),
                onnx_path=None,
                inputs=(ExportInput("state", (1, 1)),),
                outputs=("actions",),
            )
            with (
                patch("ohho.parity.gate.prepare_reference", return_value=prepared),
                patch("ohho.parity.gate.run_prepared", return_value=outputs),
                patch("ohho.parity.gate.run_candidate", return_value=outputs),
            ):
                report = evaluate(
                    str(candidate), str(candidate), inputs="random", n=1, seed=0
                )
        broken = dict(report)
        broken.pop("parity_set")
        problems = report_problems(broken)
        self.assertTrue(any("parity_set" in item for item in problems))

    def test_help_and_blank_flag(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["parity", "--help"])
        self.assertEqual(caught.exception.code, 0)
        text = out.getvalue()
        self.assertIn("--parity-set", text)
        self.assertIn("capture", text)
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["parity", "capture", "--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--id", out.getvalue())
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["optimize", "--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--parity-set", out.getvalue())
        with self.assertRaises(OptimizeError):
            build_plan(
                OptimizeRequest(
                    model="reference-act",
                    target="x86-cpu",
                    precision="fp32",
                    out_dir="out",
                    parity_set="  ",
                )
            )
        code, text, _err = _run(
            [
                "parity",
                "--reference",
                "reference-act",
                "--candidate",
                "missing.pt",
                "--json",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("--inputs", text)


class TestOptimizeReusesOneSet(unittest.TestCase):
    @unittest.skipUnless(
        _has_export_stack(), "optimize extra (torch, onnx, onnxruntime)"
    )
    def test_capture_from_a_dataset_and_two_optimize_runs_share_the_sha(self) -> None:
        width = int(REFERENCE_ACT["state_dim"])
        frames = [
            [((row + 1) * (col + 1)) / 40 for col in range(width)] for row in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset, frames)
            set_dir = root / "act-golden"
            code, text, err = _run(
                [
                    "parity",
                    "capture",
                    "--reference",
                    "reference-act",
                    "--inputs",
                    str(dataset),
                    "--n",
                    "4",
                    "--seed",
                    "0",
                    "--out",
                    str(set_dir),
                    "--id",
                    "act-golden",
                    "--json",
                ]
            )
            self.assertEqual(code, 0, err + text)
            captured = json.loads(text)
            self.assertEqual(captured["id"], "act-golden")
            self.assertEqual(captured["source"], "dataset")
            self.assertFalse(captured["synthetic"])
            self.assertEqual(len(captured["sha256"]), 64)
            shas = []
            for name in ("run-a", "run-b"):
                out = root / name
                code, text, err = _run(
                    [
                        "optimize",
                        "--model",
                        "reference-act",
                        "--target",
                        "x86-cpu",
                        "--precision",
                        "fp32",
                        "--ort-ep",
                        "cpu",
                        "--parity-set",
                        str(set_dir),
                        "--parity-inputs",
                        "random",
                        "--out",
                        str(out),
                        "--json",
                    ]
                )
                self.assertEqual(code, 0, err + text)
                document = json.loads(text)
                parity_set = document["parity"]["parity_set"]
                self.assertEqual(parity_set["id"], "act-golden")
                self.assertEqual(parity_set["sha256"], captured["sha256"])
                self.assertFalse(parity_set["synthetic"])
                self.assertEqual(document["parity"]["inputs"]["source"], "dataset")
                self.assertEqual(document["parity"]["result"], "pass")
                shas.append(parity_set["sha256"])
            self.assertEqual(shas[0], shas[1])
            synthetic_out = root / "synthetic"
            code, text, err = _run(
                [
                    "optimize",
                    "--model",
                    "reference-act",
                    "--target",
                    "x86-cpu",
                    "--precision",
                    "fp32",
                    "--ort-ep",
                    "cpu",
                    "--parity-n",
                    "4",
                    "--parity-seed",
                    "0",
                    "--out",
                    str(synthetic_out),
                    "--json",
                ]
            )
            self.assertEqual(code, 0, err + text)
            synthetic = json.loads(text)["parity"]["parity_set"]
            self.assertTrue(synthetic["synthetic"])
            self.assertEqual(synthetic["id"], "synthetic")
            self.assertEqual(synthetic["source"], "synthetic")
            self.assertEqual(len(synthetic["sha256"]), 64)
            self.assertNotEqual(synthetic["sha256"], captured["sha256"])


if __name__ == "__main__":
    unittest.main()
