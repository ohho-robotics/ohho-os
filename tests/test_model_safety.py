"""OHH-485: safe model files and the worker-job sandbox spec."""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import stat
import struct
import sys
import tempfile
import types
import unittest
import zipfile
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

from ohho.model_safety import (
    UnsafeModelFile,
    ensure_torch_weights_allowed,
    load_weights,
    prepare_model_load,
    scan_pickle_bytes,
)
from ohho.worker_sandbox import (
    CONSOLE_API_HOST,
    FORBIDDEN_CREDENTIAL_ENV,
    JOB_TOKEN_ENV,
    REGISTRY_STORAGE_HOST,
    EgressTarget,
    JobScratch,
    WorkerSandboxConfig,
    check_egress,
    container_run_spec,
    open_allowed_connection,
)


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def _has_fastapi() -> bool:
    return importlib.util.find_spec("fastapi") is not None


def _has_onnx() -> bool:
    return importlib.util.find_spec("onnx") is not None


def _temp_file(suffix: str) -> Path:
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return Path(name)


def _proto0_call(module: str, name: str) -> bytes:
    """Pickle that would call ``module.name()`` if it were loaded.

    The bytes are assembled here. Nothing imports or calls ``module.name``.
    """
    return b"c" + module.encode("ascii") + b"\n" + name.encode("ascii") + b"\n(tR."


def _write_safetensors(path: Path) -> bytes:
    raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    header = {
        "w": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, len(raw)],
        }
    }
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload += b" " * ((8 - (len(payload) % 8)) % 8)
    path.write_bytes(len(payload).to_bytes(8, "little") + payload + raw)
    return raw


class _FakeSocket:
    def __init__(self) -> None:
        self.connected: tuple[str, int] | None = None
        self.closed = False

    def connect(self, address: tuple[str, int]) -> None:
        self.connected = address

    def close(self) -> None:
        self.closed = True


class TestPickleScanner(unittest.TestCase):
    def test_dangerous_globals_are_refused_and_never_executed(self) -> None:
        fixtures = {
            "os.system": _proto0_call("os", "system"),
            "builtins.exec": _proto0_call("builtins", "exec"),
            "subprocess.Popen": _proto0_call("subprocess", "Popen"),
        }

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("pickle payload was executed")

        with (
            patch("os.system", _boom),
            patch("builtins.exec", _boom),
            patch("builtins.eval", _boom),
            patch("subprocess.Popen", _boom),
            patch("pickle.loads", _boom),
            patch("pickle.Unpickler", _boom),
        ):
            for qual, blob in fixtures.items():
                with self.subTest(qual=qual):
                    path = _temp_file(".pt")
                    self.addCleanup(path.unlink, missing_ok=True)
                    path.write_bytes(blob)
                    with self.assertRaises(UnsafeModelFile) as caught:
                        load_weights(path)
                    self.assertIn(qual, str(caught.exception))
                    self.assertIn("dangerous global", str(caught.exception))

    def test_reduce_fixture_from_dumps_is_not_executed(self) -> None:
        class _Boom:
            def __reduce__(self) -> tuple[object, tuple[str]]:
                import os

                return (os.system, ("OHHO-TEST-MUST-NOT-RUN",))

        blob = pickle.dumps(_Boom(), protocol=4)
        called: list[object] = []

        def _record(*args: object, **_kwargs: object) -> int:
            called.append(args)
            raise AssertionError("os.system was called")

        with (
            patch("os.system", _record),
            patch("pickle.loads", side_effect=AssertionError("pickle.loads")),
        ):
            with self.assertRaises(UnsafeModelFile) as caught:
                scan_pickle_bytes(blob)
        self.assertIn("dangerous global", str(caught.exception))
        self.assertTrue(
            str(caught.exception).endswith("system")
            or ".system" in str(caught.exception)
        )
        self.assertEqual(called, [])

    def test_global_outside_the_allowlist_is_refused(self) -> None:
        blob = _proto0_call("random", "random")
        with self.assertRaises(UnsafeModelFile) as caught:
            scan_pickle_bytes(blob)
        self.assertIn("random.random", str(caught.exception))
        self.assertIn("allowlist", str(caught.exception))

    def test_benign_ordered_dict_passes_the_scan(self) -> None:
        blob = pickle.dumps(OrderedDict([("w", 1.0)]), protocol=4)
        scan_pickle_bytes(blob)

    def test_dummy_bytes_pass_locally_and_fail_when_managed(self) -> None:
        path = _temp_file(".bin")
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_bytes(b"dummy_weights")
        ensure_torch_weights_allowed(path, managed=False)
        with self.assertRaises(UnsafeModelFile) as caught:
            ensure_torch_weights_allowed(path, managed=True)
        self.assertIn("weights-only", str(caught.exception))

    def test_zip_member_escape_is_refused(self) -> None:
        path = _temp_file(".pt")
        self.addCleanup(path.unlink, missing_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("../evil.pkl", _proto0_call("os", "system"))
        with self.assertRaises(UnsafeModelFile) as caught:
            ensure_torch_weights_allowed(path, managed=True)
        self.assertIn("member", str(caught.exception))


class TestSafeLoaders(unittest.TestCase):
    def test_safetensors_roundtrip(self) -> None:
        path = _temp_file(".safetensors")
        self.addCleanup(path.unlink, missing_ok=True)
        raw = _write_safetensors(path)
        loaded = load_weights(path, device="cuda")
        self.assertEqual(loaded.format, "safetensors")
        self.assertEqual(len(loaded.tensors), 1)
        self.assertEqual(loaded.tensors[0].name, "w")
        self.assertEqual(loaded.tensors[0].dtype, "F32")
        self.assertEqual(loaded.tensors[0].shape, (2, 2))
        self.assertEqual(loaded.tensors[0].data, raw)
        if importlib.util.find_spec("torch") is not None:
            import torch

            if torch.cuda.is_available():
                self.assertEqual(loaded.device, "cuda")
            else:
                self.assertEqual(loaded.device, "cpu")
                self.assertIn("cuda", loaded.device_note)
        else:
            self.assertEqual(loaded.device, "cpu")
            self.assertIn("cuda", loaded.device_note)

    def test_invalid_safetensors_header_is_refused(self) -> None:
        path = _temp_file(".safetensors")
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_bytes(b"\xff" * 8)
        with self.assertRaises(UnsafeModelFile) as caught:
            load_weights(path)
        self.assertIn("header length", str(caught.exception))

    def test_empty_safetensors_is_not_a_startup_failure(self) -> None:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        (root / "model.safetensors").touch()
        prepare_model_load(root)
        with self.assertRaises(UnsafeModelFile):
            load_weights(root / "model.safetensors")

    def test_pickle_renamed_to_safetensors_is_refused(self) -> None:
        path = _temp_file(".safetensors")
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_bytes(pickle.dumps(OrderedDict([("w", 1.0)]), protocol=4))
        with self.assertRaises(UnsafeModelFile) as caught:
            load_weights(path)
        self.assertIn("pickle", str(caught.exception))

    @unittest.skipUnless(_has_torch(), "torch not installed")
    def test_benign_weights_only_checkpoint_loads(self) -> None:
        import torch

        path = _temp_file(".pt")
        self.addCleanup(path.unlink, missing_ok=True)
        torch.save({"weight": torch.zeros(2, 3)}, path)
        real_load = torch.load
        seen: list[bool] = []

        def _wrapped(*args: object, **kwargs: object) -> object:
            seen.append(kwargs.get("weights_only") is True)
            return real_load(*args, **kwargs)

        with patch("torch.load", _wrapped):
            loaded = load_weights(path, device="cpu")
        self.assertEqual(seen, [True])
        self.assertEqual(loaded.format, "torch")
        self.assertEqual(tuple(loaded.state_dict["weight"].shape), (2, 3))

    @unittest.skipUnless(_has_torch(), "torch not installed")
    def test_managed_mode_does_not_retry_without_weights_only(self) -> None:
        import torch

        path = _temp_file(".pt")
        self.addCleanup(path.unlink, missing_ok=True)
        torch.save({"weight": torch.ones(1)}, path)
        calls: list[dict[str, object]] = []

        def _fail(*_args: object, **kwargs: object) -> None:
            calls.append(dict(kwargs))
            raise RuntimeError("weights_only rejected the file")

        with patch("torch.load", _fail):
            with self.assertRaises(UnsafeModelFile) as caught:
                load_weights(path, managed=True)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0].get("weights_only"), True)
        self.assertIn("weights_only=True", str(caught.exception))

    def test_act_loader_refuses_a_malicious_checkpoint(self) -> None:
        import ohho.train.act as act_mod
        from ohho.train.act import load_act_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "policy.pt").write_bytes(_proto0_call("builtins", "eval"))
            (root / "config.json").write_text("{}", encoding="utf-8")

            def _boom(*_args: object, **_kwargs: object) -> None:
                raise AssertionError("checkpoint was unpickled")

            # act.py is a module, not a package. Patching the dotted name
            # ohho.train.act.torch.load imports a submodule that does not
            # exist when torch failed to import. Bind the loader the function
            # actually calls, or a stand-in when that name was never bound.
            if act_mod.TORCH_AVAILABLE:
                blocked = patch.object(act_mod.torch, "load", _boom)
            else:
                stand_in = types.ModuleType("torch")
                setattr(stand_in, "load", _boom)
                blocked = patch.object(act_mod, "torch", stand_in, create=True)
            with blocked:
                with self.assertRaises(UnsafeModelFile) as caught:
                    load_act_checkpoint(str(root), device="cpu")
            self.assertIn("builtins.eval", str(caught.exception))

    def test_finetune_refuses_a_local_malicious_checkpoint(self) -> None:
        from ohho.train import finetune

        path = _temp_file(".pt")
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_bytes(_proto0_call("subprocess", "Popen"))
        with self.assertRaises(UnsafeModelFile) as caught:
            finetune("unused", policy="act", checkpoint=str(path), mock=True)
        self.assertIn("subprocess.Popen", str(caught.exception))

    @unittest.skipUnless(_has_fastapi(), "fastapi not installed")
    def test_serve_load_model_returns_the_refusal(self) -> None:
        from fastapi.testclient import TestClient

        from ohho.serve import build_app

        path = _temp_file(".pt")
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_bytes(_proto0_call("os", "system"))
        with self.assertRaises(UnsafeModelFile):
            build_app(model_path=str(path), auto_load=True)
        client = TestClient(build_app())
        response = client.post("/load_model", params={"model_path": str(path)})
        self.assertEqual(response.status_code, 400)
        self.assertIn("os.system", response.text)
        self.assertIn("dangerous global", response.text)

    @unittest.skipUnless(_has_onnx(), "onnx not installed")
    def test_onnx_file_loads_on_cpu(self) -> None:
        import onnx
        from onnx import TensorProto, helper

        path = _temp_file(".onnx")
        self.addCleanup(path.unlink, missing_ok=True)
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            "tiny",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path
        )
        loaded = load_weights(path, device="cuda:0")
        self.assertEqual(loaded.format, "onnx")
        self.assertIsNotNone(loaded.onnx_model)
        if importlib.util.find_spec("torch") is not None:
            import torch

            if not torch.cuda.is_available():
                self.assertEqual(loaded.device, "cpu")
                self.assertIn("cuda", loaded.device_note)

    def test_pickle_renamed_to_onnx_is_refused(self) -> None:
        path = _temp_file(".onnx")
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_bytes(pickle.dumps(OrderedDict([("w", 1)]), protocol=4))
        with self.assertRaises(UnsafeModelFile) as caught:
            load_weights(path)
        self.assertIn("pickle", str(caught.exception).lower())

    def test_managed_worker_refuses_torchscript(self) -> None:
        from ohho.policy_runner import LocomotionPolicy

        path = _temp_file(".pt")
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_bytes(b"not-a-torchscript")
        policy = LocomotionPolicy(str(path))

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("torch.jit.load was called")

        try:
            import torch as torch_mod
        except ImportError:
            torch_mod = None
        if torch_mod is not None:
            blocked = patch.object(torch_mod.jit, "load", _boom)
        else:
            # No torch install: put a stand-in where policy.load() would
            # import it. The managed-worker refusal must still happen first.
            fake_jit = types.ModuleType("torch.jit")
            setattr(fake_jit, "load", _boom)
            fake_torch = types.ModuleType("torch")
            setattr(fake_torch, "jit", fake_jit)
            blocked = patch.dict(
                sys.modules, {"torch": fake_torch, "torch.jit": fake_jit}
            )

        with patch.dict(os.environ, {"OHHO_MANAGED_WORKER": "1"}), blocked:
            with self.assertRaises(UnsafeModelFile) as caught:
                policy.load()
        self.assertIn("TorchScript", str(caught.exception))


class TestWorkerSandbox(unittest.TestCase):
    def test_run_spec_is_isolated_and_has_no_user_credentials(self) -> None:
        spec = container_run_spec()
        self.assertTrue(spec["readOnlyRootFilesystem"])
        self.assertEqual(spec["hostMounts"], [])
        self.assertFalse(spec["privileged"])
        self.assertIn("ALL", spec["capDrop"])
        self.assertTrue(spec["noNewPrivileges"])
        self.assertNotEqual(spec["user"]["uid"], 0)
        self.assertNotEqual(spec["user"]["name"], "root")
        self.assertTrue(spec["user"]["runAsNonRoot"])
        self.assertGreater(spec["resources"]["cpu"], 0)
        self.assertGreater(spec["resources"]["memoryMb"], 0)
        self.assertEqual(spec["resources"]["gpu"]["count"], 0)
        self.assertFalse(spec["resources"]["gpu"]["attached"])
        self.assertTrue(spec["scratch"]["wipeOnExit"])
        self.assertFalse(spec["network"]["hostNetwork"])
        hosts = [item["host"] for item in spec["network"]["egress"]]
        self.assertEqual(hosts, [CONSOLE_API_HOST, REGISTRY_STORAGE_HOST])
        self.assertEqual(spec["credentials"], "short-lived-job-token-only")
        names = [item["name"] for item in spec["env"]]
        self.assertEqual(names, ["OHHO_MANAGED_WORKER", JOB_TOKEN_ENV])
        self.assertEqual(spec["env"][1]["valueFrom"], "runtime-job-token")
        self.assertNotIn("value", spec["env"][1])
        for name in FORBIDDEN_CREDENTIAL_ENV:
            self.assertIn(name, spec["forbiddenEnv"])
        rendered = json.dumps(spec)
        self.assertNotIn("super-secret-token", rendered)

    def test_spec_refuses_external_credentials_without_echoing_them(self) -> None:
        secret = "super-secret-token"
        with self.assertRaises(ValueError) as caught:
            container_run_spec(extra_env={"HF_TOKEN": secret})
        self.assertIn("HF_TOKEN", str(caught.exception))
        self.assertNotIn(secret, str(caught.exception))
        with self.assertRaises(ValueError) as job_token:
            container_run_spec(extra_env={JOB_TOKEN_ENV: secret})
        self.assertIn(JOB_TOKEN_ENV, str(job_token.exception))
        self.assertNotIn(secret, str(job_token.exception))

    def test_egress_allowlist_cannot_be_widened(self) -> None:
        with self.assertRaises(ValueError):
            WorkerSandboxConfig(egress=(EgressTarget("evil.example", (443,), "no"),))
        narrowed = WorkerSandboxConfig(
            egress=(EgressTarget(CONSOLE_API_HOST, (443,), "console API"),)
        )
        spec = container_run_spec(narrowed)
        self.assertEqual(
            [item["host"] for item in spec["network"]["egress"]],
            [CONSOLE_API_HOST],
        )
        with self.assertRaises(ValueError):
            WorkerSandboxConfig(uid=0)
        with self.assertRaises(ValueError):
            WorkerSandboxConfig(host_mounts=("/var/run",))
        with self.assertRaises(ValueError):
            WorkerSandboxConfig(read_only_root=False)

    def test_scratch_dir_is_empty_after_the_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scratch = JobScratch(Path(tmp) / "job")
            with scratch:
                nested = scratch.path / "nested"
                nested.mkdir()
                target = nested / "weights.bin"
                target.write_text("checkpoint", encoding="utf-8")
                target.chmod(stat.S_IREAD)
                self.assertFalse(scratch.is_empty())
            self.assertTrue(scratch.path.is_dir())
            self.assertTrue(scratch.is_empty())
            scratch.verify_empty()

    def test_scratch_symlink_is_unlinked_without_deleting_the_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "keep.txt"
            outside.write_text("keep", encoding="utf-8")
            scratch = JobScratch(root / "job")
            scratch.prepare()
            link = scratch.path / "link"
            try:
                link.symlink_to(outside)
            except OSError:
                self.skipTest("symlinks are not available")
            scratch.wipe()
            self.assertTrue(scratch.is_empty())
            self.assertEqual(outside.read_text(encoding="utf-8"), "keep")

    def test_egress_check_uses_the_fake_resolver_and_socket(self) -> None:
        resolved: list[str] = []

        def _resolve(host: str) -> list[str]:
            resolved.append(host)
            if host != CONSOLE_API_HOST:
                raise AssertionError(host)
            return ["1.2.3.4"]

        sockets: list[_FakeSocket] = []

        def _factory() -> _FakeSocket:
            sock = _FakeSocket()
            sockets.append(sock)
            return sock

        blocked = check_egress("evil.example", 443, resolve=_resolve)
        self.assertFalse(blocked.allowed)
        self.assertEqual(resolved, [])
        decision = open_allowed_connection(
            "evil.example", 443, resolve=_resolve, socket_factory=_factory
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(sockets, [])

        allowed = open_allowed_connection(
            CONSOLE_API_HOST + ".",
            443,
            resolve=_resolve,
            socket_factory=_factory,
        )
        self.assertTrue(allowed.allowed)
        self.assertEqual(sockets[0].connected, ("1.2.3.4", 443))

        def _private(_host: str) -> list[str]:
            return ["10.0.0.8", "1.2.3.4"]

        rebound = open_allowed_connection(
            CONSOLE_API_HOST, 443, resolve=_private, socket_factory=_factory
        )
        self.assertFalse(rebound.allowed)
        self.assertIn("10.0.0.8", rebound.reason)
        self.assertEqual(len(sockets), 1)

        def _metadata(_host: str) -> list[str]:
            return ["169.254.169.254"]

        meta = check_egress(REGISTRY_STORAGE_HOST, 443, resolve=_metadata)
        self.assertFalse(meta.allowed)
        self.assertIn("169.254.169.254", meta.reason)

        def _mapped(_host: str) -> list[str]:
            return ["::ffff:127.0.0.1"]

        loop = check_egress(CONSOLE_API_HOST, 443, resolve=_mapped)
        self.assertFalse(loop.allowed)

        wrong_port = check_egress(CONSOLE_API_HOST, 22, resolve=_resolve)
        self.assertFalse(wrong_port.allowed)
        self.assertEqual(resolved, [CONSOLE_API_HOST])


if __name__ == "__main__":
    unittest.main()
