"""Policy serving — expose a trained model behind a REST endpoint.

Wraps ``packages/vla_serve`` (FastAPI + pluggable model backend) behind a simple
``serve()`` function and a ``build_app()`` that's testable without launching a
server. When the ``[serve]`` extra (fastapi + uvicorn) is installed, ``serve()``
launches a full inference server; without it, ``build_app()`` still returns a
FastAPI app for testing.

The server is model-agnostic: set ``OHHO_MODEL_CLASS`` to any ``VLAModel``
subclass and it loads on startup (``OHHO_AUTO_LOAD=1``) or on first request.
"""

from __future__ import annotations

from typing import Any, Optional

from ..hardware import resolve_device
from .runner import BASE_ACTION_SCHEMA_VERSION


class ServeUnavailable(RuntimeError):
    """Raised when the serving backend (fastapi/uvicorn) isn't installed."""


# Pydantic models defined at module level so FastAPI's response_model can resolve them.
try:
    from pydantic import BaseModel

    class InferenceRequest(BaseModel):
        instruction: str
        image_base64: str = ""
        config: dict = {}

    class InferenceResponse(BaseModel):
        action: dict
        schema_version: int = 0
        raw_output: str = ""
        latency_ms: float = 0.0

    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False


def build_app(
    model_class: str = "",
    model_path: str = "",
    device: str = "auto",
    api_key: str = "",
    auto_load: bool = False,
    mock_model: bool = False,
    robot_id: str = "",
):
    """Build a FastAPI inference app (testable without launching).

    Parameters:
        model_class: dotted path to a ``VLAModel`` subclass
            (default: ``vla_serve.models.openvla.OpenVLAModel``).
        model_path: HF hub id or local checkpoint path.
        device: ``"auto"`` or explicit.
        api_key: if non-empty, requires ``X-API-Key`` header.
        auto_load: load the model on startup vs. lazy via ``/load_model``.
        mock_model: if True, inject a no-op model that returns zero actions
            (for the record→train→serve sim loop, no GPU needed).
        robot_id: when set to a base-only robot, ``/predict``
            uses the base-velocity action schema (see ``BASE_ACTION_SCHEMA_VERSION``).

    Returns:
        A FastAPI ``FastAPI`` application instance.
    """
    try:
        from fastapi import FastAPI, Header, HTTPException
    except ImportError as e:
        raise ServeUnavailable(
            f"Building the serve app needs fastapi: pip install 'ohho-os[serve]'. ({e})"
        ) from e

    if not _PYDANTIC_AVAILABLE:
        raise ServeUnavailable(
            "Building the serve app needs pydantic: pip install 'ohho-os[serve]'."
        )

    app = FastAPI(title="OhhO Serve", version="0.1.0")
    dev = resolve_device(device)
    _model: Any = None
    _model_path = model_path
    _robot_id = robot_id
    _use_base_action = _base_action_contract(robot_id, model_class)
    app.state.model = None

    def _check_auth(x_api_key: Optional[str] = Header(None)):
        if api_key and x_api_key != api_key:
            raise HTTPException(status_code=401, detail="invalid API key")

    if not model_class and model_path:
        if _is_act_checkpoint(model_path):
            model_class = "act"

    if mock_model:
        _model = _make_mock_model(dev, model_class, robot_id)
        if model_path:
            _model.load_model(model_path)
    elif auto_load and model_path:
        cls = _resolve_model_class(model_class)
        try:
            _model = cls(device=dev)
        except TypeError:
            _model = cls()
        try:
            _model.load_model(model_path, device=dev)
        except TypeError:
            _model.load_model(model_path)
    app.state.model = _model

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": _model is not None, "device": dev}

    @app.post("/load_model")
    async def load_model(
        model_path: str = "",
        model_class: str = "",
        mock: bool = False,
        robot_id: str = "",
        x_api_key: Optional[str] = Header(None),
    ):
        _check_auth(x_api_key)
        nonlocal _model, _model_path, _robot_id, _use_base_action
        cls_name = model_class or ""
        path = model_path or _model_path
        if not path and not mock:
            raise HTTPException(400, "model_path required")
        if path:
            _model_path = path
        if robot_id:
            _robot_id = robot_id
            _use_base_action = _base_action_contract(_robot_id, cls_name)
        elif cls_name:
            _use_base_action = _base_action_contract(_robot_id, cls_name)

        if not cls_name:
            if path and _is_act_checkpoint(path):
                cls_name = "act"
            elif _use_base_action:
                from ..plugins import default_base_model_name

                cls_name = default_base_model_name()

        use_mock = mock_model or mock
        if use_mock:
            _model = _make_mock_model(dev, cls_name, _robot_id)
            _model.load_model(path or "mock")
        else:
            cls = _resolve_model_class(cls_name)
            try:
                _model = cls(device=dev)
            except TypeError:
                _model = cls()
            try:
                _model.load_model(path, device=dev)
            except TypeError:
                _model.load_model(path)
        app.state.model = _model
        from ..plugins import is_contributed_model

        if is_contributed_model(_model):
            _use_base_action = True
        return {
            "status": "loaded",
            "model": type(_model).__name__,
            "schema_version": BASE_ACTION_SCHEMA_VERSION if _use_base_action else 0,
        }

    @app.post("/predict", response_model=InferenceResponse)
    async def predict(
        req: InferenceRequest,
        x_api_key: Optional[str] = Header(None),
    ):
        _check_auth(x_api_key)
        if _model is None:
            raise HTTPException(503, "model not loaded — POST /load_model first")
        import time

        t0 = time.monotonic()
        image = _decode_image(req.image_base64) if req.image_base64 else None
        result = _model.predict_action(image, req.instruction, **req.config)
        latency = (time.monotonic() - t0) * 1000
        if isinstance(result, dict):
            action = result
        else:
            action = {"vector": list(result)}
        schema_version = 0
        if _use_base_action or "base" in action:
            action, schema_version = _normalize_base_action(action)
        return InferenceResponse(
            action=action, schema_version=schema_version, latency_ms=latency
        )

    return app


def serve(
    checkpoint: str = "",
    *,
    port: int = 8000,
    host: str = "0.0.0.0",
    device: str = "auto",
    model_class: str = "",
    api_key: str = "",
    mock: bool = False,
) -> None:
    """Launch the inference server (blocking).

    Parameters:
        checkpoint: path to the trained checkpoint (or HF hub id).
        port, host: bind address.
        device: ``"auto"`` or explicit.
        model_class: dotted path to a ``VLAModel`` subclass.
        api_key: if non-empty, requires ``X-API-Key`` header on mutations.
        mock: if True, use a no-op model (for the sim loop, no GPU needed).
    """
    try:
        import uvicorn
    except ImportError as e:
        raise ServeUnavailable(
            f"Launching the server needs uvicorn: pip install 'ohho-os[serve]'. ({e})"
        ) from e

    app = build_app(
        model_class=model_class,
        model_path=checkpoint,
        device=device,
        api_key=api_key,
        auto_load=bool(checkpoint),
        mock_model=mock,
    )
    uvicorn.run(app, host=host, port=port)


def _is_act_checkpoint(path: str) -> bool:
    """Check if a checkpoint directory or file positively identifies as an OhhO ACT policy.

    A valid ACT checkpoint has a config.json (or checkpoint.json) containing a
    positive ACT marker ('policy': 'act' or 'model_family': 'act'). Plain Hugging
    Face checkpoints containing generic config.json must NOT be treated as ACT.
    """
    if not path:
        return False
    try:
        import json
        from pathlib import Path

        p = Path(path).expanduser()
        if not p.exists():
            return False

        config_file = p / "config.json" if p.is_dir() else p.parent / "config.json"
        if not config_file.exists():
            ckpt_file = (
                p / "checkpoint.json" if p.is_dir() else p.parent / "checkpoint.json"
            )
            if ckpt_file.exists():
                config_file = ckpt_file

        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                policy_name = str(data.get("policy", "")).lower()
                model_family = str(data.get("model_family", "")).lower()
                model_type = str(data.get("model_type", "")).lower()
                if (
                    policy_name == "act"
                    or model_family == "act"
                    or model_type in ("act", "tiny_act")
                ):
                    return True
    except Exception:
        return False
    return False


def _resolve_model_class(dotted: str):
    """Import a dotted class path like 'vla_serve.models.openvla.OpenVLAModel'."""
    if not dotted:
        dotted = "vla_serve.models.openvla.OpenVLAModel"
    if dotted in ("act", "tiny_act", "ACTModel", "ohho.serve.act.ACTModel"):
        from .act import ACTModel

        return ACTModel
    from ..plugins import contributed_model_class

    found = contributed_model_class(dotted)
    if found is not None:
        return found
    parts = dotted.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid model class path: {dotted}")
    module_path, cls_name = parts
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _decode_image(b64: str):
    """Decode a base64 image to a PIL Image (best-effort)."""
    try:
        import base64
        import io

        from PIL import Image  # type: ignore
    except ImportError:
        return None
    if "://" in b64:
        b64 = b64.split(",", 1)[-1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw))


class _MockModel:
    """No-op model for the sim record→train→serve loop (no GPU needed)."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.model = None
        self._path = ""

    def load_model(self, model_path: str, **kwargs) -> None:
        self._path = model_path
        self.model = {"path": model_path, "device": self.device}

    def predict_action(self, image, instruction: str, **kwargs):
        return {"vector": [0.0] * 9, "instruction": instruction, "mock": True}


def _make_mock_model(device: str, model_class: str, robot_id: str):
    if _base_action_contract(robot_id, model_class):
        from ..plugins import default_base_model_name

        cls = _resolve_model_class(model_class or default_base_model_name())
        try:
            m = cls(device=device)
        except TypeError:
            m = cls()
        m.load_model("mock")
        m.config.update({"mock": True, "policy": "mock", "default_vx": 0.2})
        m._mock = True
        return m
    return _MockModel(device)


def _base_action_contract(robot_id: str, model_class: str) -> bool:
    from ..plugins import model_uses_base_action

    if model_class and model_uses_base_action(model_class):
        return True
    if not robot_id:
        return False
    try:
        from ..registry import get_spec

        spec = get_spec(robot_id)
        from .. import capabilities as caps

        return not spec.has(caps.MANIPULATION)
    except Exception:
        return False


def _normalize_base_action(action: dict) -> tuple[dict, int]:
    if "base" in action and isinstance(action["base"], dict):
        base = action["base"]
        out = {
            "base": {
                "vx": float(base.get("vx", 0.0)),
                "vy": float(base.get("vy", 0.0)),
                "wz": float(base.get("wz", 0.0)),
            }
        }
        if "mode" in action:
            out["mode"] = action["mode"]
        return out, BASE_ACTION_SCHEMA_VERSION
    vec = action.get("vector") or []
    if len(vec) >= 3:
        out = {"base": {"vx": float(vec[0]), "vy": float(vec[1]), "wz": float(vec[2])}}
        if action.get("mode"):
            out["mode"] = action["mode"]
        return out, BASE_ACTION_SCHEMA_VERSION
    return action, BASE_ACTION_SCHEMA_VERSION


__all__ = [
    "serve",
    "build_app",
    "ServeUnavailable",
    "BASE_ACTION_SCHEMA_VERSION",
]
