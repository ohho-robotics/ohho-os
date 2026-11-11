"""Tiny Action Chunking Transformer (ACT) model for CPU training and inference.

Implements the CVAE + Transformer architecture (Zhao et al., 2023) in pure
PyTorch without heavy external robotics dependencies. Suitable for fast CPU
fine-tuning on simulated and real demonstration datasets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..data.reader import DatasetReader
from ..metrics import log, open_metrics, summary_path
from ..model_safety import (
    UnsafeModelFile,
    ensure_torch_weights_allowed,
    managed_worker_mode,
)
from .determinism import (
    dataloader_kwargs,
    deterministic_algorithms,
    reseed_runtime,
)
from .il.progress import ValTracker, validation_due
from .precision import PrecisionSession, capabilities_for, require_precision

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TinyACTPolicy(nn.Module):
        """Tiny Action Chunking Transformer policy.

        Architecture:
          - State encoder: linear projection state_dim -> d_model
          - Action encoder: linear projection action_dim -> d_model
          - CVAE encoder: TransformerEncoder over [CLS, state, a_1..a_H] -> latent mu, logvar
          - Transformer decoder: learned action queries conditioned on state + latent z
          - Output: predicted action chunk (B, chunk_size, action_dim)
        """

        def __init__(
            self,
            state_dim: int = 9,
            action_dim: int = 9,
            chunk_size: int = 10,
            d_model: int = 64,
            nhead: int = 2,
            num_encoder_layers: int = 1,
            num_decoder_layers: int = 1,
            dim_feedforward: int = 128,
            latent_dim: int = 16,
            kl_weight: float = 10.0,
            nested_tensor: bool = True,
        ) -> None:
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.chunk_size = chunk_size
            self.d_model = d_model
            self.latent_dim = latent_dim
            self.kl_weight = kl_weight

            self.state_proj = nn.Linear(state_dim, d_model)
            self.action_proj = nn.Linear(action_dim, d_model)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            )
            self.encoder = _encoder(encoder_layer, num_encoder_layers, nested_tensor)
            self.mu_head = nn.Linear(d_model, latent_dim)
            self.logvar_head = nn.Linear(d_model, latent_dim)

            self.latent_proj = nn.Linear(latent_dim, d_model)

            self.action_queries = nn.Embedding(chunk_size, d_model)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=num_decoder_layers
            )
            self.action_head = nn.Linear(d_model, action_dim)

        def forward(
            self,
            state: torch.Tensor,
            actions: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            B = state.shape[0]
            state_embed = self.state_proj(state).unsqueeze(1)

            if actions is not None:
                act_embed = self.action_proj(actions)
                cls_tokens = self.cls_token.expand(B, -1, -1)
                enc_input = torch.cat([cls_tokens, state_embed, act_embed], dim=1)
                enc_out = self.encoder(enc_input)
                cls_out = enc_out[:, 0]
                mu = self.mu_head(cls_out)
                logvar = self.logvar_head(cls_out)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                mu = None
                logvar = None
                z = torch.zeros(B, self.latent_dim, device=state.device)

            z_embed = self.latent_proj(z).unsqueeze(1)
            memory = torch.cat([state_embed, z_embed], dim=1)

            queries = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)
            dec_out = self.decoder(queries, memory)
            pred_actions = self.action_head(dec_out)
            return pred_actions, mu, logvar

        def compute_loss(
            self,
            pred_actions: torch.Tensor,
            target_actions: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            l1_loss = F.l1_loss(pred_actions, target_actions)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = l1_loss + self.kl_weight * kl_loss
            return total_loss, l1_loss, kl_loss

        @torch.no_grad()
        def select_action(self, state: Any) -> List[float]:
            self.eval()
            if not isinstance(state, torch.Tensor):
                state_t = torch.tensor([state], dtype=torch.float32)
            else:
                state_t = state if state.dim() == 2 else state.unsqueeze(0)
            device = next(self.parameters()).device
            state_t = state_t.to(device)
            pred_actions, _, _ = self.forward(state_t)
            return [float(x) for x in pred_actions[0, 0].cpu().tolist()]

else:

    class TinyACTPolicy:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("TinyACTPolicy requires torch: pip install torch")


def train_act(
    dataset: str,
    output_dir: str = "./checkpoints/act",
    *,
    device: str = "cpu",
    train_steps: int = 200,
    batch_size: int = 8,
    lr: float = 1e-4,
    chunk_size: int = 10,
    d_model: int = 64,
    nhead: int = 2,
    num_encoder_layers: int = 1,
    num_decoder_layers: int = 1,
    dim_feedforward: int = 128,
    latent_dim: int = 16,
    kl_weight: float = 10.0,
    precision: str = "fp32",
    mock: bool = False,
    log_every: int = 25,
    pruning: Optional[Dict[str, Any]] = None,
    val_every: int | None = None,
    val_fraction: float = 0.2,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    vision_backbone: str = "none",
    on_event: Any = None,
    deterministic: bool = False,
    seed: int | None = None,
    **kwargs: Any,
) -> str:
    """Train a TinyACTPolicy on a LeRobot v2.0 dataset.

    ``precision`` is ``fp32``, ``fp16`` (CUDA GradScaler), ``bf16``, or
    ``fp8`` (Transformer Engine). The check runs before the step loop.
    ``qat``, when passed, runs quantisation-aware calibration and fine-tune
    after that loop. The saved ``policy.pt`` stays fp32 master weights.
    """
    qat_block = kwargs.pop("qat", None)
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if mock and pruning is not None:
        from .recipe_schema import RecipeConfigError

        raise RecipeConfigError(
            "pruning does not run when mock=True; the mock checkpoint has no weights"
        )

    reader = DatasetReader(dataset)
    state_dim = reader.state_dim or 9
    action_dim = reader.action_dim or 9

    if not mock and not TORCH_AVAILABLE:
        from . import TrainUnavailable

        raise TrainUnavailable(
            "Training ACT policy requires torch: pip install 'ohho-os[train]'. "
            "Use mock=True for the dependency-free sim loop."
        )

    with open_metrics(out):
        if mock:
            loss_curve = [
                (0, 0.8421),
                (50, 0.4312),
                (100, 0.1854),
                (150, 0.0891),
                (train_steps, 0.0412),
            ]
            for step, loss in loss_curve:
                if step < train_steps:
                    log(int(step), {"loss": float(loss)})
            log(int(train_steps), {"loss": float(loss_curve[-1][1])})
            config = {
                "policy": "act",
                "mock": True,
                "device": device,
                "precision": precision,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "chunk_size": chunk_size,
                "train_steps": train_steps,
            }
            if qat_block is not None:
                config["qat"] = _qat_report_without_running(qat_block)
            with open(out / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            summary_file = summary_path(out)
            mock_metrics: dict[str, Any] = {}
            if summary_file.is_file():
                try:
                    loaded = json.loads(summary_file.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        mock_metrics.update(loaded)
                except Exception:
                    pass
            mock_metrics.update(
                {
                    "loss_curve": loss_curve,
                    "final_loss": loss_curve[-1][1],
                }
            )
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(mock_metrics, f, indent=2)
            with open(out / "checkpoint.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            return str(out)

        if device == "auto":
            from ohho.hardware import resolve_device

            device = resolve_device("auto")
        plan = require_precision(precision, capabilities_for(device, precision))

        all_states: List[List[float]] = []
        all_action_chunks: List[List[List[float]]] = []
        for _ep_meta, frames in reader.iter_episodes():
            if not frames:
                continue
            for i in range(len(frames)):
                s = frames[i].observation_state
                chunk: List[List[float]] = []
                for k in range(chunk_size):
                    idx = min(i + k, len(frames) - 1)
                    chunk.append(frames[idx].action)
                all_states.append(s)
                all_action_chunks.append(chunk)

        if not all_states:
            raise ValueError(f"no recorded frames found in dataset at {dataset}")

        tensor_s = torch.tensor(all_states, dtype=torch.float32)
        tensor_a = torch.tensor(all_action_chunks, dtype=torch.float32)
        with deterministic_algorithms(torch, enabled=deterministic):
            return _train_act_body(
                out,
                tensor_s,
                tensor_a,
                state_dim=state_dim,
                action_dim=action_dim,
                device=device,
                plan=plan,
                qat_block=qat_block,
                pruning=pruning,
                train_steps=train_steps,
                batch_size=batch_size,
                lr=lr,
                chunk_size=chunk_size,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                latent_dim=latent_dim,
                kl_weight=kl_weight,
                log_every=log_every,
                val_every=val_every,
                val_fraction=val_fraction,
                early_stop_patience=early_stop_patience,
                early_stop_min_delta=early_stop_min_delta,
                vision_backbone=vision_backbone,
                on_event=on_event,
                deterministic=deterministic,
                seed=seed,
            )


def _train_act_body(
    out: Path,
    tensor_s: Any,
    tensor_a: Any,
    *,
    state_dim: int,
    action_dim: int,
    device: str,
    plan: Any,
    qat_block: Any,
    pruning: Optional[Dict[str, Any]],
    train_steps: int,
    batch_size: int,
    lr: float,
    chunk_size: int,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    latent_dim: int,
    kl_weight: float,
    log_every: int,
    val_every: int | None,
    val_fraction: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
    vision_backbone: str,
    on_event: Any,
    deterministic: bool,
    seed: int | None,
) -> str:
    """Fit one ACT run. Caller holds the deterministic-algorithm scope."""
    train_s, train_a, val_s, val_a = _split_act_tensors(
        tensor_s, tensor_a, val_every, val_fraction
    )
    loader = DataLoader(
        TensorDataset(train_s, train_a),
        batch_size=batch_size,
        **dataloader_kwargs(
            torch, deterministic=deterministic, seed=seed, shuffle=True
        ),
    )
    val_loader = DataLoader(
        TensorDataset(val_s, val_a),
        batch_size=batch_size,
        **dataloader_kwargs(
            torch, deterministic=deterministic, seed=seed, shuffle=False
        ),
    )
    if deterministic and seed is not None:
        reseed_runtime(torch, seed)

    dev = torch.device(device)
    model = TinyACTPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        latent_dim=latent_dim,
        kl_weight=kl_weight,
        nested_tensor=not deterministic,
    )
    model.to(dev)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    prune = None
    if pruning is not None:
        from .compression.pruning import PruningSession

        prune = PruningSession(
            model,
            pruning,
            train_steps=train_steps,
            export_dir=out,
        )
        train_steps = prune.optimizer_steps
    session = PrecisionSession(plan)
    session.begin()

    loss_curve: List[Tuple[int, float]] = []
    validation_curve: List[Tuple[int, float]] = []
    tracker = ValTracker(early_stop_patience, early_stop_min_delta)
    step = 0
    stopped_early = False
    last_val: float | None = None
    while step < train_steps and not stopped_early:
        for batch_s, batch_a in loader:
            if prune is not None:
                prune.before_step(step)
            batch_s = batch_s.to(dev)
            batch_a = batch_a.to(dev)

            optimizer.zero_grad()
            with session.autocast():
                pred_a, mu, logvar = model(batch_s, batch_a)
                assert mu is not None and logvar is not None
                total_loss, _l1, _kl = model.compute_loss(pred_a, batch_a, mu, logvar)
            session.backward(total_loss)
            session.optimizer_step(optimizer)
            if prune is not None:
                prune.after_step(step)

            if step % log_every == 0 or step == train_steps - 1:
                loss_val = round(float(total_loss.detach().float().item()), 4)
                loss_curve.append((step, loss_val))
                log(step, {"loss": loss_val})

            step += 1
            if val_every is not None and validation_due(
                step, train_steps, int(val_every)
            ):
                vloss = _act_validation_loss(model, val_loader, dev)
                if not math.isfinite(vloss):
                    raise ValueError(f"validation loss is not finite at step {step}")
                last_val = vloss
                stop = tracker.observe(vloss)
                validation_curve.append((step, vloss))
                if on_event is not None:
                    on_event(
                        {
                            "event": "validation",
                            "step": step,
                            "validation_loss": vloss,
                            "best_validation_loss": tracker.best,
                            "train_loss": round(float(total_loss.item()), 6),
                        }
                    )
                if stop:
                    stopped_early = True
                    if on_event is not None:
                        on_event(
                            {
                                "event": "early_stop",
                                "step": step,
                                "validation_loss": vloss,
                                "best_validation_loss": tracker.best,
                                "patience": early_stop_patience,
                            }
                        )
                    break
            if step >= train_steps:
                break
    stats = session.end(step)
    report = prune.finish() if prune is not None else None
    qat_report = _run_qat_phases(
        model,
        qat_block,
        loader=loader,
        device=dev,
        lr=lr,
        example=tensor_s[:1].to(dev),
        output_dir=out,
    )

    completed_steps = step
    torch.save(model.state_dict(), out / "policy.pt")

    config = {
        "policy": "act",
        "mock": False,
        "device": device,
        "precision": plan.precision,
        "loss_scaling": plan.loss_scaling,
        "precision_backend": plan.backend,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "chunk_size": chunk_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "dim_feedforward": dim_feedforward,
        "latent_dim": latent_dim,
        "kl_weight": kl_weight,
        "train_steps": train_steps,
        "completed_steps": completed_steps,
        "stopped_early": stopped_early,
        "vision_backbone": vision_backbone,
        "deterministic": bool(deterministic),
        "val_every": val_every,
    }
    if report is not None:
        config["pruning"] = {
            "method": report["method"],
            "schedule": report["schedule"],
            "target_sparsity": report["target_sparsity"],
            "achieved_sparsity": report["achieved_sparsity"],
        }
    if qat_report is not None:
        config["qat"] = qat_report
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    metrics: dict[str, Any] = {
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1][1] if loss_curve else 0.0,
        "train_steps": train_steps,
        "completed_steps": completed_steps,
        "stopped_early": stopped_early,
        "precision": plan.precision,
        "loss_scaling": plan.loss_scaling,
        "precision_backend": plan.backend,
        "throughput_steps_per_s": stats["throughput_steps_per_s"],
        "peak_memory_bytes": stats["peak_memory_bytes"],
    }
    if report is not None:
        metrics["pruning"] = report
    if qat_report is not None:
        metrics["qat"] = qat_report
    if last_val is not None:
        metrics["validation_loss"] = last_val
        metrics["validation_curve"] = validation_curve
    summary_file = summary_path(out)
    merged_metrics: dict[str, Any] = {}
    if summary_file.is_file():
        try:
            loaded = json.loads(summary_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                merged_metrics.update(loaded)
        except Exception:
            pass
    merged_metrics.update(metrics)
    merged_metrics["deterministic"] = bool(deterministic)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(merged_metrics, f, indent=2)

    return str(out)

def _encoder(layer: Any, num_layers: int, nested_tensor: bool) -> Any:
    """Build the ACT encoder. Deterministic runs turn the nested-tensor path off."""
    import torch.nn as nn

    if nested_tensor:
        return nn.TransformerEncoder(layer, num_layers=num_layers)
    try:
        return nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=num_layers)


class _BatchCycle:
    """Walk a dataloader forever. QAT phases are short; the set is small."""

    def __init__(self, loader: Any, device: Any) -> None:
        self.loader = loader
        self.device = device
        self._iter = iter(loader)

    def next(self) -> tuple[Any, Any]:
        try:
            state, action = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            state, action = next(self._iter)
        return state.to(self.device), action.to(self.device)


def _qat_report_without_running(block: Any) -> dict[str, Any]:
    from ohho.compression.qat import spec_from_mapping

    report = spec_from_mapping(block).to_dict()
    report["finetune_loss"] = None
    report["onnx_path"] = None
    report["ran"] = False
    return report


def _run_qat_phases(
    model: Any,
    block: Any,
    *,
    loader: Any,
    device: Any,
    lr: float,
    example: Any,
    output_dir: Path,
) -> dict[str, Any] | None:
    """Calibrate and fine-tune when the recipe passed a ``qat`` block."""
    if block is None:
        return None
    from ohho.compression.qat import apply_qat, spec_from_mapping

    spec = spec_from_mapping(block)
    cycle = _BatchCycle(loader, device)

    def forward() -> None:
        state, action = cycle.next()
        model(state, action)

    def loss_fn() -> Any:
        state, action = cycle.next()
        pred, mu, logvar = model(state, action)
        assert mu is not None and logvar is not None
        total, _l1, _kl = model.compute_loss(pred, action, mu, logvar)
        return total

    export_path = output_dir / "qdq.onnx" if spec.export_onnx else None
    return apply_qat(
        model,
        spec,
        calibration_forwards=forward,
        loss_fn=loss_fn,
        lr=lr,
        export_path=export_path,
        export_example=example,
    )

def _split_act_tensors(
    states: Any, actions: Any, val_every: int | None, val_fraction: float
) -> Tuple[Any, Any, Any, Any]:
    count = int(states.shape[0])
    val_n = 0
    if val_every is not None and count >= 2:
        val_n = int(round(count * float(val_fraction)))
        val_n = min(max(val_n, 1), count - 1)
    if val_n:
        return states[:-val_n], actions[:-val_n], states[-val_n:], actions[-val_n:]
    return states, actions, states, actions


def _act_validation_loss(model: Any, loader: Any, dev: Any) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_s, batch_a in loader:
            batch_s = batch_s.to(dev)
            batch_a = batch_a.to(dev)
            pred_a, mu, logvar = model(batch_s, batch_a)
            loss, _l1, _kl = model.compute_loss(pred_a, batch_a, mu, logvar)
            total += float(loss.item()) * int(batch_s.shape[0])
            count += int(batch_s.shape[0])
    model.train()
    return round(total / max(count, 1), 6)


def load_act_checkpoint(
    checkpoint_dir: str, device: str = "cpu"
) -> Tuple[Optional[TinyACTPolicy], Dict[str, Any]]:
    """Load an ACT model and its config from a checkpoint directory."""
    p = Path(checkpoint_dir).expanduser()
    config_file = p / "config.json"
    if not config_file.exists() and (p.parent / "config.json").exists():
        config_file = p.parent / "config.json"

    config: Dict[str, Any] = {}
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)

    pt_file = p / "policy.pt" if (p / "policy.pt").exists() else p
    if pt_file.is_file() and pt_file.suffix.lower() in {".pt", ".pth", ".bin"}:
        ensure_torch_weights_allowed(pt_file)
    if not TORCH_AVAILABLE or not pt_file.is_file():
        return None, config

    model = TinyACTPolicy(
        state_dim=config.get("state_dim", 9),
        action_dim=config.get("action_dim", 9),
        chunk_size=config.get("chunk_size", 10),
        d_model=config.get("d_model", 64),
        nhead=config.get("nhead", 2),
        num_encoder_layers=config.get("num_encoder_layers", 1),
        num_decoder_layers=config.get("num_decoder_layers", 1),
        dim_feedforward=config.get("dim_feedforward", 128),
        latent_dim=config.get("latent_dim", 16),
        kl_weight=config.get("kl_weight", 10.0),
    )
    dev = torch.device(device)
    try:
        state_dict = torch.load(pt_file, map_location=dev, weights_only=True)
    except Exception as exc:
        if managed_worker_mode():
            detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            raise UnsafeModelFile(
                "managed workers refuse this checkpoint because "
                "torch.load(weights_only=True) failed: " + detail[:200]
            ) from exc
        raise
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()
    return model, config
