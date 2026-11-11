"""Tiny conditional Diffusion Policy for CPU training and for ``ohho serve``.

The checkpoint directory matches the layout :class:`ohho.serve.act.ACTModel`
already uses for files, with a diffusion marker so serve does not treat it as
ACT. Load it through the dotted class
``ohho.train.il.diffusion_policy.DiffusionPolicyModel``, which ``/load_model``
imports without a serve code change.

``policy.pt`` is this module's ``state_dict`` saved with ``weights_only=True``
on load. ``config.json`` records ``policy`` and ``model_type`` as ``diffusion``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..determinism import (
    dataloader_kwargs,
    deterministic_algorithms,
    reseed_runtime,
)
from .data import load_windows
from .progress import ValTracker, validation_due

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

BETA_START = 1e-4
BETA_END = 0.02


if TORCH_AVAILABLE:

    class TinyDiffusionPolicy(nn.Module):
        """State-conditioned DDPM over an action chunk.

        ``scheduler`` ``ddpm`` denoises every training timestep.
        ``scheduler`` ``ddim`` denoises ``num_inference_steps`` evenly spaced
        timesteps. Training loss is noise-prediction MSE either way.
        """

        def __init__(
            self,
            state_dim: int = 9,
            action_dim: int = 9,
            chunk_size: int = 16,
            diffusion_steps: int = 100,
            d_model: int = 64,
            scheduler: str = "ddpm",
            num_inference_steps: int = 10,
            beta_start: float = BETA_START,
            beta_end: float = BETA_END,
        ) -> None:
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.chunk_size = chunk_size
            self.diffusion_steps = diffusion_steps
            self.d_model = d_model
            self.scheduler = scheduler
            self.num_inference_steps = num_inference_steps
            self.beta_start = beta_start
            self.beta_end = beta_end
            action_flat = action_dim * chunk_size
            self.state_proj = nn.Linear(state_dim, d_model)
            self.action_proj = nn.Linear(action_flat, d_model)
            self.time_embed = nn.Embedding(diffusion_steps, d_model)
            self.net = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.ReLU(),
                nn.Linear(d_model, action_flat),
            )
            betas = torch.linspace(beta_start, beta_end, diffusion_steps)
            alphas = 1.0 - betas
            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))

        def forward(
            self,
            state: torch.Tensor,
            noisy_action: torch.Tensor,
            timesteps: torch.Tensor,
        ) -> torch.Tensor:
            flat = noisy_action.reshape(noisy_action.shape[0], -1)
            hidden = torch.cat(
                [
                    self.state_proj(state),
                    self.action_proj(flat),
                    self.time_embed(timesteps),
                ],
                dim=-1,
            )
            return self.net(hidden).reshape_as(noisy_action)

        def loss(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            batch = state.shape[0]
            timesteps = torch.randint(
                0, self.diffusion_steps, (batch,), device=state.device
            )
            noise = torch.randn_like(actions)
            alpha_bar = self.alpha_bar[timesteps].view(batch, 1, 1)
            noisy = (
                torch.sqrt(alpha_bar) * actions
                + torch.sqrt((1.0 - alpha_bar).clamp(min=0.0)) * noise
            )
            predicted = self.forward(state, noisy, timesteps)
            return F.mse_loss(predicted, noise)

        def sample(self, state: torch.Tensor) -> torch.Tensor:
            """Draw one action chunk per batch row."""
            times = _inference_timesteps(
                self.diffusion_steps, self.scheduler, self.num_inference_steps
            )
            action = torch.randn(
                state.shape[0],
                self.chunk_size,
                self.action_dim,
                device=state.device,
            )
            for index, t_int in enumerate(times):
                step = torch.full(
                    (state.shape[0],), t_int, device=state.device, dtype=torch.long
                )
                predicted = self.forward(state, action, step)
                alpha_bar = self.alpha_bar[t_int].clamp(min=1e-8, max=1.0)
                if self.scheduler == "ddim":
                    previous = times[index + 1] if index + 1 < len(times) else None
                    if previous is None:
                        alpha_bar_prev = action.new_tensor(1.0)
                    else:
                        alpha_bar_prev = self.alpha_bar[previous].clamp(
                            min=1e-8, max=1.0
                        )
                    pred_x0 = (
                        action - torch.sqrt(1.0 - alpha_bar) * predicted
                    ) / torch.sqrt(alpha_bar)
                    action = (
                        torch.sqrt(alpha_bar_prev) * pred_x0
                        + torch.sqrt(1.0 - alpha_bar_prev) * predicted
                    )
                else:
                    beta = self.betas[t_int]
                    alpha = self.alphas[t_int].clamp(min=1e-8)
                    coeff = beta / torch.sqrt(1.0 - alpha_bar)
                    mean = (action - coeff * predicted) / torch.sqrt(alpha)
                    if t_int > 0:
                        action = mean + torch.sqrt(beta) * torch.randn_like(action)
                    else:
                        action = mean
            return action

        @torch.no_grad()
        def select_action(self, state: Any) -> list[float]:
            self.eval()
            if not isinstance(state, torch.Tensor):
                state_t = torch.tensor([list(state)], dtype=torch.float32)
            else:
                state_t = state if state.dim() == 2 else state.unsqueeze(0)
            device = next(self.parameters()).device
            state_t = state_t.to(device)
            actions = self.sample(state_t)
            return [float(item) for item in actions[0, 0].detach().cpu().tolist()]

else:

    class TinyDiffusionPolicy:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "TinyDiffusionPolicy requires torch: pip install 'ohho-os[train]'"
            )


def _inference_timesteps(
    diffusion_steps: int, scheduler: str, num_inference_steps: int
) -> list[int]:
    if scheduler != "ddim":
        return list(range(diffusion_steps - 1, -1, -1))
    count = max(1, min(int(num_inference_steps), diffusion_steps))
    if count == 1 or diffusion_steps == 1:
        return [diffusion_steps - 1]
    raw = [
        int(round(index * (diffusion_steps - 1) / (count - 1)))
        for index in range(count - 1, -1, -1)
    ]
    times: list[int] = []
    for value in raw:
        if not times or times[-1] != value:
            times.append(value)
    return times


def train_diffusion(
    dataset: str,
    output_dir: str = "./checkpoints/diffusion",
    *,
    device: str = "cpu",
    train_steps: int = 200,
    batch_size: int = 8,
    lr: float = 1e-4,
    chunk_size: int = 16,
    vision_backbone: str = "none",
    diffusion_steps: int = 100,
    scheduler: str = "ddpm",
    num_inference_steps: int = 10,
    d_model: int = 64,
    log_every: int = 25,
    val_every: int | None = None,
    val_fraction: float = 0.2,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    on_event: Any = None,
    deterministic: bool = False,
    seed: int | None = None,
) -> str:
    """Train :class:`TinyDiffusionPolicy` and write the serve checkpoint layout."""
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if not TORCH_AVAILABLE:
        from ohho.train import TrainUnavailable

        raise TrainUnavailable(
            "Training a diffusion policy requires torch: pip install 'ohho-os[train]'."
        )
    states, chunks, state_dim, action_dim = load_windows(Path(dataset), chunk_size)
    tensor_s = torch.tensor(states, dtype=torch.float32)
    tensor_a = torch.tensor(chunks, dtype=torch.float32)
    with deterministic_algorithms(torch, enabled=deterministic):
        return _train_diffusion_body(
            out,
            tensor_s,
            tensor_a,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            train_steps=train_steps,
            batch_size=batch_size,
            lr=lr,
            chunk_size=chunk_size,
            vision_backbone=vision_backbone,
            diffusion_steps=diffusion_steps,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps,
            d_model=d_model,
            log_every=log_every,
            val_every=val_every,
            val_fraction=val_fraction,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            on_event=on_event,
            deterministic=deterministic,
            seed=seed,
        )


def _train_diffusion_body(
    out: Path,
    tensor_s: Any,
    tensor_a: Any,
    *,
    state_dim: int,
    action_dim: int,
    device: str,
    train_steps: int,
    batch_size: int,
    lr: float,
    chunk_size: int,
    vision_backbone: str,
    diffusion_steps: int,
    scheduler: str,
    num_inference_steps: int,
    d_model: int,
    log_every: int,
    val_every: int | None,
    val_fraction: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
    on_event: Any,
    deterministic: bool,
    seed: int | None,
) -> str:
    """Fit one diffusion run. Caller holds the deterministic-algorithm scope."""
    train_s, train_a, val_s, val_a = _split_tensors(
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
    model = TinyDiffusionPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        diffusion_steps=diffusion_steps,
        d_model=d_model,
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
    )
    model.to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_curve: list[tuple[int, float]] = []
    validation_curve: list[tuple[int, float]] = []
    tracker = ValTracker(early_stop_patience, early_stop_min_delta)
    step = 0
    stopped_early = False
    last_val: float | None = None
    while step < train_steps and not stopped_early:
        for batch_s, batch_a in loader:
            batch_s = batch_s.to(dev)
            batch_a = batch_a.to(dev)
            optimizer.zero_grad()
            total_loss = model.loss(batch_s, batch_a)
            total_loss.backward()
            optimizer.step()
            if step % log_every == 0 or step == train_steps - 1:
                loss_curve.append((step, round(float(total_loss.item()), 4)))
            step += 1
            if val_every is not None and validation_due(step, train_steps, val_every):
                vloss = _validation_loss(model, val_loader, dev)
                if not math.isfinite(vloss):
                    raise ValueError(f"validation loss is not finite at step {step}")
                last_val = vloss
                stop = tracker.observe(vloss)
                validation_curve.append((step, vloss))
                _emit_validation(
                    on_event,
                    step=step,
                    validation_loss=vloss,
                    best=float(tracker.best if tracker.best is not None else vloss),
                    train_loss=round(float(total_loss.item()), 6),
                )
                if stop:
                    stopped_early = True
                    _emit_early_stop(
                        on_event,
                        step=step,
                        validation_loss=vloss,
                        best=float(tracker.best if tracker.best is not None else vloss),
                        patience=early_stop_patience,
                    )
                    break
            if step >= train_steps:
                break
    completed = step
    torch.save(model.state_dict(), out / "policy.pt")
    final_loss = loss_curve[-1][1] if loss_curve else 0.0
    config = _config(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        diffusion_steps=diffusion_steps,
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        d_model=d_model,
        vision_backbone=vision_backbone,
        device=device,
        train_steps=train_steps,
        completed_steps=completed,
        stopped_early=stopped_early,
        val_every=val_every,
        final_loss=final_loss,
        deterministic=bool(deterministic),
    )
    _write_json(out / "config.json", config)
    metrics: dict[str, Any] = {
        "loss_curve": loss_curve,
        "final_loss": final_loss,
        "train_steps": train_steps,
        "completed_steps": completed,
        "stopped_early": stopped_early,
        "deterministic": bool(deterministic),
    }
    if last_val is not None:
        metrics["validation_loss"] = last_val
        metrics["validation_curve"] = validation_curve
    _write_json(out / "metrics.json", metrics)
    return str(out)


def load_diffusion_checkpoint(
    checkpoint_dir: str, device: str = "cpu"
) -> tuple[Any, dict[str, Any]]:
    """Load ``policy.pt`` with ``weights_only=True`` and return ``(model, config)``."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "Loading a diffusion checkpoint needs torch: pip install 'ohho-os[train]'"
        )
    directory = Path(checkpoint_dir).expanduser()
    config_file = (
        directory / "config.json"
        if directory.is_dir()
        else directory.parent / "config.json"
    )
    if not config_file.is_file():
        raise FileNotFoundError(
            f"diffusion checkpoint missing config.json in {directory}"
        )
    config = json.loads(config_file.read_text(encoding="utf-8"))
    weights = directory / "policy.pt" if directory.is_dir() else directory
    if not weights.is_file():
        raise FileNotFoundError(
            f"diffusion checkpoint missing policy.pt in {directory}"
        )
    model = TinyDiffusionPolicy(
        state_dim=int(config.get("state_dim", 9)),
        action_dim=int(config.get("action_dim", 9)),
        chunk_size=int(config.get("chunk_size", 16)),
        diffusion_steps=int(config.get("diffusion_steps", 100)),
        d_model=int(config.get("d_model", 64)),
        scheduler=str(config.get("scheduler", "ddpm")),
        num_inference_steps=int(config.get("num_inference_steps", 10)),
        beta_start=float(config.get("beta_start", BETA_START)),
        beta_end=float(config.get("beta_end", BETA_END)),
    )
    dev = torch.device(device)
    state_dict = torch.load(weights, map_location=dev, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()
    return model, config


class DiffusionPolicyModel:
    """``/load_model`` adapter. Serve imports this class by dotted path."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.policy: Any = None
        self.config: dict[str, Any] = {}

    def load_model(self, model_path: str, **kwargs: Any) -> None:
        if kwargs.get("device"):
            self.device = str(kwargs["device"])
        self.policy, self.config = load_diffusion_checkpoint(
            model_path, device=self.device
        )

    def predict_action(
        self, image: Any = None, instruction: str = "", **kwargs: Any
    ) -> dict[str, Any]:
        del image
        state_dim = int(self.config.get("state_dim", 9))
        action_dim = int(self.config.get("action_dim", 9))
        state = kwargs.get("state")
        if state is None:
            state = [0.0] * state_dim
        if self.policy is not None:
            action = self.policy.select_action(state)
        else:
            action = [0.0] * action_dim
        return {
            "vector": [float(item) for item in action],
            "instruction": instruction,
            "mock": self.policy is None,
        }


def _split_tensors(
    states: Any, actions: Any, val_every: int | None, val_fraction: float
) -> tuple[Any, Any, Any, Any]:
    count = int(states.shape[0])
    val_n = 0
    if val_every is not None and count >= 2:
        val_n = int(round(count * float(val_fraction)))
        val_n = min(max(val_n, 1), count - 1)
    if val_n:
        return states[:-val_n], actions[:-val_n], states[-val_n:], actions[-val_n:]
    return states, actions, states, actions


def _validation_loss(model: Any, loader: Any, dev: Any) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_s, batch_a in loader:
            batch_s = batch_s.to(dev)
            batch_a = batch_a.to(dev)
            loss = model.loss(batch_s, batch_a)
            total += float(loss.item()) * int(batch_s.shape[0])
            count += int(batch_s.shape[0])
    model.train()
    return round(total / max(count, 1), 6)


def _emit_validation(
    on_event: Any,
    *,
    step: int,
    validation_loss: float,
    best: float,
    train_loss: float,
) -> None:
    if on_event is None:
        return
    on_event(
        {
            "event": "validation",
            "step": step,
            "validation_loss": validation_loss,
            "best_validation_loss": best,
            "train_loss": train_loss,
        }
    )


def _emit_early_stop(
    on_event: Any,
    *,
    step: int,
    validation_loss: float,
    best: float,
    patience: int,
) -> None:
    if on_event is None:
        return
    on_event(
        {
            "event": "early_stop",
            "step": step,
            "validation_loss": validation_loss,
            "best_validation_loss": best,
            "patience": patience,
        }
    )


def _config(**fields: Any) -> dict[str, Any]:
    body = {
        "policy": "diffusion",
        "model_type": "diffusion",
        "model_family": "diffusion",
        "mock": False,
        "beta_start": BETA_START,
        "beta_end": BETA_END,
    }
    body.update(fields)
    return body


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
