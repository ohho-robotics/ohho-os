"""Tiny Action Chunking Transformer (ACT) model for CPU training and inference.

Implements the CVAE + Transformer architecture (Zhao et al., 2023) in pure
PyTorch without heavy external robotics dependencies. Suitable for fast CPU
fine-tuning on simulated and real demonstration datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..data.reader import DatasetReader

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
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_encoder_layers
            )
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
    mock: bool = False,
    log_every: int = 25,
    **kwargs: Any,
) -> str:
    """Train a TinyACTPolicy on a LeRobot v2.0 dataset on CPU."""
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    reader = DatasetReader(dataset)
    state_dim = reader.state_dim or 9
    action_dim = reader.action_dim or 9
    layout = _layout_fields(reader)

    if mock:
        loss_curve = [
            (0, 0.8421),
            (50, 0.4312),
            (100, 0.1854),
            (150, 0.0891),
            (train_steps, 0.0412),
        ]
        config = {
            "policy": "act",
            "mock": True,
            "device": device,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "chunk_size": chunk_size,
            "train_steps": train_steps,
            "loss_curve": loss_curve,
            "final_loss": loss_curve[-1][1],
        }
        config.update(layout)
        with open(out / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {"loss_curve": loss_curve, "final_loss": loss_curve[-1][1]}, f, indent=2
            )
        with open(out / "checkpoint.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return str(out)

    if not TORCH_AVAILABLE:
        from . import TrainUnavailable

        raise TrainUnavailable(
            "Training ACT policy requires torch: pip install 'ohho-os[train]'. "
            "Use mock=True for the dependency-free sim loop."
        )

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

    ds = TensorDataset(tensor_s, tensor_a)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

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
    )
    model.to(dev)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_curve: List[Tuple[int, float]] = []
    step = 0
    while step < train_steps:
        for batch_s, batch_a in loader:
            batch_s = batch_s.to(dev)
            batch_a = batch_a.to(dev)

            optimizer.zero_grad()
            pred_a, mu, logvar = model(batch_s, batch_a)
            assert mu is not None and logvar is not None
            total_loss, _l1, _kl = model.compute_loss(pred_a, batch_a, mu, logvar)
            total_loss.backward()
            optimizer.step()

            if step % log_every == 0 or step == train_steps - 1:
                loss_val = round(float(total_loss.item()), 4)
                loss_curve.append((step, loss_val))

            step += 1
            if step >= train_steps:
                break

    torch.save(model.state_dict(), out / "policy.pt")

    config = {
        "policy": "act",
        "mock": False,
        "device": device,
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
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1][1] if loss_curve else 0.0,
    }
    config.update(layout)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss_curve": loss_curve,
                "final_loss": loss_curve[-1][1] if loss_curve else 0.0,
                "train_steps": train_steps,
            },
            f,
            indent=2,
        )

    return str(out)


def _layout_fields(reader: DatasetReader) -> Dict[str, Any]:
    """State and action names from the dataset, when the writer stored them."""
    fields: Dict[str, Any] = {}
    features = reader.info.features
    for key, dest in (
        ("observation.state", "state_names"),
        ("action", "action_names"),
    ):
        feature = features.get(key) or {}
        names = feature.get("names") if isinstance(feature, dict) else None
        if (
            isinstance(names, list)
            and names
            and all(isinstance(name, str) for name in names)
        ):
            fields[dest] = list(names)
    if reader.info.arm_action:
        fields["arm_action"] = reader.info.arm_action
    return fields


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
    state_dict = torch.load(pt_file, map_location=dev, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()
    return model, config
