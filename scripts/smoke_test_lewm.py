#!/usr/bin/env python3
"""Run a minimal LeWM forward pass with fake pixels/actions.

This checks the LeWM model wiring without requiring IsaacLab, datasets, Hydra,
or stable-pretraining.  It uses a tiny fake ViT-like encoder that returns a
HuggingFace-style object with ``last_hidden_state``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from jepa import JEPA  # noqa: E402
from module import ARPredictor, Embedder, MLP, SIGReg  # noqa: E402


class FakeViTEncoder(nn.Module):
    """Small encoder that mimics transformers.ViTModel output shape."""

    def __init__(self, embed_dim: int, num_tokens: int = 257):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, embed_dim),
        )
        self.patch_tokens = nn.Parameter(torch.zeros(1, num_tokens - 1, embed_dim))

    def forward(self, pixels: torch.Tensor, **_: object) -> SimpleNamespace:
        cls = self.proj(pixels).unsqueeze(1)
        patch = self.patch_tokens.expand(pixels.size(0), -1, -1)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, patch], dim=1))


def build_model(embed_dim: int, action_dim: int, history_size: int) -> JEPA:
    return JEPA(
        encoder=FakeViTEncoder(embed_dim=embed_dim),
        predictor=ARPredictor(
            num_frames=history_size,
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            depth=2,
            heads=4,
            mlp_dim=embed_dim * 4,
            dim_head=32,
            dropout=0.0,
            emb_dropout=0.0,
        ),
        action_encoder=Embedder(input_dim=action_dim, emb_dim=embed_dim),
        projector=MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=embed_dim * 4,
            norm_fn=nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=embed_dim * 4,
            norm_fn=nn.BatchNorm1d,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-len", type=int, default=4)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()

    if args.sequence_len <= args.history_size:
        raise ValueError("--sequence-len must be greater than --history-size")

    torch.manual_seed(0)
    model = build_model(args.embed_dim, args.action_dim, args.history_size)
    sigreg = SIGReg(knots=5, num_proj=32)

    batch = {
        "pixels": torch.randn(
            args.batch_size,
            args.sequence_len,
            3,
            args.image_size,
            args.image_size,
        ),
        "action": torch.randn(args.batch_size, args.sequence_len, args.action_dim),
    }

    output = model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, : args.history_size]
    ctx_act = act_emb[:, : args.history_size]
    tgt_emb = emb[:, 1:]
    pred_emb = model.predict(ctx_emb, ctx_act)

    if pred_emb.shape != tgt_emb[:, : pred_emb.size(1)].shape:
        raise AssertionError(
            f"pred/tgt shape mismatch: pred={tuple(pred_emb.shape)} "
            f"tgt={tuple(tgt_emb.shape)}"
        )

    pred_loss = (pred_emb - tgt_emb[:, : pred_emb.size(1)]).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))
    loss = pred_loss + 0.09 * sigreg_loss

    print("LeWM smoke test passed")
    print(f"emb:         {tuple(emb.shape)}")
    print(f"act_emb:     {tuple(act_emb.shape)}")
    print(f"ctx_emb:     {tuple(ctx_emb.shape)}")
    print(f"ctx_act:     {tuple(ctx_act.shape)}")
    print(f"pred_emb:    {tuple(pred_emb.shape)}")
    print(f"tgt_emb:     {tuple(tgt_emb[:, : pred_emb.size(1)].shape)}")
    print(f"pred_loss:   {pred_loss.item():.6f}")
    print(f"sigreg_loss: {sigreg_loss.item():.6f}")
    print(f"loss:        {loss.item():.6f}")


if __name__ == "__main__":
    main()
