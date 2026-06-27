#!/usr/bin/env python3
"""Train a frozen-LeWM latent-to-Cartpole-state probe.

The probe predicts policy_obs in the fixed order:
    [pole_pos, pole_vel, cart_pos, cart_vel]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.lewm_isaaclab_common import (  # noqa: E402
    DEFAULT_ACTION_STATS_H5,
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT,
    StateProbe,
    load_action_stats,
    load_lewm,
    preprocess_pixels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent-to-cartpole-state probe for LeWM MPC.")
    parser.add_argument("--train-data", type=Path, default=Path("/home/hall/code/.stable-wm/datasets/isaaclab_policy_camera_50k.h5"))
    parser.add_argument("--test-data", type=Path, default=Path("/home/hall/code/.stable-wm/datasets/isaaclab_policy_camera_test_10k.h5"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--action-stats-h5", type=Path, nargs="+", default=DEFAULT_ACTION_STATS_H5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("/home/hall/code/.stable-wm/checkpoints/lewm_cartpole_state_probe.pt"))
    parser.add_argument("--metrics-out", type=Path, default=Path("/home/hall/code/.stable-wm/eval/lewm_cartpole_state_probe_eval.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def require_policy_obs(path: Path) -> None:
    with h5py.File(path, "r") as f:
        if "pixels" not in f or "policy_obs" not in f:
            raise KeyError(f"{path} must contain 'pixels' and 'policy_obs'.")


def encode_h5(
    model: torch.nn.Module,
    path: Path,
    img_size: int,
    batch_size: int,
    device: torch.device,
    limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    require_policy_obs(path)
    embs = []
    targets = []
    with h5py.File(path, "r") as f:
        total = int(f["pixels"].shape[0])
        if limit is not None:
            total = min(total, limit)
        for start in tqdm(range(0, total, batch_size), desc=f"Encoding {path.name}", unit="batch"):
            end = min(start + batch_size, total)
            pixels = np.asarray(f["pixels"][start:end])
            policy_obs = np.asarray(f["policy_obs"][start:end], dtype=np.float32)
            pix = preprocess_pixels(pixels, img_size=img_size, device=device).unsqueeze(0)
            with torch.no_grad():
                out = model.encode({"pixels": pix})
                emb = out["emb"].reshape(-1, out["emb"].shape[-1]).detach().cpu()
            embs.append(emb)
            targets.append(torch.from_numpy(policy_obs))
    return torch.cat(embs, dim=0), torch.cat(targets, dim=0)


def evaluate(probe: StateProbe, emb: torch.Tensor, target: torch.Tensor, device: torch.device) -> dict[str, float | list[float]]:
    probe.eval()
    preds = []
    with torch.no_grad():
        for batch_emb, in DataLoader(TensorDataset(emb), batch_size=1024, shuffle=False):
            preds.append(probe(batch_emb.to(device)).cpu())
    pred = torch.cat(preds, dim=0)
    err = pred - target
    mse_per_dim = err.pow(2).mean(dim=0)
    mae_per_dim = err.abs().mean(dim=0)
    names = ["pole_pos", "pole_vel", "cart_pos", "cart_vel"]
    return {
        "mse": float(err.pow(2).mean().item()),
        "mae": float(err.abs().mean().item()),
        "mse_per_dim": {name: float(value) for name, value in zip(names, mse_per_dim)},
        "mae_per_dim": {name: float(value) for name, value in zip(names, mae_per_dim)},
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    action_mean, _ = load_action_stats(args.action_stats_h5)
    model = load_lewm(
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        action_dim=int(action_mean.shape[-1]),
        img_size=args.img_size,
        device=device,
    )

    train_emb, train_target = encode_h5(model, args.train_data, args.img_size, args.encode_batch_size, device, args.limit_train)
    test_emb, test_target = encode_h5(model, args.test_data, args.img_size, args.encode_batch_size, device, args.limit_test)

    target_mean = train_target.mean(dim=0, keepdim=True)
    target_std = train_target.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_target_norm = (train_target - target_mean) / target_std

    probe = StateProbe(
        input_dim=train_emb.shape[-1],
        hidden_dim=args.hidden_dim,
        output_dim=train_target.shape[-1],
        target_mean=target_mean,
        target_std=target_std,
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(train_emb, train_target_norm),
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=False,
    )

    history = []
    for epoch in range(args.epochs):
        probe.train()
        total = 0.0
        seen = 0
        for batch_emb, batch_target in loader:
            batch_emb = batch_emb.to(device)
            batch_target = batch_target.to(device)
            pred = probe.forward_normalized(batch_emb)
            loss = F.mse_loss(pred, batch_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * batch_emb.shape[0]
            seen += batch_emb.shape[0]
        train_loss = total / max(seen, 1)
        history.append(train_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch={epoch + 1:03d} train_norm_mse={train_loss:.6f}")

    train_metrics = evaluate(probe, train_emb, train_target, device)
    test_metrics = evaluate(probe, test_emb, test_target, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "config": {
                "input_dim": int(train_emb.shape[-1]),
                "hidden_dim": int(args.hidden_dim),
                "output_dim": int(train_target.shape[-1]),
                "checkpoint": args.checkpoint,
                "img_size": args.img_size,
                "state_order": ["pole_pos", "pole_vel", "cart_pos", "cart_vel"],
            },
        },
        args.out,
    )
    result = {
        "checkpoint": args.checkpoint,
        "probe": str(args.out),
        "train_data": str(args.train_data),
        "test_data": str(args.test_data),
        "train_samples": int(train_emb.shape[0]),
        "test_samples": int(test_emb.shape[0]),
        "epochs": args.epochs,
        "train_loss_history": history,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    text = json.dumps(result, indent=2)
    print(text)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote probe: {args.out}")
    print(f"Wrote metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
