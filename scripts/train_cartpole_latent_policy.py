#!/usr/bin/env python3
"""Train a lightweight LeWM-latent policy head for Cartpole.

This is a deployment baseline: PPO is not loaded at runtime.  The policy head
imitates recorded actions from an offline dataset using frozen LeWM latents.
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
    LatentPolicyHead,
    load_action_stats,
    load_lewm,
    preprocess_pixels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent policy head from IsaacLab HDF5 actions.")
    parser.add_argument("--train-data", type=Path, default=Path("/home/hall/code/.stable-wm/datasets/isaaclab_full_angle_120k.h5"))
    parser.add_argument("--test-data", type=Path, default=Path("/home/hall/code/.stable-wm/datasets/isaaclab_full_angle_test_10k_seed9317.h5"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--action-stats-h5", type=Path, nargs="+", default=DEFAULT_ACTION_STATS_H5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-low", type=float, default=-1.0)
    parser.add_argument("--action-high", type=float, default=1.0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("/home/hall/code/.stable-wm/checkpoints/lewm_full_angle_latent_policy.pt"))
    parser.add_argument("--metrics-out", type=Path, default=Path("/home/hall/code/.stable-wm/eval/lewm_full_angle_latent_policy_eval.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def read_episode_spans(path: Path, total: int) -> list[tuple[int, int]]:
    with h5py.File(path, "r") as f:
        if "ep_offset" not in f or "ep_len" not in f:
            return [(0, total)]
        offsets = np.asarray(f["ep_offset"][:], dtype=np.int64)
        lengths = np.asarray(f["ep_len"][:], dtype=np.int64)
    spans = []
    for offset, length in zip(offsets, lengths):
        start = int(offset)
        end = min(int(offset + length), total)
        if start < total and end > start:
            spans.append((start, end))
    return spans


def encode_h5(
    model: torch.nn.Module,
    path: Path,
    img_size: int,
    batch_size: int,
    device: torch.device,
    limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    embs = []
    actions = []
    with h5py.File(path, "r") as f:
        if "pixels" not in f or "action" not in f:
            raise KeyError(f"{path} must contain 'pixels' and 'action'.")
        total = int(f["pixels"].shape[0])
        if limit is not None:
            total = min(total, limit)
        for start in tqdm(range(0, total, batch_size), desc=f"Encoding {path.name}", unit="batch"):
            end = min(start + batch_size, total)
            pixels = np.asarray(f["pixels"][start:end])
            action = np.asarray(f["action"][start:end], dtype=np.float32)
            pix = preprocess_pixels(pixels, img_size=img_size, device=device).unsqueeze(0)
            with torch.no_grad():
                out = model.encode({"pixels": pix})
                emb = out["emb"].reshape(-1, out["emb"].shape[-1]).detach().cpu()
            embs.append(emb)
            actions.append(torch.from_numpy(action.reshape(action.shape[0], -1)))
    return torch.cat(embs, dim=0), torch.cat(actions, dim=0)


def make_history_inputs(
    emb: torch.Tensor,
    action: torch.Tensor,
    data_path: Path,
    history_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = []
    targets = []
    for start, end in read_episode_spans(data_path, emb.shape[0]):
        if end - start < history_size:
            continue
        for idx in range(start + history_size - 1, end):
            inputs.append(emb[idx - history_size + 1 : idx + 1].reshape(-1))
            targets.append(action[idx])
    if not inputs:
        raise ValueError(f"No valid policy windows in {data_path}.")
    return torch.stack(inputs, dim=0), torch.stack(targets, dim=0)


def evaluate(
    policy: LatentPolicyHead,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> dict[str, float | list[float]]:
    policy.eval()
    preds = []
    with torch.no_grad():
        for batch_inputs, in DataLoader(TensorDataset(inputs), batch_size=2048, shuffle=False):
            preds.append(policy(batch_inputs.to(device)).cpu())
    pred = torch.cat(preds, dim=0)
    err = pred - targets
    sign_match = (torch.sign(pred) == torch.sign(targets)).float().mean()
    return {
        "mse": float(err.pow(2).mean().item()),
        "mae": float(err.abs().mean().item()),
        "action_std": float(targets.std().item()),
        "pred_std": float(pred.std().item()),
        "sign_match": float(sign_match.item()),
        "target_mean": targets.mean(dim=0).tolist(),
        "pred_mean": pred.mean(dim=0).tolist(),
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

    train_emb, train_action = encode_h5(model, args.train_data, args.img_size, args.encode_batch_size, device, args.limit_train)
    test_emb, test_action = encode_h5(model, args.test_data, args.img_size, args.encode_batch_size, device, args.limit_test)
    train_input, train_target = make_history_inputs(train_emb, train_action, args.train_data, args.history_size)
    test_input, test_target = make_history_inputs(test_emb, test_action, args.test_data, args.history_size)

    policy = LatentPolicyHead(
        input_dim=train_input.shape[-1],
        hidden_dim=args.hidden_dim,
        action_dim=train_target.shape[-1],
        action_low=args.action_low,
        action_high=args.action_high,
    ).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(train_input, train_target),
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=False,
    )

    history = []
    for epoch in range(args.epochs):
        policy.train()
        total = 0.0
        seen = 0
        for batch_input, batch_target in loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            pred = policy(batch_input)
            loss = F.mse_loss(pred, batch_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * batch_input.shape[0]
            seen += batch_input.shape[0]
        train_loss = total / max(seen, 1)
        history.append(train_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch={epoch + 1:03d} train_mse={train_loss:.6f}")

    train_metrics = evaluate(policy, train_input, train_target, device)
    test_metrics = evaluate(policy, test_input, test_target, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "config": {
                "input_dim": int(train_input.shape[-1]),
                "latent_dim": int(train_emb.shape[-1]),
                "hidden_dim": int(args.hidden_dim),
                "action_dim": int(train_target.shape[-1]),
                "history_size": int(args.history_size),
                "checkpoint": args.checkpoint,
                "img_size": int(args.img_size),
                "action_low": float(args.action_low),
                "action_high": float(args.action_high),
            },
        },
        args.out,
    )
    result = {
        "checkpoint": args.checkpoint,
        "policy": str(args.out),
        "train_data": str(args.train_data),
        "test_data": str(args.test_data),
        "train_samples": int(train_input.shape[0]),
        "test_samples": int(test_input.shape[0]),
        "epochs": args.epochs,
        "train_loss_history": history,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    text = json.dumps(result, indent=2)
    print(text)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote policy: {args.out}")
    print(f"Wrote metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
