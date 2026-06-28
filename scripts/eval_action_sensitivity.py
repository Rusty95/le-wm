#!/usr/bin/env python3
"""Diagnose whether LeWM rollouts are sensitive to different action plans.

The script fixes real image/action histories from an IsaacLab HDF5 dataset,
rolls out several hand-coded future action sequences, and compares the
predicted embeddings.  If -1/0/+1 plans produce nearly identical embeddings,
MPC has little usable action-conditioned signal to optimize.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.lewm_isaaclab_common import (  # noqa: E402
    DEFAULT_ACTION_STATS_H5,
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT,
    infer_history_size,
    load_action_stats,
    load_lewm,
    load_state_probe,
    normalize_actions,
    preprocess_pixels,
    rollout_predictions_online,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LeWM action sensitivity on fixed H5 histories.")
    parser.add_argument("--data", type=Path, default=Path("/home/hall/code/.stable-wm/datasets/isaaclab_policy_camera_test_10k.h5"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--probe", type=Path, default=Path("/home/hall/code/.stable-wm/checkpoints/lewm_cartpole_state_probe.pt"))
    parser.add_argument("--action-stats-h5", type=Path, nargs="+", default=DEFAULT_ACTION_STATS_H5)
    parser.add_argument("--max-stats-rows-per-file", type=int, default=50000)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--target-pole-threshold", type=float, default=0.08)
    parser.add_argument("--target-cart-threshold", type=float, default=0.25)
    parser.add_argument("--target-max-frames", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("/home/hall/code/.stable-wm/eval/lewm_action_sensitivity.json"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_action_plans(horizon: int, action_dim: int, device: torch.device) -> tuple[list[str], torch.Tensor]:
    specs = {
        "all_neg": [-1.0] * horizon,
        "all_zero": [0.0] * horizon,
        "all_pos": [1.0] * horizon,
        "neg_then_zero": [-1.0 if i < max(1, horizon // 2) else 0.0 for i in range(horizon)],
        "pos_then_zero": [1.0 if i < max(1, horizon // 2) else 0.0 for i in range(horizon)],
        "neg_pos_alt": [-1.0 if i % 2 == 0 else 1.0 for i in range(horizon)],
        "pos_neg_alt": [1.0 if i % 2 == 0 else -1.0 for i in range(horizon)],
    }
    names = list(specs.keys())
    plans = torch.tensor([specs[name] for name in names], dtype=torch.float32, device=device).unsqueeze(-1)
    if action_dim > 1:
        plans = plans.expand(-1, -1, action_dim).contiguous()
    return names, plans


def select_starts(h5: Any, history_size: int, horizon: int, samples: int) -> np.ndarray:
    total_needed = history_size + horizon
    if "ep_offset" in h5 and "ep_len" in h5:
        starts = []
        for offset, length in zip(np.asarray(h5["ep_offset"]), np.asarray(h5["ep_len"])):
            max_start = int(length) - total_needed
            if max_start >= 0:
                starts.extend((int(offset) + np.arange(max_start + 1)).tolist())
        starts = np.asarray(starts, dtype=np.int64)
    else:
        n = int(h5["pixels"].shape[0])
        starts = np.arange(max(0, n - total_needed + 1), dtype=np.int64)
    if len(starts) == 0:
        raise ValueError(f"No valid windows for history={history_size}, horizon={horizon}.")
    if len(starts) > samples:
        idx = np.linspace(0, len(starts) - 1, samples).round().astype(np.int64)
        starts = starts[idx]
    return starts


def build_target_emb(
    model: torch.nn.Module,
    h5: Any,
    img_size: int,
    device: torch.device,
    max_frames: int,
    pole_threshold: float,
    cart_threshold: float,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if "policy_obs" not in h5:
        return None, None
    obs = np.asarray(h5["policy_obs"][:], dtype=np.float32)
    mask = (np.abs(obs[:, 0]) <= pole_threshold) & (np.abs(obs[:, 2]) <= cart_threshold)
    indices = np.nonzero(mask)[0]
    if len(indices) == 0:
        return None, None
    if len(indices) > max_frames:
        pick = np.linspace(0, len(indices) - 1, max_frames).round().astype(np.int64)
        indices = indices[pick]
    frames = np.asarray(h5["pixels"][np.sort(indices)], dtype=np.float32)
    embs = []
    with torch.no_grad():
        for start in range(0, len(frames), 128):
            pixels = preprocess_pixels(frames[start : start + 128], img_size=img_size, device=device).unsqueeze(0)
            embs.append(model.encode({"pixels": pixels})["emb"].squeeze(0))
    target = F.normalize(torch.cat(embs, dim=0).mean(dim=0), dim=0)
    meta = {
        "target_frames": int(len(frames)),
        "target_pole_threshold": pole_threshold,
        "target_cart_threshold": cart_threshold,
    }
    return target, meta


def pairwise_metrics(pred: torch.Tensor, plan_names: list[str]) -> dict[str, Any]:
    # pred: (P, H, D)
    out: dict[str, Any] = {}
    for h_idx in range(pred.shape[1]):
        emb = pred[:, h_idx]
        mse = torch.cdist(emb, emb, p=2).square() / emb.shape[-1]
        cos = 1.0 - F.cosine_similarity(emb[:, None, :], emb[None, :, :], dim=-1)
        pairs = {}
        for i, ni in enumerate(plan_names):
            for j, nj in enumerate(plan_names):
                if j <= i:
                    continue
                pairs[f"{ni}_vs_{nj}"] = {
                    "mse": float(mse[i, j].cpu()),
                    "cosine_distance": float(cos[i, j].cpu()),
                }
        out[f"h{h_idx + 1}"] = pairs
    return out


def main() -> None:
    args = parse_args()
    import h5py

    device = resolve_device(args.device)
    action_mean, action_std = load_action_stats(args.action_stats_h5, args.max_stats_rows_per_file)
    action_mean = action_mean.to(device)
    action_std = action_std.to(device)
    action_dim = int(action_mean.shape[-1])

    model = load_lewm(args.checkpoint, args.cache_dir, action_dim, args.img_size, device)
    probe = load_state_probe(args.probe, device) if args.probe.exists() else None
    history_size = infer_history_size(model)
    plan_names, plans_raw = make_action_plans(args.horizon, action_dim, device)

    totals = {
        "pairwise_mse": {name: [0.0] * args.horizon for name in []},
        "all_neg_vs_all_zero_cos": [0.0] * args.horizon,
        "all_pos_vs_all_zero_cos": [0.0] * args.horizon,
        "all_neg_vs_all_pos_cos": [0.0] * args.horizon,
        "latent_effect_norm": [0.0] * args.horizon,
        "target_cost": {name: 0.0 for name in plan_names},
        "probe_pole_abs": {name: [0.0] * args.horizon for name in plan_names},
        "probe_pole_vel_abs": {name: [0.0] * args.horizon for name in plan_names},
    }
    examples = []

    with h5py.File(args.data, "r") as h5:
        starts = select_starts(h5, history_size, args.horizon, args.samples)
        target_emb, target_meta = build_target_emb(
            model=model,
            h5=h5,
            img_size=args.img_size,
            device=device,
            max_frames=args.target_max_frames,
            pole_threshold=args.target_pole_threshold,
            cart_threshold=args.target_cart_threshold,
        )

        with torch.no_grad():
            for sample_idx, start in enumerate(starts):
                start = int(start)
                frames = np.asarray(h5["pixels"][start : start + history_size], dtype=np.float32)
                past_actions = np.asarray(h5["action"][start : start + history_size], dtype=np.float32)
                pixels = preprocess_pixels(frames, img_size=args.img_size, device=device).unsqueeze(0)
                emb = model.encode({"pixels": pixels})["emb"].expand(len(plan_names), -1, -1).contiguous()

                past = torch.as_tensor(past_actions, dtype=torch.float32, device=device)
                if past.ndim == 1:
                    past = past.unsqueeze(-1)
                past = past[-(history_size - 1) :].unsqueeze(0).expand(len(plan_names), -1, -1)
                full_actions_raw = torch.cat([past, plans_raw], dim=1)
                full_actions = normalize_actions(full_actions_raw, action_mean, action_std, device)
                act_emb = model.action_encoder(full_actions)
                pred = rollout_predictions_online(model, emb, act_emb, history_size, args.horizon)

                cos = 1.0 - F.cosine_similarity(pred[:, None, :, :], pred[None, :, :, :], dim=-1)
                idx = {name: i for i, name in enumerate(plan_names)}
                for h in range(args.horizon):
                    totals["all_neg_vs_all_zero_cos"][h] += float(cos[idx["all_neg"], idx["all_zero"], h].cpu())
                    totals["all_pos_vs_all_zero_cos"][h] += float(cos[idx["all_pos"], idx["all_zero"], h].cpu())
                    totals["all_neg_vs_all_pos_cos"][h] += float(cos[idx["all_neg"], idx["all_pos"], h].cpu())
                    totals["latent_effect_norm"][h] += float((pred[:, h].max(dim=0).values - pred[:, h].min(dim=0).values).norm().cpu())

                if target_emb is not None:
                    target = target_emb.reshape(1, 1, -1).expand_as(pred)
                    target_cost = 1.0 - F.cosine_similarity(pred, target, dim=-1)
                    for i, name in enumerate(plan_names):
                        totals["target_cost"][name] += float(target_cost[i].sum().cpu())

                probe_states = None
                if probe is not None:
                    probe_states = probe(pred)
                    for i, name in enumerate(plan_names):
                        totals["probe_pole_abs"][name] = [
                            totals["probe_pole_abs"][name][h] + float(abs(probe_states[i, h, 0]).cpu())
                            for h in range(args.horizon)
                        ]
                        totals["probe_pole_vel_abs"][name] = [
                            totals["probe_pole_vel_abs"][name][h] + float(abs(probe_states[i, h, 1]).cpu())
                            for h in range(args.horizon)
                        ]

                if sample_idx < 3:
                    ex: dict[str, Any] = {
                        "sample": sample_idx,
                        "start": start,
                        "pairwise": pairwise_metrics(pred, plan_names),
                    }
                    if target_emb is not None:
                        ex["target_cost"] = {
                            name: float(target_cost[i].sum().cpu()) for i, name in enumerate(plan_names)
                        }
                    if probe_states is not None:
                        ex["probe_final_state"] = {
                            name: probe_states[i, -1].detach().cpu().tolist() for i, name in enumerate(plan_names)
                        }
                    examples.append(ex)

    n = float(len(starts))
    summary = {
        "all_neg_vs_all_zero_cos": [v / n for v in totals["all_neg_vs_all_zero_cos"]],
        "all_pos_vs_all_zero_cos": [v / n for v in totals["all_pos_vs_all_zero_cos"]],
        "all_neg_vs_all_pos_cos": [v / n for v in totals["all_neg_vs_all_pos_cos"]],
        "latent_effect_norm": [v / n for v in totals["latent_effect_norm"]],
        "target_cost": {k: v / n for k, v in totals["target_cost"].items()},
        "probe_pole_abs": {k: [x / n for x in v] for k, v in totals["probe_pole_abs"].items()},
        "probe_pole_vel_abs": {k: [x / n for x in v] for k, v in totals["probe_pole_vel_abs"].items()},
    }
    result = {
        "checkpoint": args.checkpoint,
        "data": str(args.data),
        "device": str(device),
        "samples": int(len(starts)),
        "history_size": history_size,
        "horizon": args.horizon,
        "plans": plan_names,
        "target": target_meta,
        "summary": summary,
        "examples": examples,
    }
    text = json.dumps(result, indent=2)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"[INFO] wrote {args.out}")


if __name__ == "__main__":
    main()
