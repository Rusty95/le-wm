#!/usr/bin/env python3
"""Evaluate multi-step latent rollout error for LeWM checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from utils import get_column_normalizer, get_img_preprocessor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate autoregressive latent rollout error on an IsaacLab HDF5 dataset."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="isaaclab_random_100k")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle windows before applying --limit-batches for representative sampling.",
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--action-mode",
        choices=["recorded", "zero", "shuffle"],
        default="recorded",
        help="Use recorded actions, mean actions (normalized zero), or actions from another batch item.",
    )
    parser.add_argument(
        "--stratify-angle",
        action="store_true",
        help="Report metrics by absolute pole-angle range; requires policy_obs.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_cache_dir(cache_dir: str | None) -> str | None:
    root = cache_dir or os.environ.get("LOCAL_DATASET_DIR") or os.environ.get("STABLEWM_HOME")
    if root is None:
        return None
    path = Path(root).expanduser()
    if path.name == "datasets":
        path = path.parent
    return str(path)


def resolve_dataset_name(dataset_name: str, cache_dir: str | None) -> str:
    name_path = Path(dataset_name).expanduser()
    if name_path.exists():
        return str(name_path)

    datasets_dir = (
        Path(cache_dir).expanduser() / "datasets"
        if cache_dir is not None
        else Path(os.environ.get("STABLEWM_HOME", "~/.stable_worldmodel")).expanduser() / "datasets"
    )
    local = name_path if name_path.is_absolute() else datasets_dir / name_path
    if local.exists():
        return dataset_name

    if name_path.suffix == "":
        h5_name = f"{dataset_name}.h5"
        if (datasets_dir / h5_name).exists():
            return h5_name
    return dataset_name


def infer_history_size(model: torch.nn.Module, fallback: int | None) -> int:
    if fallback is not None:
        return fallback
    predictor = getattr(model, "predictor", None)
    pos_embedding = getattr(predictor, "pos_embedding", None)
    if pos_embedding is None:
        raise ValueError("Could not infer history size; pass --history-size.")
    return int(pos_embedding.shape[1])


def build_dataset(
    data_name: str,
    cache_dir: str | None,
    history_size: int,
    max_horizon: int,
    frameskip: int,
    img_size: int,
    load_policy_obs: bool = False,
):
    keys_to_load = ["pixels", "action"]
    if load_policy_obs:
        keys_to_load.append("policy_obs")
    dataset = swm.data.load_dataset(
        data_name,
        cache_dir=cache_dir,
        transform=None,
        num_steps=history_size + max_horizon,
        frameskip=frameskip,
        keys_to_load=keys_to_load,
        keys_to_cache=["action"],
    )
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=img_size),
        get_column_normalizer(dataset, "action", "action"),
    )
    return dataset


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def rollout_predictions(model: torch.nn.Module, emb: torch.Tensor, act_emb: torch.Tensor, history_size: int, max_horizon: int):
    emb_list = list(emb[:, :history_size].unbind(dim=1))
    for step in range(max_horizon):
        lo = max(0, history_size + step - history_size)
        ctx_emb = torch.stack(emb_list[lo:], dim=1)
        ctx_act = act_emb[:, lo : history_size + step]
        emb_list.append(model.predict(ctx_emb, ctx_act)[:, -1])
    return torch.stack(emb_list[history_size:], dim=1)


def main() -> None:
    args = parse_args()
    cache_dir = resolve_cache_dir(args.cache_dir)
    data_name = resolve_dataset_name(args.data, cache_dir)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    horizons = sorted(set(args.horizons))
    max_horizon = max(horizons)

    model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=cache_dir)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    history_size = infer_history_size(model, args.history_size)

    dataset = build_dataset(
        data_name=data_name,
        cache_dir=cache_dir,
        history_size=history_size,
        max_horizon=max_horizon,
        frameskip=args.frameskip,
        img_size=args.img_size,
        load_policy_obs=args.stratify_angle,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )

    totals = {
        h: {"squared_error": 0.0, "target_power": 0.0, "cosine": 0.0}
        for h in horizons
    }
    angle_edges = [0.0, 0.25, 0.75, 1.5, 2.5, math.inf]
    angle_labels = ["0.00-0.25", "0.25-0.75", "0.75-1.50", "1.50-2.50", "2.50-pi"]
    angle_totals = {
        h: {
            label: {"count": 0, "squared_error": 0.0, "target_power": 0.0, "cosine": 0.0}
            for label in angle_labels
        }
        for h in horizons
    }
    seen = 0
    num_batches = 0
    shapes = None

    with torch.no_grad():
        for batch in tqdm(loader, desc="Rollout eval", unit="batch"):
            if args.limit_batches is not None and num_batches >= args.limit_batches:
                break
            batch = move_batch(batch, device)
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            if args.action_mode == "zero":
                batch["action"] = torch.zeros_like(batch["action"])
            elif args.action_mode == "shuffle":
                batch["action"] = torch.roll(batch["action"], shifts=1, dims=0)
            out = model.encode(batch)
            emb = out["emb"]
            act_emb = out["act_emb"]
            pred = rollout_predictions(model, emb, act_emb, history_size, max_horizon)
            tgt = emb[:, history_size : history_size + max_horizon]

            batch_size = int(emb.shape[0])
            for horizon in horizons:
                pred_h = pred[:, horizon - 1]
                tgt_h = tgt[:, horizon - 1]
                totals[horizon]["squared_error"] += float(
                    (pred_h - tgt_h).pow(2).mean(dim=-1).sum().cpu()
                )
                totals[horizon]["target_power"] += float(
                    tgt_h.pow(2).mean(dim=-1).sum().cpu()
                )
                totals[horizon]["cosine"] += float(
                    torch.cosine_similarity(pred_h, tgt_h, dim=-1).sum().cpu()
                )
                if args.stratify_angle:
                    raw_angles = batch["policy_obs"][:, history_size + horizon - 1, 0]
                    angles = torch.atan2(torch.sin(raw_angles), torch.cos(raw_angles)).abs()
                    sample_mse = (pred_h - tgt_h).pow(2).mean(dim=-1)
                    sample_power = tgt_h.pow(2).mean(dim=-1)
                    sample_cosine = torch.cosine_similarity(pred_h, tgt_h, dim=-1)
                    for bin_index, label in enumerate(angle_labels):
                        mask = (angles >= angle_edges[bin_index]) & (
                            angles < angle_edges[bin_index + 1]
                        )
                        count = int(mask.sum().item())
                        if count == 0:
                            continue
                        bucket = angle_totals[horizon][label]
                        bucket["count"] += count
                        bucket["squared_error"] += float(sample_mse[mask].sum().cpu())
                        bucket["target_power"] += float(sample_power[mask].sum().cpu())
                        bucket["cosine"] += float(sample_cosine[mask].sum().cpu())
            seen += batch_size
            num_batches += 1
            if shapes is None:
                shapes = {
                    "emb_shape": tuple(emb.shape),
                    "act_emb_shape": tuple(act_emb.shape),
                    "pred_rollout_shape": tuple(pred.shape),
                    "tgt_rollout_shape": tuple(tgt.shape),
                }

    if seen == 0:
        raise RuntimeError("No batches were evaluated.")

    metrics = {}
    for horizon in horizons:
        mse = totals[horizon]["squared_error"] / seen
        target_power = totals[horizon]["target_power"] / seen
        metrics[f"horizon_{horizon}_mse"] = mse
        metrics[f"horizon_{horizon}_rmse"] = math.sqrt(mse)
        metrics[f"horizon_{horizon}_relative_rmse"] = math.sqrt(
            mse / max(target_power, 1e-12)
        )
        metrics[f"horizon_{horizon}_cosine_similarity"] = (
            totals[horizon]["cosine"] / seen
        )

    angle_metrics = None
    if args.stratify_angle:
        angle_metrics = {}
        for horizon in horizons:
            horizon_metrics = {}
            for label in angle_labels:
                bucket = angle_totals[horizon][label]
                count = bucket["count"]
                if count == 0:
                    continue
                mse = bucket["squared_error"] / count
                target_power = bucket["target_power"] / count
                horizon_metrics[label] = {
                    "count": count,
                    "mse": mse,
                    "relative_rmse": math.sqrt(mse / max(target_power, 1e-12)),
                    "cosine_similarity": bucket["cosine"] / count,
                }
            angle_metrics[f"horizon_{horizon}"] = horizon_metrics

    result = {
        "checkpoint": args.checkpoint,
        "data": data_name,
        "cache_dir": cache_dir,
        "device": str(device),
        "num_samples": seen,
        "num_batches": num_batches,
        "history_size": history_size,
        "horizons": horizons,
        "frameskip": args.frameskip,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "action_mode": args.action_mode,
        "metrics": metrics,
        "angle_stratified_metrics": angle_metrics,
        "shapes": shapes,
        "note": "Uses dataset windows from the provided HDF5. This is not a new held-out collection unless --data points to a separately collected test file.",
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
