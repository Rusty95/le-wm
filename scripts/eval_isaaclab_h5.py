#!/usr/bin/env python3
"""Offline evaluation for LeWM checkpoints on IsaacLab HDF5 datasets."""

from __future__ import annotations

import argparse
import json
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

from module import SIGReg  # noqa: E402
from utils import get_column_normalizer, get_img_preprocessor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LeWM checkpoint on a local stable-worldmodel HDF5 dataset."
    )
    parser.add_argument(
        "--checkpoint",
        default="lewm/weights_epoch_100.pt",
        help=(
            "Checkpoint name relative to $STABLEWM_HOME/checkpoints, or an absolute .pt path. "
            "Example: lewm/weights_epoch_100.pt"
        ),
    )
    parser.add_argument(
        "--data",
        default="isaaclab_random.h5",
        help="Dataset name/path. 'isaaclab_random' is accepted if isaaclab_random.h5 exists.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="stable-worldmodel cache root. Defaults to LOCAL_DATASET_DIR, then STABLEWM_HOME.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for evaluation.",
    )
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
    if pos_embedding is not None:
        return int(pos_embedding.shape[1])

    raise ValueError("Could not infer history size; pass --history-size explicitly.")


def load_model(checkpoint: str, cache_dir: str | None, device: torch.device) -> torch.nn.Module:
    model = swm.wm.utils.load_pretrained(checkpoint, cache_dir=cache_dir)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def build_dataset(
    data_name: str,
    cache_dir: str | None,
    history_size: int,
    num_preds: int,
    frameskip: int,
    img_size: int,
):
    dataset = swm.data.load_dataset(
        data_name,
        cache_dir=cache_dir,
        transform=None,
        num_steps=history_size + num_preds,
        frameskip=frameskip,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )

    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=img_size),
        get_column_normalizer(dataset, "action", "action"),
    ]
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def eval_batch(
    model: torch.nn.Module,
    sigreg: SIGReg,
    batch: dict,
    history_size: int,
    num_preds: int,
    sigreg_weight: float,
) -> dict[str, torch.Tensor | tuple[int, ...]]:
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, num_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))
    loss = pred_loss + sigreg_weight * sigreg_loss

    return {
        "pred_loss": pred_loss,
        "sigreg_loss": sigreg_loss,
        "loss": loss,
        "emb_shape": tuple(emb.shape),
        "act_emb_shape": tuple(act_emb.shape),
        "pred_emb_shape": tuple(pred_emb.shape),
        "tgt_emb_shape": tuple(tgt_emb.shape),
    }


def main() -> None:
    args = parse_args()
    cache_dir = resolve_cache_dir(args.cache_dir)
    data_name = resolve_dataset_name(args.data, cache_dir)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_model(args.checkpoint, cache_dir=cache_dir, device=device)
    history_size = infer_history_size(model, args.history_size)
    dataset = build_dataset(
        data_name=data_name,
        cache_dir=cache_dir,
        history_size=history_size,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
        img_size=args.img_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_num_proj).to(device)

    totals = {"pred_loss": 0.0, "sigreg_loss": 0.0, "loss": 0.0}
    seen = 0
    shapes = None
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            if args.limit_batches is not None and num_batches >= args.limit_batches:
                break

            batch = move_batch(batch, device)
            out = eval_batch(
                model=model,
                sigreg=sigreg,
                batch=batch,
                history_size=history_size,
                num_preds=args.num_preds,
                sigreg_weight=args.sigreg_weight,
            )
            batch_size = int(batch["pixels"].shape[0])
            for key in totals:
                totals[key] += float(out[key].detach().cpu()) * batch_size
            seen += batch_size
            num_batches += 1
            if shapes is None:
                shapes = {k: v for k, v in out.items() if k.endswith("_shape")}

    if seen == 0:
        raise RuntimeError("No batches were evaluated.")

    metrics = {key: value / seen for key, value in totals.items()}
    result = {
        "checkpoint": args.checkpoint,
        "data": data_name,
        "cache_dir": cache_dir,
        "device": str(device),
        "num_samples": seen,
        "num_batches": num_batches,
        "history_size": history_size,
        "num_preds": args.num_preds,
        "frameskip": args.frameskip,
        "metrics": metrics,
        "shapes": shapes,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
