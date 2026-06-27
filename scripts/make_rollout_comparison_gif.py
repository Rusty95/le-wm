#!/usr/bin/env python3
"""Create a GIF comparing rollout targets with nearest-neighbor predicted frames.

LeWM predicts latent embeddings, not pixels.  To make a visual comparison without
a decoder, each predicted latent is matched to its nearest real frame embedding
from the same episode, then that nearest frame is shown next to the target.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.eval_multistep_rollout import infer_history_size, rollout_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lewm_isaaclab_100k/weights_epoch_100.pt")
    parser.add_argument("--data", default="isaaclab_random_100k")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--thumb-size", type=int, default=160)
    parser.add_argument("--crop", default=None, help="Optional visualization crop box: x0,y0,x1,y1 in image pixels.")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=Path("/home/hall/code/.stable-wm/visualizations/lewm_rollout_compare.gif"))
    parser.add_argument("--sheet", type=Path, default=Path("/home/hall/code/.stable-wm/visualizations/lewm_rollout_compare_first.png"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_cache_dir(cache_dir: str | None) -> Path:
    root = cache_dir or os.environ.get("LOCAL_DATASET_DIR") or os.environ.get("STABLEWM_HOME")
    if root is None:
        root = "~/.stable_worldmodel"
    path = Path(root).expanduser()
    return path.parent if path.name == "datasets" else path


def resolve_h5_path(data: str, cache_dir: Path) -> Path:
    path = Path(data).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(cache_dir / "datasets" / path)
    if path.suffix == "":
        candidates.append(cache_dir / "datasets" / f"{data}.h5")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve dataset {data!r} under {cache_dir}")


def read_episode(h5_path: Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        offset = int(handle["ep_offset"][episode])
        length = int(handle["ep_len"][episode])
        pixels = np.asarray(handle["pixels"][offset : offset + length])
        actions = np.asarray(handle["action"][offset : offset + length])
    return pixels, actions


def action_stats(h5_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(h5_path, "r") as handle:
        action = np.asarray(handle["action"])
    data = torch.from_numpy(action).float()
    data = data[~torch.isnan(data).any(dim=1)]
    return data.mean(0, keepdim=True), data.std(0, keepdim=True).clamp_min(1e-6)


def visualize_pixels(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim == 3:
        frames = frames[None]
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if frames.dtype == np.uint8:
        return frames
    frames = frames.astype(np.float32)
    finite = np.isfinite(frames)
    if not finite.any():
        return np.zeros_like(frames, dtype=np.uint8)
    lo = float(frames[finite].min())
    hi = float(frames[finite].max())
    frames = (frames - lo) / max(hi - lo, 1e-6) * 255.0
    return np.clip(frames, 0, 255).astype(np.uint8)


def parse_crop(crop: str | None) -> tuple[int, int, int, int] | None:
    if crop is None:
        return None
    parts = [int(value.strip()) for value in crop.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop must be formatted as x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("--crop requires x1>x0 and y1>y0")
    return x0, y0, x1, y1


def crop_frames(frames: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return frames
    x0, y0, x1, y1 = crop
    height, width = frames.shape[1:3]
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Crop {crop} is outside frame shape {frames.shape}")
    return frames[:, y0:y1, x0:x1]


def preprocess_pixels(pixels: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(pixels).float()
    if tensor.ndim != 4:
        raise ValueError(f"Expected pixels with shape (T,H,W,C) or (T,C,H,W), got {tuple(tensor.shape)}")
    if tensor.shape[-1] in (1, 3, 4):
        tensor = tensor[..., :3].permute(0, 3, 1, 2)
    elif tensor.shape[1] in (1, 3, 4):
        tensor = tensor[:, :3]
    else:
        raise ValueError(f"Cannot infer channel axis from pixels shape {tuple(tensor.shape)}")
    tensor = F.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.to(device)


def encode_episode(model: torch.nn.Module, pixels: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(pixels), 32):
            pix = preprocess_pixels(pixels[start : start + 32], img_size, device)
            out = model.encode({"pixels": pix.unsqueeze(0)})
            chunks.append(out["emb"].reshape(-1, out["emb"].shape[-1]).detach())
    return torch.cat(chunks, dim=0)


def make_batch(
    pixels: np.ndarray,
    actions: np.ndarray,
    start: int,
    length: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    img_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    pix = preprocess_pixels(pixels[start : start + length], img_size, device).unsqueeze(0)
    act = torch.from_numpy(actions[start : start + length]).float().to(device)
    act = ((act - action_mean) / action_std).unsqueeze(0)
    return {"pixels": pix, "action": act}


def resize(image: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR)


def draw_frame(
    raw_vis: np.ndarray,
    start: int,
    horizons: list[int],
    pred_indices: dict[int, int],
    mses: dict[int, float],
    thumb_size: int,
) -> Image.Image:
    font = ImageFont.load_default()
    label_h = 24
    left_w = 92
    cols = 2
    rows = len(horizons)
    width = left_w + cols * thumb_size
    height = label_h + rows * (thumb_size + label_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((left_w + 8, 5), "target", fill=(20, 20, 20), font=font)
    draw.text((left_w + thumb_size + 8, 5), "pred NN", fill=(20, 20, 20), font=font)

    for row, horizon in enumerate(horizons):
        y = label_h + row * (thumb_size + label_h)
        target_idx = start + horizon
        pred_idx = pred_indices[horizon]
        draw.text((6, y + 4), f"+{horizon}", fill=(10, 48, 72), font=font)
        draw.text((6, y + 17), f"mse {mses[horizon]:.4f}", fill=(80, 80, 80), font=font)
        canvas.paste(resize(raw_vis[target_idx], thumb_size), (left_w, y))
        canvas.paste(resize(raw_vis[pred_idx], thumb_size), (left_w + thumb_size, y))
        draw.text((left_w + 4, y + thumb_size + 4), f"frame {target_idx}", fill=(20, 20, 20), font=font)
        draw.text((left_w + thumb_size + 4, y + thumb_size + 4), f"frame {pred_idx}", fill=(20, 20, 20), font=font)
    return canvas


def main() -> None:
    args = parse_args()
    cache_dir = resolve_cache_dir(args.cache_dir)
    h5_path = resolve_h5_path(args.data, cache_dir)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    horizons = sorted(set(args.horizons))
    max_horizon = max(horizons)
    model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=str(cache_dir))
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    history_size = infer_history_size(model, None)

    pixels, actions = read_episode(h5_path, args.episode)
    raw_vis = crop_frames(visualize_pixels(pixels), parse_crop(args.crop))
    action_mean, action_std = action_stats(h5_path)
    action_mean = action_mean.to(device)
    action_std = action_std.to(device)
    episode_emb = encode_episode(model, pixels, args.img_size, device)

    max_start = len(pixels) - history_size - max_horizon
    starts = list(range(args.start, min(args.start + args.num_frames, max_start + 1)))
    frames = []
    with torch.no_grad():
        for start in starts:
            batch = make_batch(
                pixels,
                actions,
                start,
                history_size + max_horizon,
                action_mean,
                action_std,
                args.img_size,
                device,
            )
            out = model.encode(batch)
            emb = out["emb"]
            act_emb = out["act_emb"]
            pred = rollout_predictions(model, emb, act_emb, history_size, max_horizon)[0]
            tgt = emb[0, history_size : history_size + max_horizon]
            pred_indices = {}
            mses = {}
            for horizon in horizons:
                pred_h = pred[horizon - 1]
                dists = (episode_emb - pred_h).pow(2).mean(dim=1)
                pred_indices[horizon] = int(dists.argmin().item())
                mses[horizon] = float((pred_h - tgt[horizon - 1]).pow(2).mean().cpu())
            frames.append(draw_frame(raw_vis, start, horizons, pred_indices, mses, args.thumb_size))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.sheet.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, [np.asarray(frame) for frame in frames], duration=1.0 / args.fps)
    frames[0].save(args.sheet)
    print(f"Wrote GIF: {args.out}")
    print(f"Wrote first-frame sheet: {args.sheet}")
    print(f"episode={args.episode}, starts={starts[0]}..{starts[-1]}, horizons={horizons}")


if __name__ == "__main__":
    main()
