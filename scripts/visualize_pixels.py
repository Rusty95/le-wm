#!/usr/bin/env python3
"""Visualize pixels saved from IsaacLab NPZ or stable-worldmodel HDF5 files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a contact sheet and optional GIF from a pixels column."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input .npz episode or .h5 dataset.",
    )
    parser.add_argument(
        "--key",
        default="pixels",
        help="Image key/column to visualize.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index for HDF5 files. Ignored for NPZ files.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame inside the episode.")
    parser.add_argument("--count", type=int, default=16, help="Number of frames to draw.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in the contact sheet.",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=160,
        help="Long edge size for each thumbnail.",
    )
    parser.add_argument(
        "--crop",
        default=None,
        help="Optional crop box before visualization: x0,y0,x1,y1 in image pixels.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/hall/code/.stable-wm/visualizations/pixels_sheet.png"),
        help="Output PNG contact sheet.",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        help="Optional output GIF path.",
    )
    parser.add_argument("--fps", type=float, default=8.0, help="GIF frames per second.")
    return parser.parse_args()


def to_hwc(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 3:
        frames = frames[None]
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (T,H,W,C) or (T,C,H,W), got {frames.shape}")

    if frames.shape[-1] in (1, 3, 4):
        out = frames
    elif frames.shape[1] in (1, 3, 4):
        out = np.moveaxis(frames, 1, -1)
    else:
        raise ValueError(f"Cannot infer image channel axis from shape {frames.shape}")

    if out.shape[-1] == 1:
        out = np.repeat(out, 3, axis=-1)
    if out.shape[-1] == 4:
        out = out[..., :3]
    return out


def to_uint8(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.dtype == np.uint8:
        return frames

    frames = frames.astype(np.float32)
    finite = np.isfinite(frames)
    if not finite.any():
        return np.zeros_like(frames, dtype=np.uint8)

    min_value = float(frames[finite].min())
    max_value = float(frames[finite].max())
    if min_value >= 0.0 and max_value <= 1.0:
        frames = frames * 255.0
    else:
        frames = (frames - min_value) / max(max_value - min_value, 1e-6) * 255.0

    return np.clip(frames, 0, 255).astype(np.uint8)


def read_npz(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(f"{path} has no key {key!r}; available keys: {list(data.files)}")
        return np.asarray(data[key])


def read_h5_episode(path: Path, key: str, episode: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise KeyError(f"{path} has no key {key!r}; available keys: {list(handle.keys())}")
        if "ep_offset" not in handle or "ep_len" not in handle:
            raise KeyError("HDF5 file must contain ep_offset and ep_len for episode slicing")
        if episode < 0 or episode >= len(handle["ep_len"]):
            raise IndexError(f"episode {episode} out of range; file has {len(handle['ep_len'])} episodes")

        offset = int(handle["ep_offset"][episode])
        length = int(handle["ep_len"][episode])
        return handle[key][offset : offset + length]


def load_frames(path: Path, key: str, episode: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        frames = read_npz(path, key)
    elif suffix in {".h5", ".hdf5"}:
        frames = read_h5_episode(path, key, episode)
    else:
        raise ValueError(f"Unsupported input suffix {path.suffix!r}; expected .npz, .h5, or .hdf5")
    return to_uint8(to_hwc(frames))


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


def resize_frame(frame: np.ndarray, long_edge: int) -> Image.Image:
    image = Image.fromarray(frame)
    w, h = image.size
    scale = long_edge / max(w, h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


def make_contact_sheet(frames: np.ndarray, cols: int, thumb_size: int) -> Image.Image:
    thumbs = [resize_frame(frame, thumb_size) for frame in frames]
    cell_w = max(thumb.width for thumb in thumbs)
    cell_h = max(thumb.height for thumb in thumbs) + 22
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = col * cell_w + (cell_w - thumb.width) // 2
        y = row * cell_h
        sheet.paste(thumb, (x, y))
        draw.text((col * cell_w + 6, y + thumb.height + 4), f"frame {idx}", fill=(20, 20, 20))

    return sheet


def main() -> None:
    args = parse_args()
    frames = crop_frames(load_frames(args.input, args.key, args.episode), parse_crop(args.crop))

    indices = np.arange(args.start, len(frames), args.stride)[: args.count]
    if len(indices) == 0:
        raise ValueError(f"No frames selected from {args.input}")
    selected = frames[indices]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet = make_contact_sheet(selected, args.cols, args.thumb_size)
    sheet.save(args.out)
    print(f"Wrote contact sheet: {args.out}")
    print(f"Source frames: shape={frames.shape}, dtype={frames.dtype}, selected={indices.tolist()}")

    if args.gif is not None:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        duration = 1.0 / args.fps
        imageio.mimsave(args.gif, selected, duration=duration)
        print(f"Wrote GIF: {args.gif}")


if __name__ == "__main__":
    main()
