#!/usr/bin/env python3
"""Convert IsaacLab episode NPZ files into stable-worldmodel HDF5.

Input layout:
  one or more ``.npz`` files, each containing at least:
    pixels: (T,H,W,C) or (T,C,H,W)
    action: (T,A)

Optional keys such as reward, done, proprio, and state are preserved.

Output layout:
  flat HDF5 columns plus ep_len/ep_offset, matching
  stable_worldmodel.data.HDF5Dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


RESERVED = {"ep_len", "ep_offset"}


def _to_hwc_pixels(pixels: np.ndarray) -> np.ndarray:
    if pixels.ndim != 4:
        raise ValueError(f"pixels must have shape (T,H,W,C) or (T,C,H,W), got {pixels.shape}")
    if pixels.shape[-1] in (1, 3, 4):
        out = pixels
    elif pixels.shape[1] in (1, 3, 4):
        out = np.moveaxis(pixels, 1, -1)
    else:
        raise ValueError(
            "Cannot infer pixel channel axis. Expected channel dimension to be "
            f"1, 3, or 4 in axis 1 or -1, got {pixels.shape}."
        )
    if out.shape[-1] == 4:
        out = out[..., :3]
    return out


def _as_episode_dict(npz_path: Path, keys: list[str] | None) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        available = set(data.files) - RESERVED
        selected = keys or sorted(available)
        missing = [key for key in selected if key not in available]
        if missing:
            raise KeyError(f"{npz_path} is missing keys: {missing}")

        episode = {key: np.asarray(data[key]) for key in selected}

    if "pixels" not in episode:
        raise KeyError(f"{npz_path} must contain a 'pixels' array")
    if "action" not in episode:
        raise KeyError(f"{npz_path} must contain an 'action' array")

    episode["pixels"] = _to_hwc_pixels(episode["pixels"])

    lengths = {key: value.shape[0] for key, value in episode.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{npz_path} has inconsistent time lengths: {lengths}")

    # LeWM image transforms expect image-like numeric arrays.
    if episode["pixels"].dtype == np.float64:
        episode["pixels"] = episode["pixels"].astype(np.float32)
    if episode["action"].dtype == np.float64:
        episode["action"] = episode["action"].astype(np.float32)

    return episode


def _init_dataset(handle: h5py.File, key: str, values: np.ndarray) -> None:
    sample_shape = values.shape[1:]
    handle.create_dataset(
        key,
        shape=(0, *sample_shape),
        maxshape=(None, *sample_shape),
        dtype=values.dtype,
        chunks=(1, *sample_shape),
    )


def write_h5(episodes: list[dict[str, np.ndarray]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    keys = list(episodes[0].keys())
    for idx, episode in enumerate(episodes):
        if set(episode) != set(keys):
            raise ValueError(f"Episode {idx} schema does not match first episode")

    with h5py.File(output, "w", libver="latest") as handle:
        for key in keys:
            _init_dataset(handle, key, episodes[0][key])
        handle.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32)
        handle.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64)

        offset = 0
        for ep_idx, episode in enumerate(episodes):
            length = next(iter(episode.values())).shape[0]
            for key in keys:
                ds = handle[key]
                ds.resize(offset + length, axis=0)
                ds[offset : offset + length] = episode[key]

            handle["ep_len"].resize(ep_idx + 1, axis=0)
            handle["ep_len"][ep_idx] = length
            handle["ep_offset"].resize(ep_idx + 1, axis=0)
            handle["ep_offset"][ep_idx] = offset
            offset += length


def write_h5_stream(npz_files: list[Path], output: Path, keys: list[str] | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    first_episode = _as_episode_dict(npz_files[0], keys)
    schema_keys = list(first_episode.keys())

    def append_episode(handle: h5py.File, ep_idx: int, offset: int, episode: dict[str, np.ndarray]) -> int:
        if set(episode) != set(schema_keys):
            raise ValueError(f"Episode {ep_idx} schema does not match first episode")
        length = next(iter(episode.values())).shape[0]
        for key in schema_keys:
            ds = handle[key]
            ds.resize(offset + length, axis=0)
            ds[offset : offset + length] = episode[key]

        handle["ep_len"].resize(ep_idx + 1, axis=0)
        handle["ep_len"][ep_idx] = length
        handle["ep_offset"].resize(ep_idx + 1, axis=0)
        handle["ep_offset"][ep_idx] = offset
        return offset + length

    with h5py.File(output, "w", libver="latest") as handle:
        for key in schema_keys:
            _init_dataset(handle, key, first_episode[key])
        handle.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32)
        handle.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64)

        offset = append_episode(handle, 0, 0, first_episode)
        for ep_idx, npz_path in enumerate(npz_files[1:], start=1):
            episode = _as_episode_dict(npz_path, keys)
            offset = append_episode(handle, ep_idx, offset, episode)
            if ep_idx % 100 == 0:
                print(f"Converted {ep_idx + 1}/{len(npz_files)} episodes, steps={offset}")

    print(f"Wrote {len(npz_files)} episodes / {offset} steps to {output}")
    print("Columns:", ", ".join(schema_keys))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="NPZ episode file or directory of NPZ files")
    parser.add_argument("output", type=Path, help="Output .h5 path")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Keys to preserve. Defaults to all keys in each NPZ.",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        npz_files = sorted(args.input.glob("*.npz"))
    else:
        npz_files = [args.input]

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {args.input}")

    write_h5_stream(npz_files, args.output, args.keys)


if __name__ == "__main__":
    main()
