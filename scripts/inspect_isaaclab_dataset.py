#!/usr/bin/env python3
"""Inspect IsaacLab NPZ episodes or converted HDF5 datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def inspect_npz_dir(path: Path) -> None:
    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {path}")

    total_steps = 0
    first_pixels = None
    first_action = None
    episode_lengths = []
    disturbance_values = []
    invalid_predictions = 0
    done_count = 0
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            pixels = data["pixels"]
            action = data["action"]
            total_steps += int(pixels.shape[0])
            episode_lengths.append(int(pixels.shape[0]))
            if "disturbance" in data:
                disturbance = np.asarray(data["disturbance"])
                disturbance_values.append(disturbance[disturbance != 0])
            if "prediction_valid" in data:
                invalid_predictions += int((~np.asarray(data["prediction_valid"], dtype=np.bool_)).sum())
            if "done" in data:
                done_count += int(np.asarray(data["done"], dtype=np.bool_).sum())
            if first_pixels is None:
                first_pixels = (pixels.shape, pixels.dtype)
                first_action = (action.shape, action.dtype)

    print(f"path: {path}")
    print(f"format: npz episodes")
    print(f"episodes: {len(files)}")
    print(f"frames: {total_steps}")
    print(f"episode_len: min={min(episode_lengths)}, max={max(episode_lengths)}")
    print(f"first_pixels: shape={first_pixels[0]}, dtype={first_pixels[1]}")
    print(f"first_action: shape={first_action[0]}, dtype={first_action[1]}")
    if disturbance_values:
        nonzero = np.concatenate(disturbance_values) if any(v.size for v in disturbance_values) else np.empty(0)
        print(f"disturbances: {nonzero.size}")
        if nonzero.size:
            print(f"disturbance_abs_range: [{np.abs(nonzero).min():.4f}, {np.abs(nonzero).max():.4f}]")
        print(f"prediction_invalid: {invalid_predictions} ({invalid_predictions / total_steps:.4%})")
        print(f"done_count: {done_count}")


def inspect_h5(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        if "ep_len" in handle:
            episodes = int(len(handle["ep_len"]))
            frames = int(np.asarray(handle["ep_len"]).sum())
        else:
            episodes = -1
            frames = int(handle["pixels"].shape[0])

        print(f"path: {path}")
        print("format: hdf5")
        print(f"episodes: {episodes}")
        print(f"frames: {frames}")
        for key in handle.keys():
            value = handle[key]
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        if "disturbance" in handle:
            disturbance = np.asarray(handle["disturbance"])
            nonzero = disturbance[disturbance != 0]
            print(f"disturbances: {nonzero.size}")
            if nonzero.size:
                print(f"disturbance_abs_range: [{np.abs(nonzero).min():.4f}, {np.abs(nonzero).max():.4f}]")
        if "prediction_valid" in handle:
            valid = np.asarray(handle["prediction_valid"], dtype=np.bool_)
            invalid = int((~valid).sum())
            print(f"prediction_invalid: {invalid} ({invalid / valid.size:.4%})")
        if "done" in handle:
            print(f"done_count: {int(np.asarray(handle['done'], dtype=np.bool_).sum())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="NPZ directory, single NPZ file, or HDF5 file")
    args = parser.parse_args()

    path = args.path.expanduser()
    if path.is_dir():
        inspect_npz_dir(path)
    elif path.suffix == ".npz":
        inspect_npz_dir(path.parent)
    elif path.suffix in {".h5", ".hdf5"}:
        inspect_h5(path)
    else:
        raise ValueError(f"Unsupported path: {path}")


if __name__ == "__main__":
    main()
