#!/usr/bin/env python3
"""Report angular and action coverage of a Cartpole HDF5 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--bins", type=int, default=12)
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as handle:
        obs = np.asarray(handle["policy_obs"][:], dtype=np.float32)
        actions = np.asarray(handle["action"][:], dtype=np.float32)
        lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)

    angles = np.arctan2(np.sin(obs[:, 0]), np.cos(obs[:, 0]))
    hist, edges = np.histogram(angles, bins=args.bins, range=(-np.pi, np.pi))
    initial_angles = angles[offsets]
    initial_hist, _ = np.histogram(initial_angles, bins=args.bins, range=(-np.pi, np.pi))

    print(f"dataset: {args.dataset}")
    print(f"episodes: {len(lengths)}, frames: {len(angles)}")
    print(f"angle range: [{angles.min():.4f}, {angles.max():.4f}] rad")
    print(f"near upright |theta|<0.25: {(np.abs(angles) < 0.25).mean():.4%}")
    print(f"near bottom |theta|>2.75: {(np.abs(angles) > 2.75).mean():.4%}")
    print(f"action mean/std: {actions.mean():.4f} / {actions.std():.4f}")
    print(f"action saturation |a|>0.95: {(np.abs(actions) > 0.95).mean():.4%}")
    print("angle bins:")
    for left, right, count, initial_count in zip(edges[:-1], edges[1:], hist, initial_hist):
        print(f"  [{left:+.3f}, {right:+.3f}): frames={count:7d}, resets={initial_count:4d}")


if __name__ == "__main__":
    main()
