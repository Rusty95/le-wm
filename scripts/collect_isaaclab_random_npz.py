#!/usr/bin/env python3
"""Collect random-policy IsaacLab episodes as NPZ files for LeWM.

Run this script from the IsaacLab environment, for example:

  source /home/hall/code/activate_isaaclab.sh
  python le-wm/scripts/collect_isaaclab_random_npz.py \
      --task Isaac-Cartpole-RGB-Camera-Direct-v0 \
      --episodes 4 --episode-len 80 --output-dir /tmp/isaaclab_lewm_npz \
      --headless --enable_cameras

The default pixel source is obs["policy"], which matches IsaacLab camera
tasks such as Cartpole RGB camera.  For non-camera tasks, pass
``--use-render`` and create the environment with ``render_mode=rgb_array``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect IsaacLab random-policy episodes for LeWM.")
parser.add_argument("--task", type=str, required=True, help="IsaacLab gym task id")
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--episode-len", type=int, default=80)
parser.add_argument(
    "--target-frames",
    type=int,
    default=None,
    help="Optional total frame target. Overrides --episodes with ceil(target_frames / episode_len).",
)
parser.add_argument(
    "--start-index",
    type=int,
    default=None,
    help="First episode index to write. Defaults to the next available episode_*.npz index.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing episode files instead of skipping them.",
)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--pixel-key", type=str, default="policy", help="Observation key to use as pixels")
parser.add_argument("--use-render", action="store_true", help="Use env.render() instead of an observation key")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.use_render:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_env(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim > 0 and args_cli.num_envs == 1:
        return arr[0]
    return arr


def _get_pixels(env, obs: Any) -> np.ndarray:
    if args_cli.use_render:
        frame = env.render()
        return np.asarray(frame)

    if isinstance(obs, dict):
        if args_cli.pixel_key not in obs:
            raise KeyError(
                f"Observation dict has keys {list(obs.keys())}, "
                f"but --pixel-key={args_cli.pixel_key!r} was requested."
            )
        pixels = obs[args_cli.pixel_key]
    else:
        pixels = obs

    pixels_np = _first_env(pixels)
    if pixels_np.ndim == 3 and pixels_np.shape[0] in (1, 3, 4):
        pixels_np = np.moveaxis(pixels_np, 0, -1)
    if pixels_np.shape[-1] == 4:
        pixels_np = pixels_np[..., :3]
    return pixels_np


def _sample_actions(env) -> torch.Tensor:
    shape = env.action_space.shape
    return 2.0 * torch.rand(shape, device=env.unwrapped.device) - 1.0


def _next_episode_index(output_dir: Path) -> int:
    indices = []
    for path in output_dir.glob("episode_*.npz"):
        stem = path.stem
        try:
            indices.append(int(stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(indices, default=-1) + 1


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError(
            "This v1 collector writes one NPZ per episode and currently "
            "requires --num-envs 1."
        )

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    render_mode = "rgb_array" if args_cli.use_render else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = args_cli.episodes
    if args_cli.target_frames is not None:
        episodes = int(np.ceil(args_cli.target_frames / args_cli.episode_len))
    start_index = args_cli.start_index
    if start_index is None:
        start_index = _next_episode_index(args_cli.output_dir)

    print(f"[INFO] observation space: {env.observation_space}")
    print(f"[INFO] action space: {env.action_space}")
    print(
        "[INFO] collection plan: "
        f"episodes={episodes}, episode_len={args_cli.episode_len}, "
        f"target_frames={episodes * args_cli.episode_len}, start_index={start_index}"
    )

    try:
        written = 0
        skipped = 0
        for local_ep_idx in range(episodes):
            ep_idx = start_index + local_ep_idx
            out_path = args_cli.output_dir / f"episode_{ep_idx:05d}.npz"
            if out_path.exists() and not args_cli.overwrite:
                skipped += 1
                print(f"[INFO] skip existing {out_path}")
                continue

            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

            pixels, actions, rewards, dones = [], [], [], []
            for _ in range(args_cli.episode_len):
                action = _sample_actions(env)
                pixels.append(_get_pixels(env, obs))
                actions.append(_first_env(action))

                obs, reward, terminated, truncated, _ = env.step(action)
                done = torch.logical_or(terminated, truncated)
                rewards.append(_first_env(reward))
                dones.append(_first_env(done))

            np.savez_compressed(
                out_path,
                pixels=np.asarray(pixels),
                action=np.asarray(actions, dtype=np.float32),
                reward=np.asarray(rewards, dtype=np.float32),
                done=np.asarray(dones, dtype=np.bool_),
            )
            written += 1
            frames_done = (written + skipped) * args_cli.episode_len
            print(
                f"[INFO] wrote {out_path} "
                f"({written} written, {skipped} skipped, approx_frames={frames_done})"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
