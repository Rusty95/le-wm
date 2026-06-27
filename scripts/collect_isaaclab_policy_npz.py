#!/usr/bin/env python3
"""Collect IsaacLab episodes with a trained standalone PPO policy.

By default this uses the Cartpole RGB-camera task for pixels and constructs the
trained PPO policy input from the environment's low-dimensional Cartpole state.
This keeps the visual domain aligned with random camera-observation datasets
while actions come from the learned controller.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


DEFAULT_POLICY = (
    "/home/hall/code/RL-Learning-BasedOn-IsaacLab/logs/standalone/ppo/cartpole/"
    "2026-06-07_21-41-11/model_149.pt"
)
RL_REPO = Path("/home/hall/code/RL-Learning-BasedOn-IsaacLab/source/rl_lab_learning")
if str(RL_REPO) not in sys.path:
    sys.path.insert(0, str(RL_REPO))


parser = argparse.ArgumentParser(description="Collect IsaacLab LeWM data with a PPO policy.")
parser.add_argument("--task", type=str, default="Isaac-Cartpole-RGB-Camera-Direct-v0")
parser.add_argument("--policy-checkpoint", type=Path, default=Path(DEFAULT_POLICY))
parser.add_argument("--episodes", type=int, default=625)
parser.add_argument("--episode-len", type=int, default=80)
parser.add_argument("--target-frames", type=int, default=None)
parser.add_argument("--start-index", type=int, default=None)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--output-dir", type=Path, default=Path("/home/hall/code/.stable-wm/isaaclab_policy_npz_test"))
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--pixel-key", type=str, default="policy")
parser.add_argument("--obs-key", type=str, default="policy")
parser.add_argument(
    "--policy-obs-source",
    choices=["cartpole-state", "obs"],
    default="cartpole-state",
    help="Use env internals for the trained 4D Cartpole policy obs, or read --obs-key from obs.",
)
parser.add_argument("--use-render", action="store_true", help="Use env.render() instead of camera observation pixels.")
parser.add_argument("--image-size", type=int, default=224, help="Resize rendered pixels to a square size. Use 0 to keep native render size.")
parser.add_argument("--stochastic", action="store_true", help="Sample from the PPO Gaussian instead of using actor mean.")
parser.add_argument("--action-noise-std", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.use_render or "Camera" in args_cli.task:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import rl_lab_learning.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_lab_learning.algorithms.ppo import ActorCritic  # noqa: E402


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_env(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim > 0 and args_cli.num_envs == 1:
        return arr[0]
    return arr


def _get_nested(obs: Any, key: str) -> Any:
    if isinstance(obs, dict):
        if key not in obs:
            raise KeyError(f"Observation dict has keys {list(obs.keys())}, missing {key!r}")
        return obs[key]
    return obs


def _get_cartpole_state(env) -> torch.Tensor:
    unwrapped = env.unwrapped
    joint_pos = getattr(unwrapped, "joint_pos", None)
    joint_vel = getattr(unwrapped, "joint_vel", None)
    if joint_pos is None or joint_vel is None:
        cartpole = getattr(unwrapped, "_cartpole", None) or getattr(unwrapped, "cartpole", None)
        if cartpole is None:
            raise AttributeError("Could not find Cartpole joint state on env.unwrapped.")
        joint_pos = cartpole.data.joint_pos
        joint_vel = cartpole.data.joint_vel

    cart_idx = getattr(unwrapped, "_cart_dof_idx", None)
    pole_idx = getattr(unwrapped, "_pole_dof_idx", None)
    if cart_idx is None or pole_idx is None:
        raise AttributeError("Could not find Cartpole joint indices on env.unwrapped.")

    return torch.cat(
        (
            joint_pos[:, pole_idx[0]].unsqueeze(1),
            joint_vel[:, pole_idx[0]].unsqueeze(1),
            joint_pos[:, cart_idx[0]].unsqueeze(1),
            joint_vel[:, cart_idx[0]].unsqueeze(1),
        ),
        dim=-1,
    )


def _get_pixels(env, obs: Any) -> np.ndarray:
    if args_cli.use_render:
        pixels_np = np.asarray(env.render())
        if args_cli.image_size and (pixels_np.shape[0] != args_cli.image_size or pixels_np.shape[1] != args_cli.image_size):
            pixels_t = torch.as_tensor(pixels_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            pixels_t = torch.nn.functional.interpolate(
                pixels_t,
                size=(args_cli.image_size, args_cli.image_size),
                mode="bilinear",
                align_corners=False,
            )
            pixels_np = pixels_t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
        return pixels_np
    pixels_np = _first_env(_get_nested(obs, args_cli.pixel_key))
    if pixels_np.ndim == 3 and pixels_np.shape[0] in (1, 3, 4):
        pixels_np = np.moveaxis(pixels_np, 0, -1)
    if pixels_np.shape[-1] == 4:
        pixels_np = pixels_np[..., :3]
    return pixels_np


def _next_episode_index(output_dir: Path) -> int:
    indices = []
    for path in output_dir.glob("episode_*.npz"):
        try:
            indices.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(indices, default=-1) + 1


def _load_policy(path: Path, device: torch.device) -> ActorCritic:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model_state_dict"]
    hidden_dim = int(checkpoint.get("config", {}).get("hidden_dim", state["actor.0.weight"].shape[0]))
    obs_dim = int(state["actor.0.weight"].shape[1])
    action_dim = int(state["actor.4.weight"].shape[0])
    policy = ActorCritic(obs_dim, action_dim, hidden_dim).to(device)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def _get_policy_obs(env, obs: Any) -> torch.Tensor:
    if args_cli.policy_obs_source == "cartpole-state":
        return _get_cartpole_state(env)
    return _get_nested(obs, args_cli.obs_key)


def _policy_action(policy: ActorCritic, env, obs: Any, device: torch.device) -> torch.Tensor:
    policy_obs = _get_policy_obs(env, obs)
    obs_tensor = policy_obs if isinstance(policy_obs, torch.Tensor) else torch.as_tensor(policy_obs)
    obs_tensor = obs_tensor.to(device).float()
    with torch.no_grad():
        if args_cli.stochastic:
            action, _, _ = policy.act(obs_tensor)
        else:
            action = policy.act_inference(obs_tensor)
        if args_cli.action_noise_std > 0:
            action = action + args_cli.action_noise_std * torch.randn_like(action)
    return action.clamp(-1.0, 1.0)


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("This collector writes one NPZ per episode and requires --num-envs 1.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.use_render else None)
    device = env.unwrapped.device
    policy = _load_policy(args_cli.policy_checkpoint, torch.device(device))

    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = args_cli.episodes
    if args_cli.target_frames is not None:
        episodes = int(np.ceil(args_cli.target_frames / args_cli.episode_len))
    start_index = args_cli.start_index if args_cli.start_index is not None else _next_episode_index(args_cli.output_dir)

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] policy={args_cli.policy_checkpoint}")
    print(f"[INFO] observation space: {env.observation_space}")
    print(f"[INFO] action space: {env.action_space}")
    print(f"[INFO] collection plan: episodes={episodes}, episode_len={args_cli.episode_len}, start_index={start_index}")

    try:
        for local_ep_idx in range(episodes):
            ep_idx = start_index + local_ep_idx
            out_path = args_cli.output_dir / f"episode_{ep_idx:05d}.npz"
            if out_path.exists() and not args_cli.overwrite:
                print(f"[INFO] skip existing {out_path}")
                continue

            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            pixels, actions, rewards, dones, policy_obs = [], [], [], [], []
            for _ in range(args_cli.episode_len):
                action = _policy_action(policy, env, obs, torch.device(device))
                pixels.append(_get_pixels(env, obs))
                actions.append(_first_env(action))
                policy_obs.append(_first_env(_get_policy_obs(env, obs)))

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
                policy_obs=np.asarray(policy_obs, dtype=np.float32),
            )
            frames_done = (local_ep_idx + 1) * args_cli.episode_len
            print(f"[INFO] wrote {out_path} (approx_new_frames={frames_done})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
