#!/usr/bin/env python3
"""Collect IsaacLab episodes with a trained standalone PPO policy.

By default this uses the Cartpole RGB-camera task for pixels and constructs the
trained PPO policy input from the environment's low-dimensional Cartpole state.
This keeps the visual domain aligned with random camera-observation datasets
while actions come from the learned controller.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


DEFAULT_POLICY = (
    "/home/hall/code/RL-Learning-BasedOn-IsaacLab/logs/standalone/ppo/"
    "cartpole_swingup_centered/2026-06-30_01-24-53_reward_v2/model_200.pt"
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
parser.add_argument("--high-contrast-scene", action="store_true", help="Use a black ground, cyan cart, and yellow pole.")
parser.add_argument("--stochastic", action="store_true", help="Sample from the PPO Gaussian instead of using actor mean.")
parser.add_argument("--action-noise-std", type=float, default=0.0)
parser.add_argument(
    "--random-action-prob",
    type=float,
    default=0.0,
    help="Probability of replacing the PPO action with a uniform random action.",
)
parser.add_argument("--random-action-scale", type=float, default=1.0)
parser.add_argument(
    "--continuous-disturbance",
    action="store_true",
    help="Keep the environment running across output chunks and repeatedly disturb the pole after stabilization.",
)
parser.add_argument("--disturbance-min", type=float, default=2.4, help="Minimum pole velocity impulse in rad/s.")
parser.add_argument("--disturbance-max", type=float, default=6.0, help="Maximum pole velocity impulse in rad/s.")
parser.add_argument("--disturbance-stable-steps", type=int, default=60)
parser.add_argument("--disturbance-cooldown-steps", type=int, default=600)
parser.add_argument("--disturbance-angle-threshold", type=float, default=0.15)
parser.add_argument("--disturbance-cart-threshold", type=float, default=0.5)
parser.add_argument("--stable-pole-vel-threshold", type=float, default=0.8)
parser.add_argument("--stable-cart-vel-threshold", type=float, default=0.5)
parser.add_argument(
    "--end-on-stable-steps",
    type=int,
    default=0,
    help="For episodic collection, reset after this many stable steps; zero disables early reset.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--initial-pole-angle-range",
    type=float,
    nargs=2,
    default=(-math.pi, math.pi),
    metavar=("MIN", "MAX"),
)
parser.add_argument(
    "--env-episode-length-s",
    type=float,
    default=3600.0,
    help="Environment time limit. Output chunks are still bounded by --episode-len.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.use_render or "Camera" in args_cli.task:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
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
    if np.issubdtype(pixels_np.dtype, np.floating):
        if pixels_np.size and float(np.nanmax(pixels_np)) <= 1.0:
            pixels_np = pixels_np * 255.0
        pixels_np = np.nan_to_num(pixels_np, nan=0.0, posinf=255.0, neginf=0.0)
        pixels_np = np.clip(pixels_np, 0.0, 255.0).astype(np.uint8)
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
    policy_obs_dim = int(policy.actor[0].in_features)
    if policy_obs_dim == 5 and obs_tensor.shape[-1] == 4:
        pole_pos = obs_tensor[:, 0]
        obs_tensor = torch.stack(
            (
                torch.sin(pole_pos),
                torch.cos(pole_pos),
                obs_tensor[:, 1],
                obs_tensor[:, 2],
                obs_tensor[:, 3],
            ),
            dim=-1,
        )
    elif obs_tensor.shape[-1] != policy_obs_dim:
        raise ValueError(
            f"Policy expects {policy_obs_dim} observations, but collector produced {obs_tensor.shape[-1]}."
        )
    with torch.no_grad():
        if args_cli.stochastic:
            action, _, _ = policy.act(obs_tensor)
        else:
            action = policy.act_inference(obs_tensor)
        if args_cli.action_noise_std > 0:
            action = action + args_cli.action_noise_std * torch.randn_like(action)
    return action.clamp(-1.0, 1.0)


def _apply_pole_velocity_disturbance(env, velocity_delta: float) -> None:
    unwrapped = env.unwrapped
    cartpole = getattr(unwrapped, "_cartpole", None) or getattr(unwrapped, "cartpole", None)
    pole_idx = getattr(unwrapped, "_pole_dof_idx", None)
    if cartpole is None or pole_idx is None:
        raise AttributeError("Could not find Cartpole articulation or pole joint index.")

    joint_pos = cartpole.data.joint_pos.clone()
    joint_vel = cartpole.data.joint_vel.clone()
    joint_vel[:, pole_idx[0]] += velocity_delta
    env_ids = torch.arange(joint_pos.shape[0], device=joint_pos.device)
    cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def _apply_high_contrast_scene() -> None:
    stage = omni.usd.get_context().get_stage()
    materials = {
        "ground": (
            "/World/Looks/CollectionGroundBlack",
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.01, 0.01, 0.01), roughness=0.8),
        ),
        "cart": (
            "/World/Looks/CollectionCartCyan",
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 1.0), roughness=0.35),
        ),
        "pole": (
            "/World/Looks/CollectionPoleYellow",
            sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.9, 0.0), roughness=0.35),
        ),
    }
    for material_path, material_cfg in materials.values():
        material_cfg.func(material_path, material_cfg)

    ground = stage.GetPrimAtPath("/World/ground")
    if ground.IsValid():
        sim_utils.bind_visual_material("/World/ground", materials["ground"][0], stage=stage)

    counts = {"cart": 0, "pole": 0}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName()
        if "/Robot/" in path and name in counts:
            sim_utils.bind_visual_material(path, materials[name][0], stage=stage)
            counts[name] += 1
    print(f"[INFO] high-contrast scene: cart={counts['cart']}, pole={counts['pole']}")


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("This collector writes one NPZ per episode and requires --num-envs 1.")
    if not 0.0 <= args_cli.random_action_prob <= 1.0:
        raise ValueError("--random-action-prob must be in [0, 1].")
    if not 0.0 < args_cli.random_action_scale <= 1.0:
        raise ValueError("--random-action-scale must be in (0, 1].")
    if args_cli.end_on_stable_steps < 0:
        raise ValueError("--end-on-stable-steps must be non-negative.")
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.episode_length_s = args_cli.env_episode_length_s
    env_cfg.seed = args_cli.seed
    angle_min, angle_max = args_cli.initial_pole_angle_range
    if angle_min > angle_max:
        raise ValueError("--initial-pole-angle-range requires MIN <= MAX.")
    if hasattr(env_cfg, "initial_pole_angle_range_rad"):
        env_cfg.initial_pole_angle_range_rad = [angle_min, angle_max]
    elif hasattr(env_cfg, "initial_pole_angle_range"):
        env_cfg.initial_pole_angle_range = [angle_min / math.pi, angle_max / math.pi]
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.use_render else None)
    if args_cli.high_contrast_scene:
        _apply_high_contrast_scene()
    device = env.unwrapped.device
    policy = _load_policy(args_cli.policy_checkpoint, torch.device(device))
    rng = np.random.default_rng(args_cli.seed)

    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    start_index = args_cli.start_index if args_cli.start_index is not None else _next_episode_index(args_cli.output_dir)

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] policy={args_cli.policy_checkpoint}")
    print(f"[INFO] observation space: {env.observation_space}")
    print(f"[INFO] action space: {env.action_space}")
    print(
        "[INFO] collection plan: "
        f"episodes={args_cli.episodes}, target_frames={args_cli.target_frames}, "
        f"max_chunk_len={args_cli.episode_len}, start_index={start_index}, "
        f"continuous_disturbance={args_cli.continuous_disturbance}"
    )
    if args_cli.continuous_disturbance:
        if args_cli.disturbance_min <= 0 or args_cli.disturbance_min > args_cli.disturbance_max:
            raise ValueError("Require 0 < --disturbance-min <= --disturbance-max.")
        print(
            "[INFO] disturbance range: "
            f"[{args_cli.disturbance_min}, {args_cli.disturbance_max}] rad/s"
        )

    try:
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        total_frames = 0
        chunks_written = 0
        global_step = 0
        stable_steps = 0
        last_disturbance_step = -args_cli.disturbance_cooldown_steps
        recovery_phase = False

        while (
            total_frames < args_cli.target_frames
            if args_cli.target_frames is not None
            else chunks_written < args_cli.episodes
        ):
            ep_idx = start_index + chunks_written
            out_path = args_cli.output_dir / f"episode_{ep_idx:05d}.npz"
            if out_path.exists() and not args_cli.overwrite:
                raise FileExistsError(f"Refusing to break continuous collection by skipping {out_path}")

            pixels, actions, rewards, dones, policy_obs = [], [], [], [], []
            disturbances, stable_flags, recovery_flags, prediction_valid = [], [], [], []
            episode_stable_steps = 0
            remaining = (
                args_cli.target_frames - total_frames
                if args_cli.target_frames is not None
                else args_cli.episode_len
            )
            chunk_limit = min(args_cli.episode_len, remaining)
            for _ in range(chunk_limit):
                state_before = _get_cartpole_state(env)
                pole_angle = torch.atan2(torch.sin(state_before[:, 0]), torch.cos(state_before[:, 0]))
                is_stable_before = bool(
                    torch.all(pole_angle.abs() < args_cli.disturbance_angle_threshold)
                    and torch.all(state_before[:, 1].abs() < args_cli.stable_pole_vel_threshold)
                    and torch.all(state_before[:, 2].abs() < args_cli.disturbance_cart_threshold)
                    and torch.all(state_before[:, 3].abs() < args_cli.stable_cart_vel_threshold)
                )
                action = _policy_action(policy, env, obs, torch.device(device))
                if rng.random() < args_cli.random_action_prob:
                    action = torch.empty_like(action).uniform_(
                        -args_cli.random_action_scale,
                        args_cli.random_action_scale,
                    )
                pixels.append(_get_pixels(env, obs))
                actions.append(_first_env(action))
                policy_obs.append(_first_env(_get_policy_obs(env, obs)))
                stable_flags.append(is_stable_before)
                recovery_flags.append(recovery_phase)

                obs, reward, terminated, truncated, _ = env.step(action)
                done = torch.logical_or(terminated, truncated)
                done_now = bool(_first_env(done))
                rewards.append(_first_env(reward))
                dones.append(done_now)
                global_step += 1

                state_after = _get_cartpole_state(env)
                wrapped_angle = torch.atan2(torch.sin(state_after[:, 0]), torch.cos(state_after[:, 0]))
                is_stable_after = bool(
                    torch.all(wrapped_angle.abs() < args_cli.disturbance_angle_threshold)
                    and torch.all(state_after[:, 1].abs() < args_cli.stable_pole_vel_threshold)
                    and torch.all(state_after[:, 2].abs() < args_cli.disturbance_cart_threshold)
                    and torch.all(state_after[:, 3].abs() < args_cli.stable_cart_vel_threshold)
                )
                stable_steps = stable_steps + 1 if is_stable_after else 0
                episode_stable_steps = episode_stable_steps + 1 if is_stable_after else 0
                reached_stable_end = (
                    not args_cli.continuous_disturbance
                    and args_cli.end_on_stable_steps > 0
                    and episode_stable_steps >= args_cli.end_on_stable_steps
                )
                if reached_stable_end:
                    done_now = True
                    dones[-1] = True
                if recovery_phase and stable_steps >= args_cli.disturbance_stable_steps:
                    recovery_phase = False

                disturbance = 0.0
                disturbance_ready = (
                    args_cli.continuous_disturbance
                    and not done_now
                    and not recovery_phase
                    and stable_steps >= args_cli.disturbance_stable_steps
                    and global_step - last_disturbance_step >= args_cli.disturbance_cooldown_steps
                )
                if disturbance_ready:
                    magnitude = float(rng.uniform(args_cli.disturbance_min, args_cli.disturbance_max))
                    disturbance = magnitude if bool(rng.integers(0, 2)) else -magnitude
                    _apply_pole_velocity_disturbance(env, disturbance)
                    last_disturbance_step = global_step
                    stable_steps = 0
                    recovery_phase = True
                    print(f"[DISTURBANCE] step={global_step}, impulse={disturbance:+.3f} rad/s")

                disturbances.append(disturbance)
                prediction_valid.append(disturbance == 0.0)
                if done_now:
                    stable_steps = 0
                    recovery_phase = False
                    last_disturbance_step = global_step - args_cli.disturbance_cooldown_steps
                    break

            np.savez_compressed(
                out_path,
                pixels=np.asarray(pixels),
                action=np.asarray(actions, dtype=np.float32),
                reward=np.asarray(rewards, dtype=np.float32),
                done=np.asarray(dones, dtype=np.bool_),
                policy_obs=np.asarray(policy_obs, dtype=np.float32),
                disturbance=np.asarray(disturbances, dtype=np.float32),
                stable=np.asarray(stable_flags, dtype=np.bool_),
                recovery_phase=np.asarray(recovery_flags, dtype=np.bool_),
                prediction_valid=np.asarray(prediction_valid, dtype=np.bool_),
            )
            chunk_frames = len(pixels)
            total_frames += chunk_frames
            chunks_written += 1
            print(
                f"[INFO] wrote {out_path} "
                f"(frames={chunk_frames}, total_frames={total_frames}, done={dones[-1]})"
            )

            if not args_cli.continuous_disturbance:
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
