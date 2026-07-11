#!/usr/bin/env python3
"""Run Cartpole in IsaacLab with LeWM encoder + latent policy head.

This script does not load PPO.  It uses a frozen LeWM encoder and a small
behavior-cloned policy head trained on LeWM latent histories.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def parse_args() -> argparse.Namespace:
    from scripts.lewm_isaaclab_common import DEFAULT_ACTION_STATS_H5, DEFAULT_CACHE_DIR

    parser = argparse.ArgumentParser(description="LeWM latent-policy controller for IsaacLab Cartpole.")
    parser.add_argument("--task", type=str, default="RLLab-Cartpole-SwingUp-RGB-Camera-Direct-v0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy-head", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--action-stats-h5", type=Path, nargs="+", default=DEFAULT_ACTION_STATS_H5)
    parser.add_argument("--max-stats-rows-per-file", type=int, default=50000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-len", type=int, default=300)
    parser.add_argument("--episode-length-s", type=float, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--pixel-key", type=str, default="policy")
    parser.add_argument("--use-render", action="store_true")
    parser.add_argument("--high-contrast-scene", action="store_true")
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--initial-pole-angle-range", type=float, nargs=2, default=(-0.25, 0.25))
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--action-low", type=float, default=-1.0)
    parser.add_argument("--action-high", type=float, default=1.0)
    parser.add_argument("--disturbance-start-step", type=int, default=-1, help="First eligible disturbance step; negative disables it.")
    parser.add_argument("--disturbance-interval", type=int, default=100)
    parser.add_argument("--disturbance-count", type=int, default=2)
    parser.add_argument("--disturbance-min", type=float, default=2.4)
    parser.add_argument("--disturbance-max", type=float, default=6.0)
    parser.add_argument("--disturbance-stable-steps", type=int, default=60)
    parser.add_argument("--disturbance-angle-threshold", type=float, default=0.15)
    parser.add_argument("--disturbance-pole-vel-threshold", type=float, default=0.8)
    parser.add_argument("--disturbance-cart-threshold", type=float, default=0.8)
    parser.add_argument("--disturbance-cart-vel-threshold", type=float, default=0.5)
    parser.add_argument(
        "--disturbance-first-immediate",
        action="store_true",
        help="Apply the first disturbance at start-step; later disturbances still require recovery.",
    )
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--gif-out", type=Path, default=Path("/home/hall/code/.stable-wm/visualizations/lewm_latent_policy_cartpole.gif"))
    parser.add_argument("--out", type=Path, default=Path("/home/hall/code/.stable-wm/eval/lewm_latent_policy_cartpole_eval.json"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.use_render or "Camera" in args.task:
        args.enable_cameras = True
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
import rl_lab_learning.tasks  # noqa: F401,E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from scripts.lewm_isaaclab_common import (  # noqa: E402
    first_env,
    get_cartpole_state,
    get_pixels,
    load_action_stats,
    load_latent_policy_head,
    load_lewm,
    preprocess_pixels,
)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def apply_high_contrast_scene() -> None:
    stage = omni.usd.get_context().get_stage()
    materials = {
        "ground": ("/World/Looks/LeWMGroundBlack", sim_utils.PreviewSurfaceCfg(diffuse_color=(0.01, 0.01, 0.01), roughness=0.8)),
        "cart": ("/World/Looks/LeWMCartCyan", sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 1.0), roughness=0.35)),
        "pole": ("/World/Looks/LeWMPoleYellow", sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.9, 0.0), roughness=0.35)),
    }
    for material_path, material_cfg in materials.values():
        material_cfg.func(material_path, material_cfg)
    if stage.GetPrimAtPath("/World/ground").IsValid():
        sim_utils.bind_visual_material("/World/ground", materials["ground"][0], stage=stage)
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName()
        if "/Robot/" in path and name in ("cart", "pole"):
            sim_utils.bind_visual_material(path, materials[name][0], stage=stage)


def read_pixels(env, obs) -> np.ndarray:
    if args_cli.use_render:
        pixels = np.asarray(env.render())
        if pixels.ndim == 4:
            pixels = pixels[0]
        if pixels.shape[-1] == 4:
            pixels = pixels[..., :3]
        return pixels
    return get_pixels(obs, args_cli.pixel_key, args_cli.num_envs)


def visualize_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame[..., :3]
    frame = frame.astype(np.float32)
    finite = np.isfinite(frame)
    if not finite.any():
        return np.zeros(frame.shape, dtype=np.uint8)
    lo = float(frame[finite].min())
    hi = float(frame[finite].max())
    return np.clip((frame - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)[..., :3]


def choose_policy_action(
    model: torch.nn.Module,
    policy: torch.nn.Module,
    pixel_history: deque[np.ndarray],
    img_size: int,
    device: torch.device,
) -> torch.Tensor:
    pixels = preprocess_pixels(list(pixel_history), img_size=img_size, device=device).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode({"pixels": pixels})["emb"].reshape(1, -1)
        action = policy(emb).reshape(-1)
    action = action * args_cli.action_scale
    return action.clamp(args_cli.action_low, args_cli.action_high)


def apply_pole_velocity_disturbance(env: Any, velocity_delta: float) -> None:
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


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("LeWM latent policy v1 requires --num-envs 1.")
    if args_cli.disturbance_interval <= 0:
        raise ValueError("--disturbance-interval must be positive.")
    if args_cli.disturbance_count < 0:
        raise ValueError("--disturbance-count must be non-negative.")
    if args_cli.disturbance_stable_steps <= 0:
        raise ValueError("--disturbance-stable-steps must be positive.")
    if not 0.0 < args_cli.disturbance_min <= args_cli.disturbance_max:
        raise ValueError("Require 0 < --disturbance-min <= --disturbance-max.")
    if args_cli.seed is not None:
        np.random.seed(args_cli.seed)
        torch.manual_seed(args_cli.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args_cli.seed)
    disturbance_rng = np.random.default_rng(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    angle_min, angle_max = args_cli.initial_pole_angle_range
    if hasattr(env_cfg, "initial_pole_angle_range_rad"):
        env_cfg.initial_pole_angle_range_rad = [angle_min, angle_max]

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.use_render else None)
    if args_cli.high_contrast_scene:
        apply_high_contrast_scene()

    device = resolve_device(args_cli.device)
    action_mean, _ = load_action_stats(args_cli.action_stats_h5, args_cli.max_stats_rows_per_file)
    model = load_lewm(args_cli.checkpoint, args_cli.cache_dir, int(action_mean.shape[-1]), args_cli.img_size, device)
    policy = load_latent_policy_head(args_cli.policy_head, device)
    history_size = int(getattr(policy, "config", {}).get("history_size", 3))

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] checkpoint={args_cli.checkpoint}")
    print(f"[INFO] policy_head={args_cli.policy_head}")
    print(f"[INFO] device={device}, history_size={history_size}")
    print("[INFO] PPO policy is not loaded; actions are selected by LeWM latent policy head.")

    episodes = []
    gif_frames = []
    try:
        for ep_idx in range(args_cli.episodes):
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            first_pixel = read_pixels(env, obs)
            pixel_history: deque[np.ndarray] = deque([first_pixel.copy() for _ in range(history_size)], maxlen=history_size)
            rewards = []
            dones = []
            states = []
            actions = []
            disturbance_events = []
            disturbances_applied = 0
            stable_steps = 0
            awaiting_recovery = False
            last_disturbance_step = -args_cli.disturbance_interval
            survival_steps = 0
            for step in range(args_cli.episode_len):
                state = first_env(get_cartpole_state(env), args_cli.num_envs).astype(np.float32)
                wrapped_angle = float(np.arctan2(np.sin(state[0]), np.cos(state[0])))
                is_stable = (
                    abs(wrapped_angle) < args_cli.disturbance_angle_threshold
                    and abs(float(state[1])) < args_cli.disturbance_pole_vel_threshold
                    and abs(float(state[2])) < args_cli.disturbance_cart_threshold
                    and abs(float(state[3])) < args_cli.disturbance_cart_vel_threshold
                )
                stable_steps = stable_steps + 1 if is_stable else 0
                if awaiting_recovery and stable_steps >= args_cli.disturbance_stable_steps:
                    recovery_steps = step - last_disturbance_step
                    disturbance_events[-1]["recovery_step"] = step
                    disturbance_events[-1]["recovery_steps"] = recovery_steps
                    awaiting_recovery = False
                    print(f"[DISTURBANCE] episode={ep_idx} recovered after {recovery_steps} steps")

                first_immediate_due = (
                    args_cli.disturbance_first_immediate
                    and disturbances_applied == 0
                    and args_cli.disturbance_start_step >= 0
                    and step >= args_cli.disturbance_start_step
                    and disturbances_applied < args_cli.disturbance_count
                )
                recovered_disturbance_due = (
                    args_cli.disturbance_start_step >= 0
                    and step >= args_cli.disturbance_start_step
                    and disturbances_applied < args_cli.disturbance_count
                    and not awaiting_recovery
                    and stable_steps >= args_cli.disturbance_stable_steps
                    and step - last_disturbance_step >= args_cli.disturbance_interval
                )
                disturbance = 0.0
                if first_immediate_due or recovered_disturbance_due:
                    magnitude = float(disturbance_rng.uniform(args_cli.disturbance_min, args_cli.disturbance_max))
                    disturbance = magnitude if bool(disturbance_rng.integers(0, 2)) else -magnitude
                    apply_pole_velocity_disturbance(env, disturbance)
                    disturbances_applied += 1
                    last_disturbance_step = step
                    awaiting_recovery = True
                    stable_steps = 0
                    disturbance_events.append(
                        {
                            "step": step,
                            "pole_velocity_delta": disturbance,
                            "recovery_step": None,
                            "recovery_steps": None,
                            "state_before": state.tolist(),
                        }
                    )
                    print(
                        f"[DISTURBANCE] episode={ep_idx} step={step} "
                        f"pole_velocity_delta={disturbance:+.3f} rad/s"
                    )
                    state = first_env(get_cartpole_state(env), args_cli.num_envs).astype(np.float32)
                states.append(state)
                action = choose_policy_action(model, policy, pixel_history, args_cli.img_size, device)
                action_env = action.reshape(1, -1).to(env.unwrapped.device)
                obs, reward, terminated, truncated, _ = env.step(action_env)
                done = torch.logical_or(terminated, truncated)
                pixel = read_pixels(env, obs)
                pixel_history.append(pixel)
                action_np = action.detach().cpu().numpy().astype(np.float32)
                actions.append(action_np)
                reward_float = float(np.asarray(first_env(reward, args_cli.num_envs)).reshape(-1)[0])
                done_bool = bool(np.asarray(first_env(done, args_cli.num_envs)).reshape(-1)[0])
                rewards.append(reward_float)
                dones.append(done_bool)
                survival_steps += 1
                if args_cli.save_gif and ep_idx == 0:
                    gif_frames.append(visualize_frame(pixel))
                if done_bool:
                    break
            states_np = np.asarray(states, dtype=np.float32)
            actions_np = np.asarray(actions, dtype=np.float32)
            episode = {
                "episode": ep_idx,
                "reward_sum": float(np.sum(rewards)),
                "survival_steps": int(survival_steps),
                "done_count": int(np.sum(dones)),
                "mean_abs_pole_angle": float(np.abs(np.arctan2(np.sin(states_np[:, 0]), np.cos(states_np[:, 0]))).mean()) if len(states_np) else None,
                "max_abs_pole_angle": float(np.abs(np.arctan2(np.sin(states_np[:, 0]), np.cos(states_np[:, 0]))).max()) if len(states_np) else None,
                "mean_abs_cart_pos": float(np.abs(states_np[:, 2]).mean()) if len(states_np) else None,
                "mean_abs_action": float(np.abs(actions_np).mean()) if len(actions_np) else None,
                "disturbances": disturbance_events,
            }
            episodes.append(episode)
            print(f"[INFO] episode={ep_idx} survival={survival_steps} reward={episode['reward_sum']:.3f}")
    finally:
        env.close()

    result = {
        "mode": "lewm_latent_policy_cartpole",
        "task": args_cli.task,
        "seed": args_cli.seed,
        "checkpoint": args_cli.checkpoint,
        "policy_head": str(args_cli.policy_head),
        "episodes": episodes,
        "summary": {
            "episodes": len(episodes),
            "mean_reward_sum": float(np.mean([ep["reward_sum"] for ep in episodes])) if episodes else None,
            "mean_survival_steps": float(np.mean([ep["survival_steps"] for ep in episodes])) if episodes else None,
            "mean_abs_pole_angle": float(np.mean([ep["mean_abs_pole_angle"] for ep in episodes])) if episodes else None,
            "mean_abs_cart_pos": float(np.mean([ep["mean_abs_cart_pos"] for ep in episodes])) if episodes else None,
            "terminated_episodes": int(np.sum([ep["done_count"] > 0 for ep in episodes])),
        },
        "disturbance": {
            "start_step": args_cli.disturbance_start_step,
            "interval": args_cli.disturbance_interval,
            "count": args_cli.disturbance_count,
            "min": args_cli.disturbance_min,
            "max": args_cli.disturbance_max,
            "stable_steps": args_cli.disturbance_stable_steps,
            "angle_threshold": args_cli.disturbance_angle_threshold,
            "pole_vel_threshold": args_cli.disturbance_pole_vel_threshold,
            "cart_threshold": args_cli.disturbance_cart_threshold,
            "cart_vel_threshold": args_cli.disturbance_cart_vel_threshold,
            "first_immediate": args_cli.disturbance_first_immediate,
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    args_cli.out.write_text(text + "\n", encoding="utf-8")
    print(f"[INFO] wrote {args_cli.out}")

    if args_cli.save_gif and gif_frames:
        args_cli.gif_out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args_cli.gif_out, gif_frames, duration=1 / 10)
        print(f"[INFO] wrote {args_cli.gif_out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
