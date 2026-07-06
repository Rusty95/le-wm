#!/usr/bin/env python3
"""Continuously play a standalone PPO Cartpole policy in IsaacLab.

This script does not collect or write any dataset.  It runs until the Isaac Sim
window is closed or the process receives Ctrl+C.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


DEFAULT_POLICY = Path(
    "/home/hall/code/RL-Learning-BasedOn-IsaacLab/logs/standalone/ppo/"
    "cartpole_swingup_centered/2026-06-30_01-24-53_reward_v2/model_200.pt"
)

LOCAL_REPOS = [
    Path("/home/hall/code/RL-Learning-BasedOn-IsaacLab/source/rl_lab_learning"),
]
for repo in LOCAL_REPOS:
    if repo.exists() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously play a standalone PPO Cartpole policy.")
    parser.add_argument("--task", default="Isaac-Cartpole-RGB-Camera-Direct-v0")
    parser.add_argument("--policy-checkpoint", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true", help="Sample PPO actions instead of using the actor mean.")
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument("--realtime", action="store_true", help="Sleep after each step to approximately match simulation time.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many environment steps; 0 keeps running until interrupted.",
    )
    parser.add_argument(
        "--disturbance-pole-velocity",
        type=float,
        default=0.0,
        help="Pole angular-velocity impulse in rad/s; 0 disables disturbances.",
    )
    parser.add_argument("--disturbance-count", type=int, default=3)
    parser.add_argument("--disturbance-stable-steps", type=int, default=60)
    parser.add_argument("--disturbance-cooldown-steps", type=int, default=600)
    parser.add_argument("--disturbance-angle-threshold", type=float, default=0.15)
    parser.add_argument("--disturbance-cart-threshold", type=float, default=0.5)
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=3600.0,
        help="Time-limit reset interval in simulation seconds. Physical failure still triggers a reset.",
    )
    parser.add_argument(
        "--initial-pole-angle-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-0.4, 0.4),
        help="Uniform pole-angle reset range in radians.",
    )
    parser.add_argument("--no-ground-plane", action="store_true", help="Do not add the default light grey ground plane.")
    parser.add_argument("--ground-z", type=float, default=-0.05)
    parser.add_argument("--ground-size", type=float, default=10.0)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if "Camera" in args.task:
        args.enable_cameras = True
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from omni.physx.scripts import physicsUtils  # noqa: E402
from pxr import Gf  # noqa: E402

import rl_lab_learning.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_lab_learning.algorithms.ppo import ActorCritic  # noqa: E402


def load_policy(path: Path, device: torch.device) -> ActorCritic:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model_state_dict"]
    hidden_dim = int(checkpoint.get("config", {}).get("hidden_dim", state["actor.0.weight"].shape[0]))
    obs_dim = int(state["actor.0.weight"].shape[1])
    action_dim = int(state["actor.4.weight"].shape[0])
    policy = ActorCritic(obs_dim, action_dim, hidden_dim).to(device)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def get_cartpole_state(env) -> torch.Tensor:
    unwrapped = env.unwrapped
    joint_pos = getattr(unwrapped, "joint_pos", None)
    joint_vel = getattr(unwrapped, "joint_vel", None)
    if joint_pos is None or joint_vel is None:
        cartpole = getattr(unwrapped, "_cartpole", None) or getattr(unwrapped, "cartpole", None)
        if cartpole is None:
            raise AttributeError("Could not find Cartpole joint state.")
        joint_pos = cartpole.data.joint_pos
        joint_vel = cartpole.data.joint_vel

    cart_idx = getattr(unwrapped, "_cart_dof_idx", None)
    pole_idx = getattr(unwrapped, "_pole_dof_idx", None)
    if cart_idx is None or pole_idx is None:
        raise AttributeError("Could not find Cartpole joint indices.")

    return torch.cat(
        (
            joint_pos[:, pole_idx[0]].unsqueeze(1),
            joint_vel[:, pole_idx[0]].unsqueeze(1),
            joint_pos[:, cart_idx[0]].unsqueeze(1),
            joint_vel[:, cart_idx[0]].unsqueeze(1),
        ),
        dim=-1,
    )


def policy_action(policy: ActorCritic, env, device: torch.device) -> torch.Tensor:
    state = get_cartpole_state(env).to(device).float()
    obs_dim = int(policy.actor[0].in_features)
    if obs_dim == 4:
        obs = state
    elif obs_dim == 5:
        pole_pos = state[:, 0]
        obs = torch.stack(
            (
                torch.sin(pole_pos),
                torch.cos(pole_pos),
                state[:, 1],
                state[:, 2],
                state[:, 3],
            ),
            dim=-1,
        )
    else:
        raise ValueError(f"Unsupported PPO observation dimension: {obs_dim}")
    with torch.no_grad():
        if args_cli.stochastic:
            action, _, _ = policy.act(obs)
        else:
            action = policy.act_inference(obs)
        if args_cli.action_noise_std > 0.0:
            action = action + args_cli.action_noise_std * torch.randn_like(action)
    return action.clamp(-1.0, 1.0)


def simulation_step_seconds(env) -> float:
    unwrapped = env.unwrapped
    step_dt = getattr(unwrapped, "step_dt", None)
    if step_dt is not None:
        return float(step_dt)
    cfg = getattr(unwrapped, "cfg", None)
    if cfg is not None:
        return float(cfg.sim.dt) * int(cfg.decimation)
    return 0.0


def apply_pole_velocity_disturbance(env, velocity_delta: float) -> None:
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


def add_ground_plane(path: str = "/World/PPOGroundPlane") -> None:
    stage = omni.usd.get_context().get_stage()
    existing_ground = "/World/ground"
    if stage.GetPrimAtPath(existing_ground).IsValid():
        material_path = "/World/Looks/PPOGroundBlack"
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.01, 0.01, 0.01), roughness=0.8)
        material.func(material_path, material)
        sim_utils.bind_visual_material(existing_ground, material_path, stage=stage)
        return
    if stage.GetPrimAtPath(path).IsValid():
        return
    physicsUtils.add_ground_plane(
        stage,
        path,
        "Z",
        args_cli.ground_size,
        Gf.Vec3f(0.0, 0.0, args_cli.ground_z),
        Gf.Vec3f(0.01, 0.01, 0.01),
    )


def apply_high_contrast_materials() -> None:
    stage = omni.usd.get_context().get_stage()
    cart_material_path = "/World/Looks/PPOCartCyan"
    pole_material_path = "/World/Looks/PPOPoleYellow"

    cart_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.0, 0.75, 1.0),
        roughness=0.35,
        metallic=0.05,
    )
    pole_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 0.9, 0.0),
        roughness=0.35,
        metallic=0.05,
    )
    cart_material.func(cart_material_path, cart_material)
    pole_material.func(pole_material_path, pole_material)

    cart_paths = []
    pole_paths = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/Robot/" not in path:
            continue
        if prim.GetName() == "cart":
            cart_paths.append(path)
        elif prim.GetName() == "pole":
            pole_paths.append(path)

    for path in cart_paths:
        sim_utils.bind_visual_material(path, cart_material_path, stage=stage)
    for path in pole_paths:
        sim_utils.bind_visual_material(path, pole_material_path, stage=stage)

    print(f"[INFO] high-contrast materials: cart={len(cart_paths)}, pole={len(pole_paths)}")


def main() -> None:
    if not args_cli.policy_checkpoint.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {args_cli.policy_checkpoint}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.episode_length_s = args_cli.episode_length_s
    angle_min, angle_max = args_cli.initial_pole_angle_range
    if angle_min > angle_max:
        raise ValueError("--initial-pole-angle-range requires MIN <= MAX.")
    if hasattr(env_cfg, "initial_pole_angle_range_rad"):
        env_cfg.initial_pole_angle_range_rad = [angle_min, angle_max]
    else:
        # IsaacLab's built-in Cartpole config stores fractions of pi even
        # though the field name does not make that unit explicit.
        env_cfg.initial_pole_angle_range = [angle_min / math.pi, angle_max / math.pi]
    env = gym.make(args_cli.task, cfg=env_cfg)
    if not args_cli.no_ground_plane:
        add_ground_plane()
    apply_high_contrast_materials()
    device = torch.device(env.unwrapped.device)
    policy = load_policy(args_cli.policy_checkpoint, device)
    step_seconds = simulation_step_seconds(env)

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] policy={args_cli.policy_checkpoint}")
    print(f"[INFO] device={device}, num_envs={args_cli.num_envs}")
    print(f"[INFO] initial_pole_angle_range=[{angle_min}, {angle_max}] rad")
    print("[INFO] No data will be saved. Close Isaac Sim or press Ctrl+C to stop.")

    reset_out: Any = env.reset()
    _ = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    step_count = 0
    reset_count = 0
    reward_sum = 0.0
    abs_angle_sum = 0.0
    upright_steps = 0
    max_abs_cart_pos = 0.0
    stable_steps = 0
    disturbance_count = 0
    last_disturbance_step = -args_cli.disturbance_cooldown_steps
    awaiting_recovery = False
    recovery_steps: list[int] = []
    try:
        while simulation_app.is_running():
            started = time.perf_counter()
            action = policy_action(policy, env, device)
            _, reward, terminated, truncated, _ = env.step(action)
            state = get_cartpole_state(env)
            wrapped_angle = torch.atan2(torch.sin(state[:, 0]), torch.cos(state[:, 0]))
            step_count += 1
            reset_count += int(torch.count_nonzero(terminated | truncated).item())
            reward_sum += float(reward.mean().item())
            abs_angle_sum += float(wrapped_angle.abs().mean().item())
            upright_steps += int((wrapped_angle.abs() < 0.2).sum().item())
            max_abs_cart_pos = max(max_abs_cart_pos, float(state[:, 2].abs().max().item()))

            is_stable = bool(
                torch.all(wrapped_angle.abs() < args_cli.disturbance_angle_threshold)
                and torch.all(state[:, 2].abs() < args_cli.disturbance_cart_threshold)
            )
            stable_steps = stable_steps + 1 if is_stable else 0
            if awaiting_recovery and stable_steps >= args_cli.disturbance_stable_steps:
                recovery = step_count - last_disturbance_step
                recovery_steps.append(recovery)
                awaiting_recovery = False
                print(f"[DISTURBANCE] recovered after {recovery} steps")

            disturbance_ready = (
                args_cli.disturbance_pole_velocity > 0.0
                and disturbance_count < args_cli.disturbance_count
                and not awaiting_recovery
                and stable_steps >= args_cli.disturbance_stable_steps
                and step_count - last_disturbance_step >= args_cli.disturbance_cooldown_steps
            )
            if disturbance_ready:
                direction = 1.0 if disturbance_count % 2 == 0 else -1.0
                velocity_delta = direction * args_cli.disturbance_pole_velocity
                apply_pole_velocity_disturbance(env, velocity_delta)
                disturbance_count += 1
                last_disturbance_step = step_count
                awaiting_recovery = True
                stable_steps = 0
                print(
                    f"[DISTURBANCE] step={step_count}, impulse={velocity_delta:+.3f} rad/s, "
                    f"count={disturbance_count}/{args_cli.disturbance_count}"
                )
            if args_cli.realtime and step_seconds > 0.0:
                elapsed = time.perf_counter() - started
                time.sleep(max(0.0, step_seconds - elapsed))
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break
    except KeyboardInterrupt:
        print("\n[INFO] PPO playback stopped.")
    finally:
        if step_count:
            print(
                "[RESULT] "
                f"steps={step_count}, resets={reset_count}, "
                f"mean_reward={reward_sum / step_count:.4f}, "
                f"mean_abs_pole_angle={abs_angle_sum / step_count:.4f} rad, "
                f"upright_ratio={upright_steps / (step_count * args_cli.num_envs):.3f}, "
                f"max_abs_cart_pos={max_abs_cart_pos:.4f}, "
                f"disturbances={disturbance_count}, recoveries={recovery_steps}"
            )
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
