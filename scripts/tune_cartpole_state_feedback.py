#!/usr/bin/env python3
"""Search Cartpole state-feedback gains in parallel IsaacLab environments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

RL_SOURCE = Path("/home/hall/code/RL-Learning-BasedOn-IsaacLab/source/rl_lab_learning")
if str(RL_SOURCE) not in sys.path:
    sys.path.insert(0, str(RL_SOURCE))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="RLLab-Cartpole-SwingUp-Direct-v0")
    parser.add_argument("--num-candidates", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--disturbance-steps", type=int, nargs="+", default=[200, 500, 800])
    parser.add_argument("--disturbance-values", type=float, nargs="+", default=[-2.4, 2.6, -2.8])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/hall/code/.stable-wm/eval/cartpole_state_feedback_sweep.json"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import torch  # noqa: E402
import rl_lab_learning.tasks  # noqa: F401,E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def get_state(env) -> torch.Tensor:
    unwrapped = env.unwrapped
    cartpole = getattr(unwrapped, "_cartpole", None) or getattr(unwrapped, "cartpole", None)
    pole_idx = getattr(unwrapped, "_pole_dof_idx", None)
    cart_idx = getattr(unwrapped, "_cart_dof_idx", None)
    if cartpole is None or pole_idx is None or cart_idx is None:
        raise AttributeError("Could not find Cartpole articulation or joint indices.")
    return torch.stack(
        (
            cartpole.data.joint_pos[:, pole_idx[0]],
            cartpole.data.joint_vel[:, pole_idx[0]],
            cartpole.data.joint_pos[:, cart_idx[0]],
            cartpole.data.joint_vel[:, cart_idx[0]],
        ),
        dim=-1,
    )


def apply_disturbance(env, velocity_delta: float) -> None:
    unwrapped = env.unwrapped
    cartpole = getattr(unwrapped, "_cartpole", None) or getattr(unwrapped, "cartpole", None)
    pole_idx = getattr(unwrapped, "_pole_dof_idx", None)
    joint_pos = cartpole.data.joint_pos.clone()
    joint_vel = cartpole.data.joint_vel.clone()
    joint_vel[:, pole_idx[0]] += velocity_delta
    env_ids = torch.arange(joint_pos.shape[0], device=joint_pos.device)
    cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def sample_gains(num_candidates: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(args_cli.seed)
    ranges = torch.tensor(
        [
            [2.5, 6.0],    # pole position
            [0.5, 1.6],    # pole velocity
            [-1.2, -0.2],  # cart position
            [-1.5, -0.2],  # cart velocity
        ],
        device=device,
    )
    gains = ranges[:, 0] + torch.rand((num_candidates, 4), generator=generator, device=device) * (
        ranges[:, 1] - ranges[:, 0]
    )
    seeds = torch.tensor(
        [
            [3.6743, 0.9486, -0.5142, -0.6950],
            [4.3783, 1.1104, -0.7534, -0.9340],
            [4.2178, 1.0733, -0.6879, -0.8771],
            [3.7132, 0.9590, -0.5203, -0.7017],
        ],
        device=device,
    )
    gains[: min(len(seeds), num_candidates)] = seeds[: min(len(seeds), num_candidates)]
    return gains


def main() -> None:
    if len(args_cli.disturbance_steps) != len(args_cli.disturbance_values):
        raise ValueError("--disturbance-steps and --disturbance-values must have equal lengths.")
    num_envs = args_cli.num_candidates * args_cli.repeats
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = max(30.0, args_cli.steps / 60.0 + 2.0)
    env_cfg.initial_pole_angle_range_rad = [-0.35, 0.35]
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = torch.device(env.unwrapped.device)
    candidate_gains = sample_gains(args_cli.num_candidates, device)
    gains = candidate_gains.repeat_interleave(args_cli.repeats, dim=0)

    total_cost = torch.zeros(num_envs, device=device)
    done_count = torch.zeros(num_envs, device=device)
    upright_count = torch.zeros(num_envs, device=device)
    centered_count = torch.zeros(num_envs, device=device)
    max_angle = torch.zeros(num_envs, device=device)
    max_cart = torch.zeros(num_envs, device=device)

    env.reset()
    disturbance_map = dict(zip(args_cli.disturbance_steps, args_cli.disturbance_values))
    try:
        for step in range(args_cli.steps):
            if step in disturbance_map:
                apply_disturbance(env, disturbance_map[step])
            state = get_state(env)
            pole_pos = torch.atan2(torch.sin(state[:, 0]), torch.cos(state[:, 0]))
            normalized_state = torch.stack((pole_pos, state[:, 1], state[:, 2], state[:, 3]), dim=-1)
            action = -(gains * normalized_state).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated

            angle_abs = pole_pos.abs()
            cart_abs = state[:, 2].abs()
            step_cost = (
                12.0 * pole_pos.square()
                + 0.8 * state[:, 1].square()
                + 3.0 * state[:, 2].square()
                + 0.15 * state[:, 3].square()
                + 0.01 * action[:, 0].square()
            )
            total_cost += step_cost + 500.0 * done.float()
            done_count += done.float()
            upright_count += (angle_abs < 0.15).float()
            centered_count += ((angle_abs < 0.15) & (cart_abs < 0.8)).float()
            max_angle = torch.maximum(max_angle, angle_abs)
            max_cart = torch.maximum(max_cart, cart_abs)
    finally:
        env.close()

    def reduce(values: torch.Tensor, fn: str = "mean") -> torch.Tensor:
        grouped = values.reshape(args_cli.num_candidates, args_cli.repeats)
        return grouped.mean(dim=1) if fn == "mean" else grouped.amax(dim=1)

    metrics = {
        "cost": reduce(total_cost),
        "done_count": reduce(done_count),
        "upright_fraction": reduce(upright_count) / args_cli.steps,
        "centered_fraction": reduce(centered_count) / args_cli.steps,
        "max_angle": reduce(max_angle, "max"),
        "max_cart": reduce(max_cart, "max"),
    }
    score = (
        metrics["cost"]
        + 2000.0 * metrics["done_count"]
        + 1000.0 * (1.0 - metrics["upright_fraction"])
        + 500.0 * (1.0 - metrics["centered_fraction"])
    )
    order = torch.argsort(score)
    rows = []
    for rank, index in enumerate(order[:20].tolist(), start=1):
        row = {
            "rank": rank,
            "score": float(score[index].cpu()),
            "gains": {
                "pole_kp": float(candidate_gains[index, 0].cpu()),
                "pole_kd": float(candidate_gains[index, 1].cpu()),
                "cart_kp": float(candidate_gains[index, 2].cpu()),
                "cart_kd": float(candidate_gains[index, 3].cpu()),
            },
            **{name: float(value[index].cpu()) for name, value in metrics.items()},
        }
        rows.append(row)
        print(json.dumps(row))

    result = {
        "task": args_cli.task,
        "num_candidates": args_cli.num_candidates,
        "repeats": args_cli.repeats,
        "steps": args_cli.steps,
        "disturbances": list(disturbance_map.items()),
        "top": rows,
    }
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    args_cli.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
