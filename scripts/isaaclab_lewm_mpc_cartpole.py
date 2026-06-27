#!/usr/bin/env python3
"""Control IsaacLab Cartpole with LeWM + state probe + CEM MPC.

No PPO policy is loaded.  PPO-collected data is only used offline through the
trained LeWM checkpoint, action normalizer, and latent-to-state probe.
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
    from scripts.lewm_isaaclab_common import DEFAULT_ACTION_STATS_H5, DEFAULT_CACHE_DIR, DEFAULT_CHECKPOINT

    parser = argparse.ArgumentParser(description="LeWM-only MPC controller for IsaacLab Cartpole.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-RGB-Camera-Direct-v0")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--probe", type=Path, default=Path("/home/hall/code/.stable-wm/checkpoints/lewm_cartpole_state_probe.pt"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--action-stats-h5", type=Path, nargs="+", default=DEFAULT_ACTION_STATS_H5)
    parser.add_argument("--max-stats-rows-per-file", type=int, default=50000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-len", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--num-candidates", type=int, default=512)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--cem-iters", type=int, default=3)
    parser.add_argument("--init-std", type=float, default=0.7)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--action-low", type=float, default=-1.0)
    parser.add_argument("--action-high", type=float, default=1.0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--pixel-key", type=str, default="policy")
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--gif-out", type=Path, default=Path("/home/hall/code/.stable-wm/visualizations/lewm_mpc_cartpole.gif"))
    parser.add_argument("--out", type=Path, default=Path("/home/hall/code/.stable-wm/eval/lewm_mpc_cartpole_eval.json"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if "Camera" in args.task:
        args.enable_cameras = True
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from scripts.lewm_isaaclab_common import (  # noqa: E402
    first_env,
    get_cartpole_state,
    get_pixels,
    infer_history_size,
    load_action_stats,
    load_lewm,
    load_state_probe,
    normalize_actions,
    preprocess_pixels,
    rollout_predictions,
)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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


def predict_candidate_states(
    model: torch.nn.Module,
    probe: torch.nn.Module,
    pixel_history: deque[np.ndarray],
    action_history: deque[np.ndarray],
    candidates_raw: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    img_size: int,
    history_size: int,
    device: torch.device,
) -> torch.Tensor:
    pixels = preprocess_pixels(list(pixel_history), img_size=img_size, device=device).unsqueeze(0)
    past_actions = torch.as_tensor(np.asarray(action_history), dtype=torch.float32)
    if past_actions.ndim == 1:
        past_actions = past_actions.unsqueeze(-1)
    past_actions = past_actions.to(device)
    num_candidates = candidates_raw.shape[0]

    with torch.no_grad():
        emb = model.encode({"pixels": pixels})["emb"].expand(num_candidates, -1, -1).contiguous()
        past = past_actions.unsqueeze(0).expand(num_candidates, -1, -1)
        full_actions_raw = torch.cat([past, candidates_raw], dim=1)
        full_actions = normalize_actions(full_actions_raw, action_mean, action_std, device=device)
        act_emb = model.action_encoder(full_actions)
        pred_emb = rollout_predictions(model, emb, act_emb, history_size, candidates_raw.shape[1])
        states = probe(pred_emb)
    return states


def candidate_cost(states: torch.Tensor, candidates: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
    pole_pos = states[..., 0]
    pole_vel = states[..., 1]
    cart_pos = states[..., 2]
    cart_vel = states[..., 3]
    cost = (
        8.0 * pole_pos.square()
        + 0.5 * pole_vel.square()
        + 1.0 * cart_pos.square()
        + 0.05 * cart_vel.square()
    ).sum(dim=1)
    cost = cost + 0.01 * candidates.square().sum(dim=(1, 2))
    prev_action = last_action.reshape(1, 1, -1).expand(candidates.shape[0], 1, -1)
    prev = torch.cat([prev_action, candidates[:, :-1]], dim=1)
    cost = cost + 0.02 * (candidates - prev).square().sum(dim=(1, 2))
    return cost


def choose_action(
    model: torch.nn.Module,
    probe: torch.nn.Module,
    pixel_history: deque[np.ndarray],
    action_history: deque[np.ndarray],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    img_size: int,
    history_size: int,
    action_dim: int,
    last_action: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    horizon = args_cli.horizon
    num_candidates = args_cli.num_candidates
    elite_count = max(1, int(num_candidates * args_cli.elite_frac))
    mean = last_action.reshape(1, action_dim).repeat(horizon, 1).to(device)
    std = torch.full_like(mean, args_cli.init_std)

    best_action = mean[0].clone()
    best_cost = torch.tensor(float("inf"), device=device)
    best_state = None
    for _ in range(args_cli.cem_iters):
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(num_candidates, horizon, action_dim, device=device)
        samples = samples.clamp(args_cli.action_low, args_cli.action_high)
        states = predict_candidate_states(
            model=model,
            probe=probe,
            pixel_history=pixel_history,
            action_history=action_history,
            candidates_raw=samples,
            action_mean=action_mean,
            action_std=action_std,
            img_size=img_size,
            history_size=history_size,
            device=device,
        )
        costs = candidate_cost(states, samples, last_action.to(device))
        elite_idx = torch.topk(costs, elite_count, largest=False).indices
        elite = samples[elite_idx]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0).clamp_min(args_cli.min_std)
        current_best = int(costs.argmin().item())
        if costs[current_best] < best_cost:
            best_cost = costs[current_best]
            best_action = samples[current_best, 0].detach()
            best_state = states[current_best, 0].detach()

    info = {
        "best_cost": float(best_cost.detach().cpu()),
        "pred_next_state": best_state.detach().cpu().tolist() if best_state is not None else None,
    }
    return best_action.clamp(args_cli.action_low, args_cli.action_high), info


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("LeWM MPC v1 requires --num-envs 1.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = resolve_device(args_cli.device)
    action_mean, action_std = load_action_stats(args_cli.action_stats_h5, args_cli.max_stats_rows_per_file)
    action_mean = action_mean.to(device)
    action_std = action_std.to(device)
    action_dim = int(action_mean.shape[-1])
    model = load_lewm(args_cli.checkpoint, args_cli.cache_dir, action_dim, args_cli.img_size, device)
    probe = load_state_probe(args_cli.probe, device)
    history_size = infer_history_size(model)

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] checkpoint={args_cli.checkpoint}")
    print(f"[INFO] probe={args_cli.probe}")
    print(f"[INFO] device={device}, history_size={history_size}, action_dim={action_dim}")
    print("[INFO] PPO policy is not loaded; actions are selected by LeWM MPC.")

    episodes = []
    gif_frames = []
    try:
        for ep_idx in range(args_cli.episodes):
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            first_pixel = get_pixels(obs, args_cli.pixel_key, args_cli.num_envs)
            pixel_history: deque[np.ndarray] = deque([first_pixel.copy() for _ in range(history_size)], maxlen=history_size)
            zero_action = np.zeros((action_dim,), dtype=np.float32)
            action_history: deque[np.ndarray] = deque([zero_action.copy() for _ in range(history_size)], maxlen=history_size)
            last_action = torch.zeros(action_dim, device=device)

            rewards = []
            dones = []
            true_states = []
            mpc_costs = []
            survival_steps = 0
            for step in range(args_cli.episode_len):
                true_state = get_cartpole_state(env)
                true_states.append(first_env(true_state, args_cli.num_envs).astype(np.float32))
                action, info = choose_action(
                    model=model,
                    probe=probe,
                    pixel_history=pixel_history,
                    action_history=action_history,
                    action_mean=action_mean,
                    action_std=action_std,
                    img_size=args_cli.img_size,
                    history_size=history_size,
                    action_dim=action_dim,
                    last_action=last_action,
                    device=device,
                )
                action_env = action.reshape(1, -1).to(env.unwrapped.device)
                obs, reward, terminated, truncated, _ = env.step(action_env)
                done = torch.logical_or(terminated, truncated)

                pixel = get_pixels(obs, args_cli.pixel_key, args_cli.num_envs)
                pixel_history.append(pixel)
                action_np = action.detach().cpu().numpy().astype(np.float32)
                action_history.append(action_np)
                last_action = action.detach()
                rewards.append(float(np.asarray(first_env(reward, args_cli.num_envs)).reshape(-1)[0]))
                done_bool = bool(np.asarray(first_env(done, args_cli.num_envs)).reshape(-1)[0])
                dones.append(done_bool)
                mpc_costs.append(info["best_cost"])
                survival_steps += 1
                if args_cli.save_gif and ep_idx == 0:
                    gif_frames.append(visualize_frame(pixel))
                if done_bool:
                    break

            states = np.asarray(true_states, dtype=np.float32)
            episode = {
                "episode": ep_idx,
                "reward_sum": float(np.sum(rewards)),
                "survival_steps": int(survival_steps),
                "done_count": int(np.sum(dones)),
                "mean_abs_pole_angle": float(np.abs(states[:, 0]).mean()) if len(states) else None,
                "max_abs_pole_angle": float(np.abs(states[:, 0]).max()) if len(states) else None,
                "mean_abs_cart_pos": float(np.abs(states[:, 2]).mean()) if len(states) else None,
                "mean_mpc_cost": float(np.mean(mpc_costs)) if mpc_costs else None,
            }
            episodes.append(episode)
            print(f"[INFO] episode={ep_idx} survival={survival_steps} reward={episode['reward_sum']:.3f}")
    finally:
        env.close()

    result = {
        "mode": "lewm_mpc_cartpole",
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "probe": str(args_cli.probe),
        "episodes": episodes,
        "summary": {
            "episodes": len(episodes),
            "mean_reward_sum": float(np.mean([ep["reward_sum"] for ep in episodes])) if episodes else None,
            "mean_survival_steps": float(np.mean([ep["survival_steps"] for ep in episodes])) if episodes else None,
            "mean_abs_pole_angle": float(np.mean([ep["mean_abs_pole_angle"] for ep in episodes])) if episodes else None,
            "terminated_episodes": int(np.sum([ep["done_count"] > 0 for ep in episodes])),
        },
        "mpc": {
            "horizon": args_cli.horizon,
            "num_candidates": args_cli.num_candidates,
            "elite_frac": args_cli.elite_frac,
            "cem_iters": args_cli.cem_iters,
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
