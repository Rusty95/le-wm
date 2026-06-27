#!/usr/bin/env python3
"""Run LeWM inside IsaacLab for online-style latent rollout evaluation.

This is the first deployment bridge: IsaacLab steps a live environment with a
trained PPO policy, while LeWM consumes the live camera frames/actions and
checks whether its latent rollout matches future encoded observations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


DEFAULT_POLICY = (
    "/home/hall/code/RL-Learning-BasedOn-IsaacLab/logs/standalone/ppo/cartpole/"
    "2026-06-07_21-41-11/model_149.pt"
)
REPO_DIR = Path(__file__).resolve().parents[1]
LOCAL_REPOS = [
    Path("/home/hall/code/RL-Learning-BasedOn-IsaacLab/source/rl_lab_learning"),
    REPO_DIR,
    Path("/home/hall/code/stable-worldmodel"),
    Path("/home/hall/code/stable-pretraining"),
]
for repo in LOCAL_REPOS:
    if repo.exists() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online-style IsaacLab evaluation for a LeWM checkpoint.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-RGB-Camera-Direct-v0")
    parser.add_argument("--policy-checkpoint", type=Path, default=Path(DEFAULT_POLICY))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--episode-len", type=int, default=80)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--pixel-key", type=str, default="policy")
    parser.add_argument("--obs-key", type=str, default="policy")
    parser.add_argument("--policy-obs-source", choices=["cartpole-state", "obs"], default="cartpole-state")
    parser.add_argument("--stochastic", action="store_true", help="Sample the PPO Gaussian policy.")
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument("--action-stats-h5", type=Path, nargs="+", default=DEFAULT_ACTION_STATS_H5)
    parser.add_argument("--max-stats-rows-per-file", type=int, default=50000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/hall/code/.stable-wm/eval/lewm_online_isaaclab_eval.json"),
    )
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1)
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
import isaaclab_tasks  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import rl_lab_learning.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_lab_learning.algorithms.ppo import ActorCritic  # noqa: E402
from scripts.lewm_isaaclab_common import (  # noqa: E402
    DEFAULT_ACTION_STATS_H5,
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT,
    first_env,
    get_cartpole_state,
    get_nested,
    get_pixels,
    infer_history_size,
    load_action_stats,
    load_lewm,
    normalize_actions,
    preprocess_pixels,
    rollout_predictions,
    to_numpy,
)


def _to_numpy(value: Any) -> np.ndarray:
    return to_numpy(value)


def _first_env(value: Any) -> np.ndarray:
    return first_env(value, num_envs=args_cli.num_envs)


def _get_nested(obs: Any, key: str) -> Any:
    return get_nested(obs, key)


def _get_cartpole_state(env) -> torch.Tensor:
    return get_cartpole_state(env)


def _get_pixels(obs: Any) -> np.ndarray:
    return get_pixels(obs, pixel_key=args_cli.pixel_key, num_envs=args_cli.num_envs)


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
    policy_obs = _get_nested(obs, args_cli.obs_key)
    return policy_obs if isinstance(policy_obs, torch.Tensor) else torch.as_tensor(policy_obs)


def _policy_action(policy: ActorCritic, env, obs: Any, device: torch.device) -> torch.Tensor:
    policy_obs = _get_policy_obs(env, obs).to(device).float()
    with torch.no_grad():
        if args_cli.stochastic:
            action, _, _ = policy.act(policy_obs)
        else:
            action = policy.act_inference(policy_obs)
        if args_cli.action_noise_std > 0:
            action = action + args_cli.action_noise_std * torch.randn_like(action)
    return action.clamp(-1.0, 1.0)


def _load_action_stats(paths: list[Path], max_rows_per_file: int) -> tuple[torch.Tensor, torch.Tensor]:
    return load_action_stats(paths, max_rows_per_file=max_rows_per_file)


def _preprocess_pixels(frames: list[np.ndarray], img_size: int, device: torch.device) -> torch.Tensor:
    return preprocess_pixels(frames, img_size=img_size, device=device).unsqueeze(0)


def _normalize_actions(actions: list[np.ndarray], mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> torch.Tensor:
    action = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
    if action.ndim == 1:
        action = action.unsqueeze(-1)
    return normalize_actions(action, mean, std, device=device).unsqueeze(0)


def _infer_history_size(model: torch.nn.Module) -> int:
    return infer_history_size(model)


def _rollout_predictions(
    model: torch.nn.Module,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
    max_horizon: int,
) -> torch.Tensor:
    return rollout_predictions(model, emb, act_emb, history_size, max_horizon)


def _eval_episode(
    model: torch.nn.Module,
    pixels: list[np.ndarray],
    actions: list[np.ndarray],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    history_size: int,
    horizons: list[int],
    device: torch.device,
) -> tuple[dict[int, float], int]:
    max_horizon = max(horizons)
    if len(pixels) < history_size + max_horizon:
        return {h: 0.0 for h in horizons}, 0

    with torch.no_grad():
        batch = {
            "pixels": _preprocess_pixels(pixels, args_cli.img_size, device),
            "action": _normalize_actions(actions, action_mean, action_std, device),
        }
        output = model.encode(batch)
        emb = output["emb"]
        act_emb = output["act_emb"]

        totals = {h: 0.0 for h in horizons}
        windows = 0
        for start in range(0, emb.shape[1] - history_size - max_horizon + 1):
            emb_seq = emb[:, start : start + history_size + max_horizon]
            act_seq = act_emb[:, start : start + history_size + max_horizon]
            pred = _rollout_predictions(model, emb_seq, act_seq, history_size, max_horizon)
            tgt = emb_seq[:, history_size : history_size + max_horizon]
            for horizon in horizons:
                totals[horizon] += float((pred[:, horizon - 1] - tgt[:, horizon - 1]).pow(2).mean().cpu())
            windows += 1
    return totals, windows


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("Online LeWM eval currently supports --num-envs 1.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_device = torch.device(env.unwrapped.device)

    lewm_device = torch.device(
        args_cli.device if args_cli.device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    policy = _load_policy(args_cli.policy_checkpoint, env_device)
    horizons = sorted(set(args_cli.horizons))
    action_mean, action_std = _load_action_stats(args_cli.action_stats_h5, args_cli.max_stats_rows_per_file)
    action_mean = action_mean.to(lewm_device)
    action_std = action_std.to(lewm_device)
    model = load_lewm(
        checkpoint=args_cli.checkpoint,
        cache_dir=args_cli.cache_dir,
        action_dim=int(action_mean.shape[-1]),
        img_size=args_cli.img_size,
        device=lewm_device,
    )
    history_size = _infer_history_size(model)

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] checkpoint={args_cli.checkpoint}")
    print(f"[INFO] policy={args_cli.policy_checkpoint}")
    print(f"[INFO] env_device={env_device}, lewm_device={lewm_device}")
    print(f"[INFO] history_size={history_size}, horizons={horizons}")
    print(f"[INFO] action_mean={action_mean.flatten().tolist()}, action_std={action_std.flatten().tolist()}")

    totals = {h: 0.0 for h in horizons}
    num_windows = 0
    frames = 0
    episode_summaries = []
    try:
        for ep_idx in range(args_cli.episodes):
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            pixels: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            rewards = []
            dones = []

            for _ in range(args_cli.episode_len):
                action = _policy_action(policy, env, obs, env_device)
                pixels.append(_get_pixels(obs))
                actions.append(_first_env(action))
                obs, reward, terminated, truncated, _ = env.step(action)
                done = torch.logical_or(terminated, truncated)
                rewards.append(float(np.asarray(_first_env(reward)).reshape(-1)[0]))
                dones.append(bool(np.asarray(_first_env(done)).reshape(-1)[0]))

            ep_totals, ep_windows = _eval_episode(
                model=model,
                pixels=pixels,
                actions=actions,
                action_mean=action_mean,
                action_std=action_std,
                history_size=history_size,
                horizons=horizons,
                device=lewm_device,
            )
            for horizon in horizons:
                totals[horizon] += ep_totals[horizon]
            num_windows += ep_windows
            frames += len(pixels)
            episode_summaries.append(
                {
                    "episode": ep_idx,
                    "frames": len(pixels),
                    "windows": ep_windows,
                    "reward_sum": float(np.sum(rewards)),
                    "done_count": int(np.sum(dones)),
                    "metrics": {
                        f"horizon_{h}_mse": (ep_totals[h] / ep_windows if ep_windows else None)
                        for h in horizons
                    },
                }
            )
            print(f"[INFO] episode={ep_idx} frames={len(pixels)} windows={ep_windows}")
    finally:
        env.close()

    if num_windows == 0:
        raise RuntimeError("No valid rollout windows were evaluated.")

    result = {
        "mode": "isaaclab_online_style_eval",
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "policy_checkpoint": str(args_cli.policy_checkpoint),
        "cache_dir": str(args_cli.cache_dir),
        "device": str(lewm_device),
        "episodes": args_cli.episodes,
        "frames": frames,
        "episode_len": args_cli.episode_len,
        "history_size": history_size,
        "horizons": horizons,
        "num_windows": num_windows,
        "img_size": args_cli.img_size,
        "action_stats_h5": [str(path) for path in args_cli.action_stats_h5],
        "metrics": {f"horizon_{h}_mse": totals[h] / num_windows for h in horizons},
        "episodes_detail": episode_summaries,
        "note": (
            "IsaacLab generated fresh trajectories in this process; LeWM predictions "
            "were evaluated against future latent targets from the same live episodes."
        ),
    }
    text = json.dumps(result, indent=2)
    print(text)
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    args_cli.out.write_text(text + "\n", encoding="utf-8")
    print(f"[INFO] wrote {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
