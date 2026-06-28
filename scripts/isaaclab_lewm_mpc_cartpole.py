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
    parser.add_argument("--add-ground-plane", action="store_true", help="Add a light grey visual ground plane for easier viewing.")
    parser.add_argument("--ground-z", type=float, default=-0.05)
    parser.add_argument("--ground-size", type=float, default=10.0)
    parser.add_argument("--force-rescue-candidates", action="store_true", help="Always evaluate hand-coded strong rescue action sequences.")
    parser.add_argument("--save-step-diagnostics", action="store_true", help="Store per-step true state, selected action, and MPC diagnostics in JSON.")
    parser.add_argument("--objective", choices=["state-probe", "latent-target"], default="state-probe")
    parser.add_argument("--target-h5", type=Path, default=Path("/home/hall/code/.stable-wm/datasets/isaaclab_policy_camera_50k.h5"))
    parser.add_argument("--target-max-frames", type=int, default=512)
    parser.add_argument("--target-pole-threshold", type=float, default=0.08)
    parser.add_argument("--target-cart-threshold", type=float, default=0.25)
    parser.add_argument("--latent-cost", choices=["cosine", "mse"], default="cosine")
    parser.add_argument("--latent-terminal-only", action="store_true", help="Only score the final predicted latent.")
    parser.add_argument(
        "--direction-bias-weight",
        type=float,
        default=0.0,
        help="Use current true Cartpole state to bias the first MPC action toward a rescue direction. Ablation only.",
    )
    parser.add_argument("--direction-pole-weight", type=float, default=1.5)
    parser.add_argument("--direction-pole-vel-weight", type=float, default=0.4)
    parser.add_argument("--direction-cart-weight", type=float, default=0.0)
    parser.add_argument("--direction-cart-vel-weight", type=float, default=0.0)
    parser.add_argument("--direction-cart-threshold", type=float, default=0.8)
    parser.add_argument(
        "--direction-cart-boost",
        type=float,
        default=0.0,
        help="Extra cart_pos weight applied once abs(cart_pos) exceeds --direction-cart-threshold.",
    )
    parser.add_argument(
        "--direction-cart-gate",
        action="store_true",
        help="Gate cart terms down when pole angle/velocity are unsafe.",
    )
    parser.add_argument("--direction-gate-angle-weight", type=float, default=1.0)
    parser.add_argument("--direction-gate-vel-weight", type=float, default=0.15)
    parser.add_argument("--direction-gate-threshold", type=float, default=0.35)
    parser.add_argument(
        "--action-prior-weight",
        type=float,
        default=0.0,
        help="Penalize distance from a continuous rescue action prior computed from the current true state. Ablation only.",
    )
    parser.add_argument("--prior-pole-kp", type=float, default=1.5)
    parser.add_argument("--prior-pole-kd", type=float, default=0.35)
    parser.add_argument("--prior-cart-kp", type=float, default=0.25)
    parser.add_argument("--prior-cart-kd", type=float, default=0.08)
    parser.add_argument(
        "--edge-rescue-weight",
        type=float,
        default=0.0,
        help="Extra first-action cost near cart boundaries to discourage actions that keep pushing the cart outward. Ablation only.",
    )
    parser.add_argument("--edge-rescue-threshold", type=float, default=2.3)
    parser.add_argument("--edge-rescue-velocity-threshold", type=float, default=0.2)
    parser.add_argument(
        "--edge-rescue-return-action",
        type=float,
        default=0.8,
        help="Preferred first-action magnitude toward the track centre when edge rescue is active.",
    )
    parser.add_argument(
        "--edge-rescue-prior-suppression",
        type=float,
        default=1.0,
        help="How strongly edge rescue suppresses the continuous action prior when the cart is near the boundary.",
    )
    parser.add_argument(
        "--edge-rescue-gate-scale",
        type=float,
        default=1.0,
        help="Activation value at which edge rescue fully gates the action prior.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if "Camera" in args.task:
        args.enable_cameras = True
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omni.physx.scripts import physicsUtils  # noqa: E402
from pxr import Gf  # noqa: E402

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
    rollout_predictions_online,
)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def add_visual_ground_plane(path: str = "/World/LeWMGroundPlane") -> None:
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(path).IsValid():
        return
    physicsUtils.add_ground_plane(
        stage,
        path,
        "Z",
        args_cli.ground_size,
        Gf.Vec3f(0.0, 0.0, args_cli.ground_z),
        Gf.Vec3f(0.55, 0.55, 0.55),
    )


def make_rescue_candidates(horizon: int, action_dim: int, device: torch.device) -> torch.Tensor:
    """Return a small bank of deliberately aggressive action sequences."""
    high = float(args_cli.action_high)
    low = float(args_cli.action_low)
    mid = 0.0
    patterns: list[list[float]] = []

    patterns.append([high] * horizon)
    patterns.append([low] * horizon)
    patterns.append([high if i < max(1, horizon // 2) else mid for i in range(horizon)])
    patterns.append([low if i < max(1, horizon // 2) else mid for i in range(horizon)])
    patterns.append([high if i % 2 == 0 else low for i in range(horizon)])
    patterns.append([low if i % 2 == 0 else high for i in range(horizon)])
    patterns.append([high] + [mid] * (horizon - 1))
    patterns.append([low] + [mid] * (horizon - 1))

    rescue = torch.tensor(patterns, dtype=torch.float32, device=device).unsqueeze(-1)
    if action_dim > 1:
        rescue = rescue.expand(-1, -1, action_dim).contiguous()
    return rescue


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
        past = past_actions[-(history_size - 1) :].unsqueeze(0).expand(num_candidates, -1, -1)
        full_actions_raw = torch.cat([past, candidates_raw], dim=1)
        full_actions = normalize_actions(full_actions_raw, action_mean, action_std, device=device)
        act_emb = model.action_encoder(full_actions)
        pred_emb = rollout_predictions_online(model, emb, act_emb, history_size, candidates_raw.shape[1])
        states = probe(pred_emb)
    return states


def predict_candidate_embeddings(
    model: torch.nn.Module,
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
        past = past_actions[-(history_size - 1) :].unsqueeze(0).expand(num_candidates, -1, -1)
        full_actions_raw = torch.cat([past, candidates_raw], dim=1)
        full_actions = normalize_actions(full_actions_raw, action_mean, action_std, device=device)
        act_emb = model.action_encoder(full_actions)
        pred_emb = rollout_predictions_online(model, emb, act_emb, history_size, candidates_raw.shape[1])
    return pred_emb


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


def add_action_regularization(cost: torch.Tensor, candidates: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
    cost = cost + 0.001 * candidates.square().sum(dim=(1, 2))
    prev_action = last_action.reshape(1, 1, -1).expand(candidates.shape[0], 1, -1)
    prev = torch.cat([prev_action, candidates[:, :-1]], dim=1)
    return cost + 0.002 * (candidates - prev).square().sum(dim=(1, 2))


def add_direction_bias(
    cost: torch.Tensor,
    candidates: torch.Tensor,
    current_state: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    if args_cli.direction_bias_weight <= 0.0 or current_state is None:
        return cost, None
    state = current_state.detach().float().reshape(-1).to(candidates.device)
    pole_pos = state[0]
    pole_vel = state[1]
    cart_pos = state[2]
    cart_vel = state[3]
    cart_weight = torch.as_tensor(args_cli.direction_cart_weight, dtype=state.dtype, device=state.device)
    if args_cli.direction_cart_boost > 0.0:
        excess = torch.relu(cart_pos.abs() - args_cli.direction_cart_threshold)
        cart_weight = cart_weight + args_cli.direction_cart_boost * excess
    cart_gate = torch.ones((), dtype=state.dtype, device=state.device)
    if args_cli.direction_cart_gate:
        pole_safety = (
            args_cli.direction_gate_angle_weight * pole_pos.abs()
            + args_cli.direction_gate_vel_weight * pole_vel.abs()
        )
        cart_gate = (1.0 - pole_safety / max(args_cli.direction_gate_threshold, 1e-6)).clamp(0.0, 1.0)
    score = (
        args_cli.direction_pole_weight * pole_pos
        + args_cli.direction_pole_vel_weight * pole_vel
        + cart_gate * (cart_weight * cart_pos + args_cli.direction_cart_vel_weight * cart_vel)
    )
    direction = torch.sign(score)
    if direction.abs() < 1e-6:
        return cost, {
            "score": float(score.cpu()),
            "direction": 0.0,
            "weight": args_cli.direction_bias_weight,
            "pole_pos": float(pole_pos.cpu()),
            "pole_vel": float(pole_vel.cpu()),
            "cart_pos": float(cart_pos.cpu()),
            "cart_vel": float(cart_vel.cpu()),
            "cart_weight": float(cart_weight.cpu()),
            "cart_gate": float(cart_gate.cpu()),
        }

    # If direction is positive, a negative first action is encouraged; if it is
    # negative, a positive first action is encouraged.
    bias = args_cli.direction_bias_weight * direction * candidates[:, 0, 0]
    meta = {
        "score": float(score.cpu()),
        "direction": float(direction.cpu()),
        "weight": args_cli.direction_bias_weight,
        "pole_pos": float(pole_pos.cpu()),
        "pole_vel": float(pole_vel.cpu()),
        "cart_pos": float(cart_pos.cpu()),
        "cart_vel": float(cart_vel.cpu()),
        "cart_weight": float(cart_weight.cpu()),
        "cart_gate": float(cart_gate.cpu()),
    }
    return cost + bias, meta


def add_action_prior(
    cost: torch.Tensor,
    candidates: torch.Tensor,
    current_state: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    if args_cli.action_prior_weight <= 0.0 or current_state is None:
        return cost, None
    state = current_state.detach().float().reshape(-1).to(candidates.device)
    pole_pos = state[0]
    pole_vel = state[1]
    cart_pos = state[2]
    cart_vel = state[3]
    pole_score = args_cli.prior_pole_kp * pole_pos + args_cli.prior_pole_kd * pole_vel
    cart_score = args_cli.prior_cart_kp * cart_pos + args_cli.prior_cart_kd * cart_vel
    score = pole_score + cart_score
    edge_gate = torch.zeros((), dtype=state.dtype, device=state.device)
    if args_cli.edge_rescue_weight > 0.0 and args_cli.edge_rescue_prior_suppression > 0.0:
        edge_excess = torch.relu(cart_pos.abs() - args_cli.edge_rescue_threshold)
        outward_vel = torch.relu(torch.sign(cart_pos) * cart_vel - args_cli.edge_rescue_velocity_threshold)
        edge_activation = (edge_excess * (1.0 + outward_vel)).clamp(0.0, 3.0)
        edge_gate = (edge_activation / max(args_cli.edge_rescue_gate_scale, 1e-6)).clamp(0.0, 1.0)
    effective_weight = args_cli.action_prior_weight * (
        1.0 - args_cli.edge_rescue_prior_suppression * edge_gate
    ).clamp(0.0, 1.0)

    # Positive score means the pole/cart state is leaning to the positive side,
    # so the prior favours a negative first action.  Unlike add_direction_bias,
    # this keeps the magnitude continuous instead of collapsing to sign(score).
    prior = (-score).clamp(args_cli.action_low, args_cli.action_high)
    first_action = candidates[:, 0, 0]
    prior_cost = effective_weight * (first_action - prior).square()
    meta = {
        "score": float(score.cpu()),
        "prior_action": float(prior.cpu()),
        "weight": args_cli.action_prior_weight,
        "effective_weight": float(effective_weight.cpu()),
        "edge_gate": float(edge_gate.cpu()),
        "pole_score": float(pole_score.cpu()),
        "cart_score": float(cart_score.cpu()),
        "pole_pos": float(pole_pos.cpu()),
        "pole_vel": float(pole_vel.cpu()),
        "cart_pos": float(cart_pos.cpu()),
        "cart_vel": float(cart_vel.cpu()),
        "prior_pole_kp": args_cli.prior_pole_kp,
        "prior_pole_kd": args_cli.prior_pole_kd,
        "prior_cart_kp": args_cli.prior_cart_kp,
        "prior_cart_kd": args_cli.prior_cart_kd,
    }
    return cost + prior_cost, meta


def add_edge_rescue(
    cost: torch.Tensor,
    candidates: torch.Tensor,
    current_state: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    if args_cli.edge_rescue_weight <= 0.0 or current_state is None:
        return cost, None
    state = current_state.detach().float().reshape(-1).to(candidates.device)
    cart_pos = state[2]
    cart_vel = state[3]
    abs_cart = cart_pos.abs()
    edge_excess = torch.relu(abs_cart - args_cli.edge_rescue_threshold)
    outward_vel = torch.relu(torch.sign(cart_pos) * cart_vel - args_cli.edge_rescue_velocity_threshold)
    activation = (edge_excess * (1.0 + outward_vel)).clamp(0.0, 3.0)
    if activation <= 0.0:
        return cost, {
            "active": False,
            "activation": 0.0,
            "cart_pos": float(cart_pos.cpu()),
            "cart_vel": float(cart_vel.cpu()),
            "target_action": 0.0,
            "weight": args_cli.edge_rescue_weight,
        }

    # Near the positive edge, prefer a negative first action; near the negative
    # edge, prefer a positive first action.
    target_action = (-torch.sign(cart_pos) * args_cli.edge_rescue_return_action).clamp(
        args_cli.action_low,
        args_cli.action_high,
    )
    first_action = candidates[:, 0, 0]
    outward = torch.relu(torch.sign(cart_pos) * first_action).square()
    return_error = (first_action - target_action).square()
    rescue_cost = args_cli.edge_rescue_weight * activation * (outward + 0.5 * return_error)
    meta = {
        "active": True,
        "activation": float(activation.cpu()),
        "edge_excess": float(edge_excess.cpu()),
        "outward_vel": float(outward_vel.cpu()),
        "cart_pos": float(cart_pos.cpu()),
        "cart_vel": float(cart_vel.cpu()),
        "target_action": float(target_action.cpu()),
        "weight": args_cli.edge_rescue_weight,
        "threshold": args_cli.edge_rescue_threshold,
        "velocity_threshold": args_cli.edge_rescue_velocity_threshold,
        "return_action": args_cli.edge_rescue_return_action,
    }
    return cost + rescue_cost, meta


def latent_candidate_cost(pred_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
    target = target_emb.reshape(1, 1, -1).to(pred_emb.device)
    pred = pred_emb[:, -1:, :] if args_cli.latent_terminal_only else pred_emb
    if args_cli.latent_cost == "cosine":
        cost = 1.0 - F.cosine_similarity(pred, target.expand_as(pred), dim=-1)
    else:
        cost = (pred - target).square().mean(dim=-1)
    return cost.sum(dim=1)


def build_latent_target(
    model: torch.nn.Module,
    data_path: Path,
    img_size: int,
    device: torch.device,
    max_frames: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Target H5 not found: {data_path}")

    with h5py.File(data_path, "r") as f:
        pixels = f["pixels"]
        num_frames = int(pixels.shape[0])
        if "policy_obs" in f:
            obs = np.asarray(f["policy_obs"][:], dtype=np.float32)
            mask = (np.abs(obs[:, 0]) <= args_cli.target_pole_threshold) & (
                np.abs(obs[:, 2]) <= args_cli.target_cart_threshold
            )
            indices = np.nonzero(mask)[0]
            if len(indices) == 0:
                indices = np.arange(num_frames)
        else:
            indices = np.arange(num_frames)

        if len(indices) > max_frames:
            pick = np.linspace(0, len(indices) - 1, max_frames).round().astype(np.int64)
            indices = indices[pick]
        frames = np.asarray(pixels[np.sort(indices)], dtype=np.float32)

    batch_size = 128
    embs = []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            batch = preprocess_pixels(frames[start : start + batch_size], img_size=img_size, device=device).unsqueeze(0)
            embs.append(model.encode({"pixels": batch})["emb"].squeeze(0))
    target_emb = torch.cat(embs, dim=0).mean(dim=0)
    if args_cli.latent_cost == "cosine":
        target_emb = F.normalize(target_emb, dim=0)
    meta = {
        "target_h5": str(data_path),
        "target_frames": int(len(frames)),
        "target_pole_threshold": args_cli.target_pole_threshold,
        "target_cart_threshold": args_cli.target_cart_threshold,
        "latent_cost": args_cli.latent_cost,
        "latent_terminal_only": args_cli.latent_terminal_only,
    }
    return target_emb, meta


def choose_action(
    model: torch.nn.Module,
    probe: torch.nn.Module | None,
    pixel_history: deque[np.ndarray],
    action_history: deque[np.ndarray],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    img_size: int,
    history_size: int,
    action_dim: int,
    last_action: torch.Tensor,
    device: torch.device,
    target_emb: torch.Tensor | None = None,
    current_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    horizon = args_cli.horizon
    num_candidates = args_cli.num_candidates
    elite_count = max(1, int(num_candidates * args_cli.elite_frac))
    mean = last_action.reshape(1, action_dim).repeat(horizon, 1).to(device)
    std = torch.full_like(mean, args_cli.init_std)

    best_action = mean[0].clone()
    best_cost = torch.tensor(float("inf"), device=device)
    best_state = None
    rescue_bank = make_rescue_candidates(horizon, action_dim, device) if args_cli.force_rescue_candidates else None
    best_rescue_cost = torch.tensor(float("inf"), device=device)
    best_rescue_action = None
    best_direction_meta = None
    best_action_prior_meta = None
    best_edge_rescue_meta = None
    for _ in range(args_cli.cem_iters):
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(num_candidates, horizon, action_dim, device=device)
        samples = samples.clamp(args_cli.action_low, args_cli.action_high)
        rescue_count = 0
        if rescue_bank is not None:
            rescue_count = rescue_bank.shape[0]
            samples = torch.cat([samples, rescue_bank], dim=0)
        if args_cli.objective == "latent-target":
            if target_emb is None:
                raise ValueError("target_emb is required for --objective latent-target.")
            pred_emb = predict_candidate_embeddings(
                model=model,
                pixel_history=pixel_history,
                action_history=action_history,
                candidates_raw=samples,
                action_mean=action_mean,
                action_std=action_std,
                img_size=img_size,
                history_size=history_size,
                device=device,
            )
            costs = add_action_regularization(latent_candidate_cost(pred_emb, target_emb), samples, last_action.to(device))
            states = None
        else:
            if probe is None:
                raise ValueError("probe is required for --objective state-probe.")
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
        costs, direction_meta = add_direction_bias(costs, samples, current_state)
        costs, action_prior_meta = add_action_prior(costs, samples, current_state)
        costs, edge_rescue_meta = add_edge_rescue(costs, samples, current_state)
        if rescue_count:
            rescue_costs = costs[-rescue_count:]
            rescue_best = int(rescue_costs.argmin().item())
            if rescue_costs[rescue_best] < best_rescue_cost:
                best_rescue_cost = rescue_costs[rescue_best]
                best_rescue_action = samples[-rescue_count + rescue_best, 0].detach()
        elite_idx = torch.topk(costs, elite_count, largest=False).indices
        elite = samples[elite_idx]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0).clamp_min(args_cli.min_std)
        current_best = int(costs.argmin().item())
        if costs[current_best] < best_cost:
            best_cost = costs[current_best]
            best_action = samples[current_best, 0].detach()
            best_direction_meta = direction_meta
            best_action_prior_meta = action_prior_meta
            best_edge_rescue_meta = edge_rescue_meta
            if states is not None:
                best_state = states[current_best, 0].detach()

    info = {
        "best_cost": float(best_cost.detach().cpu()),
        "pred_next_state": best_state.detach().cpu().tolist() if best_state is not None else None,
        "best_rescue_cost": float(best_rescue_cost.detach().cpu()) if args_cli.force_rescue_candidates else None,
        "best_rescue_action": best_rescue_action.detach().cpu().tolist() if best_rescue_action is not None else None,
        "direction_bias": best_direction_meta,
        "action_prior": best_action_prior_meta,
        "edge_rescue": best_edge_rescue_meta,
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
    if args_cli.add_ground_plane:
        add_visual_ground_plane()
    device = resolve_device(args_cli.device)
    action_mean, action_std = load_action_stats(args_cli.action_stats_h5, args_cli.max_stats_rows_per_file)
    action_mean = action_mean.to(device)
    action_std = action_std.to(device)
    action_dim = int(action_mean.shape[-1])
    model = load_lewm(args_cli.checkpoint, args_cli.cache_dir, action_dim, args_cli.img_size, device)
    probe = load_state_probe(args_cli.probe, device) if args_cli.objective == "state-probe" else None
    history_size = infer_history_size(model)
    target_emb = None
    target_meta = None
    if args_cli.objective == "latent-target":
        target_emb, target_meta = build_latent_target(
            model=model,
            data_path=args_cli.target_h5,
            img_size=args_cli.img_size,
            device=device,
            max_frames=args_cli.target_max_frames,
        )

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] checkpoint={args_cli.checkpoint}")
    print(f"[INFO] objective={args_cli.objective}")
    if probe is not None:
        print(f"[INFO] probe={args_cli.probe}")
    if target_meta is not None:
        print(f"[INFO] latent_target={target_meta}")
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
            step_diagnostics = []
            survival_steps = 0
            for step in range(args_cli.episode_len):
                true_state = get_cartpole_state(env)
                true_state_np = first_env(true_state, args_cli.num_envs).astype(np.float32)
                true_states.append(true_state_np)
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
                    target_emb=target_emb,
                    current_state=true_state,
                )
                action_env = action.reshape(1, -1).to(env.unwrapped.device)
                obs, reward, terminated, truncated, _ = env.step(action_env)
                done = torch.logical_or(terminated, truncated)

                pixel = get_pixels(obs, args_cli.pixel_key, args_cli.num_envs)
                pixel_history.append(pixel)
                action_np = action.detach().cpu().numpy().astype(np.float32)
                action_history.append(action_np)
                last_action = action.detach()
                reward_float = float(np.asarray(first_env(reward, args_cli.num_envs)).reshape(-1)[0])
                rewards.append(reward_float)
                done_bool = bool(np.asarray(first_env(done, args_cli.num_envs)).reshape(-1)[0])
                dones.append(done_bool)
                mpc_costs.append(info["best_cost"])
                if args_cli.save_step_diagnostics:
                    step_diagnostics.append(
                        {
                            "step": step,
                            "true_state": true_state_np.tolist(),
                            "action": action_np.tolist(),
                            "reward": reward_float,
                            "done": done_bool,
                            "best_cost": info["best_cost"],
                            "pred_next_state": info["pred_next_state"],
                            "best_rescue_cost": info["best_rescue_cost"],
                            "best_rescue_action": info["best_rescue_action"],
                            "direction_bias": info["direction_bias"],
                            "action_prior": info["action_prior"],
                            "edge_rescue": info["edge_rescue"],
                        }
                    )
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
            if args_cli.save_step_diagnostics:
                episode["steps"] = step_diagnostics
            episodes.append(episode)
            print(f"[INFO] episode={ep_idx} survival={survival_steps} reward={episode['reward_sum']:.3f}")
    finally:
        env.close()

    result = {
        "mode": "lewm_mpc_cartpole",
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "probe": str(args_cli.probe) if probe is not None else None,
        "objective": args_cli.objective,
        "latent_target": target_meta,
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
            "force_rescue_candidates": args_cli.force_rescue_candidates,
            "direction_bias_weight": args_cli.direction_bias_weight,
            "direction_pole_weight": args_cli.direction_pole_weight,
            "direction_pole_vel_weight": args_cli.direction_pole_vel_weight,
            "direction_cart_weight": args_cli.direction_cart_weight,
            "direction_cart_vel_weight": args_cli.direction_cart_vel_weight,
            "direction_cart_threshold": args_cli.direction_cart_threshold,
            "direction_cart_boost": args_cli.direction_cart_boost,
            "direction_cart_gate": args_cli.direction_cart_gate,
            "direction_gate_angle_weight": args_cli.direction_gate_angle_weight,
            "direction_gate_vel_weight": args_cli.direction_gate_vel_weight,
            "direction_gate_threshold": args_cli.direction_gate_threshold,
            "action_prior_weight": args_cli.action_prior_weight,
            "prior_pole_kp": args_cli.prior_pole_kp,
            "prior_pole_kd": args_cli.prior_pole_kd,
            "prior_cart_kp": args_cli.prior_cart_kp,
            "prior_cart_kd": args_cli.prior_cart_kd,
            "edge_rescue_weight": args_cli.edge_rescue_weight,
            "edge_rescue_threshold": args_cli.edge_rescue_threshold,
            "edge_rescue_velocity_threshold": args_cli.edge_rescue_velocity_threshold,
            "edge_rescue_return_action": args_cli.edge_rescue_return_action,
            "edge_rescue_prior_suppression": args_cli.edge_rescue_prior_suppression,
            "edge_rescue_gate_scale": args_cli.edge_rescue_gate_scale,
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
