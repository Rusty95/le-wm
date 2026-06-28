"""Shared helpers for deploying LeWM inside IsaacLab-side scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from transformers import ViTConfig, ViTModel


REPO_DIR = Path(__file__).resolve().parents[1]
RL_REPO = Path("/home/hall/code/RL-Learning-BasedOn-IsaacLab/source/rl_lab_learning")
DEFAULT_CACHE_DIR = Path("/home/hall/code/.stable-wm")
DEFAULT_CHECKPOINT = "lewm_isaaclab_mixed_balanced/weights_epoch_100.pt"
DEFAULT_ACTION_STATS_H5 = [
    Path("/home/hall/code/.stable-wm/datasets/isaaclab_random_100k.h5"),
    Path("/home/hall/code/.stable-wm/datasets/isaaclab_policy_camera_50k.h5"),
]

for repo in (REPO_DIR, RL_REPO):
    if repo.exists() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from jepa import JEPA  # noqa: E402
from module import ARPredictor, Embedder, MLP  # noqa: E402


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def first_env(value: Any, num_envs: int = 1) -> np.ndarray:
    arr = to_numpy(value)
    if arr.ndim > 0 and num_envs == 1:
        return arr[0]
    return arr


def get_nested(obs: Any, key: str) -> Any:
    if isinstance(obs, dict):
        if key not in obs:
            raise KeyError(f"Observation dict has keys {list(obs.keys())}, missing {key!r}")
        return obs[key]
    return obs


def get_pixels(obs: Any, pixel_key: str = "policy", num_envs: int = 1) -> np.ndarray:
    pixels = first_env(get_nested(obs, pixel_key), num_envs=num_envs)
    if pixels.ndim == 3 and pixels.shape[0] in (1, 3, 4):
        pixels = np.moveaxis(pixels, 0, -1)
    if pixels.shape[-1] == 4:
        pixels = pixels[..., :3]
    return pixels


def get_cartpole_state(env) -> torch.Tensor:
    """Return [pole_pos, pole_vel, cart_pos, cart_vel]."""
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


def load_action_stats(paths: list[Path], max_rows_per_file: int = 50000) -> tuple[torch.Tensor, torch.Tensor]:
    arrays = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Action stats source not found: {path}")
        with h5py.File(path, "r") as f:
            action = np.asarray(f["action"][:max_rows_per_file], dtype=np.float32)
        arrays.append(action.reshape(action.shape[0], -1))
    data = torch.from_numpy(np.concatenate(arrays, axis=0))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True)
    std = data.std(0, keepdim=True).clamp_min(1e-6)
    return mean, std


def resolve_checkpoint(checkpoint: str | Path, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    path = Path(checkpoint).expanduser()
    if path.exists():
        return path
    return cache_dir / "checkpoints" / path


def build_lewm_model(action_dim: int, img_size: int = 224) -> JEPA:
    encoder_cfg = ViTConfig(
        image_size=img_size,
        patch_size=14,
        num_channels=3,
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
        qkv_bias=True,
    )
    encoder = ViTModel(encoder_cfg, add_pooling_layer=False)
    predictor = ARPredictor(
        num_frames=3,
        input_dim=192,
        hidden_dim=192,
        output_dim=192,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=action_dim, emb_dim=192)
    projector = MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm_fn=torch.nn.BatchNorm1d)
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )


def convert_encoder_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    replacements = (
        (".attention.q_proj.", ".attention.attention.query."),
        (".attention.k_proj.", ".attention.attention.key."),
        (".attention.v_proj.", ".attention.attention.value."),
        (".attention.o_proj.", ".attention.output.dense."),
        (".mlp.fc1.", ".intermediate.dense."),
        (".mlp.fc2.", ".output.dense."),
    )
    for key, value in state.items():
        new_key = key
        if key.startswith("encoder.layers."):
            new_key = "encoder.encoder.layer." + key[len("encoder.layers.") :]
            for old, new in replacements:
                new_key = new_key.replace(old, new)
        converted[new_key] = value
    return converted


def load_lewm(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    action_dim: int = 1,
    img_size: int = 224,
    device: torch.device | str = "cpu",
) -> JEPA:
    checkpoint_path = resolve_checkpoint(checkpoint, cache_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LeWM checkpoint not found: {checkpoint_path}")
    model = build_lewm_model(action_dim=action_dim, img_size=img_size)
    state = torch.load(checkpoint_path, map_location="cpu")
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(convert_encoder_keys(state))
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def preprocess_pixels(frames: np.ndarray | list[np.ndarray], img_size: int, device: torch.device | str) -> torch.Tensor:
    pixels = torch.as_tensor(np.asarray(frames), dtype=torch.float32)
    if pixels.ndim != 4:
        raise ValueError(f"Expected raw pixels as (T,H,W,C) or (T,C,H,W), got {tuple(pixels.shape)}")
    if pixels.shape[-1] in (1, 3, 4):
        pixels = pixels[..., :3].permute(0, 3, 1, 2)
    elif pixels.shape[1] == 4:
        pixels = pixels[:, :3]
    elif pixels.shape[1] != 3:
        raise ValueError(f"Could not identify channel dimension in pixel shape {tuple(pixels.shape)}")

    if pixels.max() > 2.0:
        pixels = pixels / 255.0
    pixels = torch.nn.functional.interpolate(
        pixels,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=pixels.dtype, device=pixels.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=pixels.dtype, device=pixels.device).view(1, 3, 1, 1)
    pixels = (pixels - mean) / std
    return pixels.to(device)


def normalize_actions(
    actions: np.ndarray | list[np.ndarray] | torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    action = torch.as_tensor(np.asarray(actions), dtype=torch.float32) if not torch.is_tensor(actions) else actions.float()
    if action.ndim == 1:
        action = action.unsqueeze(-1)
    action = action.to(device)
    mean = mean.to(device)
    std = std.to(device)
    return torch.nan_to_num((action - mean) / std, 0.0)


def denormalize_actions(actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return actions * std.to(actions.device) + mean.to(actions.device)


def infer_history_size(model: torch.nn.Module) -> int:
    predictor = getattr(model, "predictor", None)
    pos_embedding = getattr(predictor, "pos_embedding", None)
    if pos_embedding is None:
        raise ValueError("Could not infer history size from model.predictor.pos_embedding.")
    return int(pos_embedding.shape[1])


def rollout_predictions(
    model: torch.nn.Module,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
    max_horizon: int,
) -> torch.Tensor:
    emb_list = list(emb[:, :history_size].unbind(dim=1))
    for step in range(max_horizon):
        lo = max(0, history_size + step - history_size)
        ctx_emb = torch.stack(emb_list[lo:], dim=1)
        ctx_act = act_emb[:, lo : history_size + step]
        emb_list.append(model.predict(ctx_emb, ctx_act)[:, -1])
    return torch.stack(emb_list[history_size:], dim=1)


def rollout_predictions_online(
    model: torch.nn.Module,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
    max_horizon: int,
) -> torch.Tensor:
    """Roll out with candidate[0] affecting the first predicted latent.

    ``emb`` contains the current visual history with length ``history_size``.
    ``act_emb`` must contain ``history_size - 1 + max_horizon`` actions:
    the last ``history_size - 1`` executed actions followed by the candidate
    future actions.  This alignment is useful for online MPC because the first
    candidate action is the action that will be sent to ``env.step`` now.
    """
    expected = history_size - 1 + max_horizon
    if act_emb.shape[1] != expected:
        raise ValueError(
            f"Online rollout expected {expected} action embeddings, got {act_emb.shape[1]} "
            f"(history_size={history_size}, max_horizon={max_horizon})."
        )

    emb_list = list(emb[:, :history_size].unbind(dim=1))
    preds = []
    for step in range(max_horizon):
        ctx_emb = torch.stack(emb_list[step : step + history_size], dim=1)
        ctx_act = act_emb[:, step : step + history_size]
        pred = model.predict(ctx_emb, ctx_act)[:, -1]
        emb_list.append(pred)
        preds.append(pred)
    return torch.stack(preds, dim=1)


class StateProbe(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 192,
        hidden_dim: int = 256,
        output_dim: int = 4,
        target_mean: torch.Tensor | None = None,
        target_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        if target_mean is None:
            target_mean = torch.zeros(1, output_dim)
        if target_std is None:
            target_std = torch.ones(1, output_dim)
        self.register_buffer("target_mean", target_mean.float().reshape(1, output_dim))
        self.register_buffer("target_std", target_std.float().reshape(1, output_dim).clamp_min(1e-6))

    def forward_normalized(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(emb) * self.target_std + self.target_mean


def load_state_probe(path: Path, device: torch.device | str) -> StateProbe:
    payload = torch.load(path, map_location="cpu")
    config = payload.get("config", {})
    target_mean = payload.get("target_mean")
    target_std = payload.get("target_std")
    probe = StateProbe(
        input_dim=int(config.get("input_dim", 192)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        output_dim=int(config.get("output_dim", 4)),
        target_mean=target_mean,
        target_std=target_std,
    )
    state = payload["state_dict"] if "state_dict" in payload else payload
    probe.load_state_dict(state)
    probe.to(device)
    probe.eval()
    return probe
