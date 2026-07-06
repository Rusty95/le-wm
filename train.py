import os
import time
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


class TrainingDiagnosticsCallback(pl.Callback):
    """Log optimizer health and throughput without changing training."""

    def __init__(self):
        self._last_batch_end = None

    def on_train_start(self, trainer, pl_module):
        self._last_batch_end = time.perf_counter()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        grads = [param.grad.detach().norm(2) for param in pl_module.parameters() if param.grad is not None]
        if grads:
            grad_norm = torch.stack(grads).norm(2)
            pl_module.log("optimization/grad_norm", grad_norm, on_step=True, on_epoch=False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        if self._last_batch_end is not None:
            elapsed = max(now - self._last_batch_end, 1e-9)
            batch_size = int(batch["action"].shape[0])
            pl_module.log("performance/batches_per_sec", 1.0 / elapsed, on_step=True, on_epoch=False)
            pl_module.log(
                "performance/samples_per_sec",
                batch_size / elapsed,
                on_step=True,
                on_epoch=False,
            )
        self._last_batch_end = now


class BalancedInterleaveDataset:
    """Interleave multiple datasets with equal per-dataset sampling weight."""

    def __init__(self, datasets):
        if not datasets:
            raise ValueError("BalancedInterleaveDataset requires at least one dataset.")
        self.datasets = list(datasets)
        self._per_dataset_len = min(len(dataset) for dataset in self.datasets)
        self._transform = None

    @property
    def column_names(self):
        return self.datasets[0].column_names

    @property
    def lengths(self):
        return np.concatenate([dataset.lengths for dataset in self.datasets])

    @property
    def offsets(self):
        return np.concatenate([dataset.offsets for dataset in self.datasets])

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value
        for dataset in self.datasets:
            dataset.transform = value

    def __len__(self):
        return self._per_dataset_len * len(self.datasets)

    def _loc(self, idx):
        if idx < 0:
            idx += len(self)
        dataset_idx = idx % len(self.datasets)
        local_idx = (idx // len(self.datasets)) % self._per_dataset_len
        return dataset_idx, local_idx

    def __getitem__(self, idx):
        dataset_idx, local_idx = self._loc(idx)
        return self.datasets[dataset_idx][local_idx]

    def __getitems__(self, indices):
        return [self[idx] for idx in indices]

    def get_col_data(self, col):
        balanced = []
        min_rows = min(len(dataset.get_col_data(col)) for dataset in self.datasets)
        for dataset in self.datasets:
            balanced.append(dataset.get_col_data(col)[:min_rows])
        return np.concatenate(balanced, axis=0)

    def get_dim(self, col):
        return self.datasets[0].get_dim(col)


def resolve_dataset_location(dataset_name: str, cache_dir: str | None):
    """Make local HDF5 dataset resolution tolerant to common env/config slips."""
    resolved_cache = Path(cache_dir).expanduser() if cache_dir else None

    # load_dataset(cache_dir=...) appends "datasets" internally. If the shell
    # still points LOCAL_DATASET_DIR at that leaf, use its parent as the cache root.
    if resolved_cache is not None and resolved_cache.name == "datasets":
        resolved_cache = resolved_cache.parent

    if resolved_cache is None:
        datasets_dir = Path(os.environ.get("STABLEWM_HOME", "~/.stable_worldmodel")).expanduser() / "datasets"
    else:
        datasets_dir = resolved_cache / "datasets"

    name_path = Path(dataset_name)
    if name_path.exists():
        return str(name_path), str(resolved_cache) if resolved_cache else cache_dir

    local = name_path if name_path.is_absolute() else datasets_dir / name_path
    if local.exists():
        return dataset_name, str(resolved_cache) if resolved_cache else cache_dir

    if name_path.suffix == "":
        h5_name = f"{dataset_name}.h5"
        h5_local = datasets_dir / h5_name
        if h5_local.exists():
            return h5_name, str(resolved_cache) if resolved_cache else cache_dir

    return dataset_name, str(resolved_cache) if resolved_cache else cache_dir


def load_train_dataset(dataset_cfg, cache_dir):
    dataset_cfg = dict(dataset_cfg)
    dataset_names = dataset_cfg.pop("names", None)
    balance = dataset_cfg.pop("balance", None)

    if dataset_names is None:
        dataset_name = dataset_cfg.pop("name")
        dataset_name, cache_dir = resolve_dataset_location(dataset_name, cache_dir)
        return swm.data.load_dataset(
            dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
        )

    datasets = []
    for dataset_name in dataset_names:
        resolved_name, resolved_cache_dir = resolve_dataset_location(dataset_name, cache_dir)
        datasets.append(
            swm.data.load_dataset(
                resolved_name, transform=None, cache_dir=resolved_cache_dir, **dataset_cfg
            )
        )

    if balance == "interleave":
        return BalancedInterleaveDataset(datasets)

    from stable_worldmodel.data.dataset import ConcatDataset

    return ConcatDataset(datasets)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss. External disturbances are not part of the policy action, so
    # the transition on which an impulse is injected can be excluded while
    # retaining all subsequent recovery dynamics.
    pred_error = (pred_emb - tgt_emb).pow(2).mean(dim=-1)
    if "prediction_valid" in batch:
        pred_valid = batch["prediction_valid"][:, : pred_emb.shape[1]].to(
            device=pred_error.device,
            dtype=pred_error.dtype,
        )
        output["pred_loss"] = (pred_error * pred_valid).sum() / pred_valid.sum().clamp_min(1.0)
        output["pred_valid_fraction"] = pred_valid.mean()
    else:
        pred_valid = torch.ones_like(pred_error)
        output["pred_loss"] = pred_error.mean()

    valid_count = pred_valid.sum().clamp_min(1.0)
    pred_cosine = F.cosine_similarity(pred_emb, tgt_emb, dim=-1)
    tgt_power = tgt_emb.pow(2).mean(dim=-1)
    output["pred_rmse"] = output["pred_loss"].sqrt()
    output["pred_relative_rmse"] = (
        output["pred_loss"] / ((tgt_power * pred_valid).sum() / valid_count).clamp_min(1e-12)
    ).sqrt()
    output["pred_cosine_similarity"] = (pred_cosine * pred_valid).sum() / valid_count
    output["embedding_mean"] = emb.mean()
    output["embedding_std"] = emb.std()
    output["embedding_norm"] = emb.norm(dim=-1).mean()
    output["prediction_norm"] = pred_emb.norm(dim=-1).mean()
    output["target_norm"] = tgt_emb.norm(dim=-1).mean()
    output["latent_step_norm"] = (emb[:, 1:] - emb[:, :-1]).norm(dim=-1).mean()
    output["action_abs_mean"] = batch["action"].abs().mean()
    output["action_std"] = batch["action"].std()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    metrics_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if isinstance(v, torch.Tensor) and v.numel() == 1
    }
    self.log_dict(
        metrics_dict,
        on_step=stage == "fit",
        on_epoch=True,
        sync_dist=True,
    )
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = load_train_dataset(dataset_cfg, cache_dir)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col == "prediction_valid":
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    loggers = []
    if cfg.wandb.enabled:
        loggers.append(WandbLogger(**cfg.wandb.config))
    if cfg.tensorboard.enabled:
        loggers.append(TensorBoardLogger(**cfg.tensorboard.config))
    for logger in loggers:
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )
    callbacks = [
        object_dump_callback,
        LearningRateMonitor(logging_interval="step"),
        TrainingDiagnosticsCallback(),
    ]

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=loggers or None,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
