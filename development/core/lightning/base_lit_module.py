# filename: base_module.py
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

try:  # Optional dependency when running under Hydra/OmegaConf.
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - OmegaConf not installed in environment
    DictConfig = ()  # type: ignore
    OmegaConf = None  # type: ignore


class BaseLitModule(LightningModule):
    """
    汎用的な LightningModule ベースクラス
    モデル, 損失関数, 評価関数(PCKなど)を引数で注入して利用可能。
    """

    def __init__(
        self,
        config,
        model: nn.Module,
        loss_fn: nn.Module,
        metric_fns: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.model = model
        self.loss_fn = loss_fn
        self.metric_fns = metric_fns or {}

    # ====== Forward ======
    def forward(self, x):
        return self.model(x)

    # ====== Train step ======
    def training_step(self, batch, batch_idx):
        images, targets = self._split_batch(batch)
        preds = self(images)
        loss = self.loss_fn(preds, targets)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ====== Validation step ======
    def validation_step(self, batch, batch_idx):
        images, targets = self._split_batch(batch)
        preds = self(images)
        loss = self.loss_fn(preds, targets)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

        for name, fn in self.metric_fns.items():
            val = fn(preds, targets)
            self.log(f"val/{name}", val, prog_bar=True, on_epoch=True)

        return loss

    # ====== Test step ======
    def test_step(self, batch, batch_idx):
        images, targets = self._split_batch(batch)
        preds = self(images)
        loss = self.loss_fn(preds, targets)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True)

        for name, fn in self.metric_fns.items():
            val = fn(preds, targets)
            self.log(f"test/{name}", val, prog_bar=True, on_epoch=True)

        return loss

    # ====== Optimizer設定 ======
    def configure_optimizers(self):
        training_cfg = _to_dict(getattr(self.config, "training", {}))

        optimizer_cfg = training_cfg.get("optimizer")
        if optimizer_cfg:
            optimizer = _build_optimizer(self.parameters(), optimizer_cfg, training_cfg)
        else:
            lr = float(training_cfg.get("lr", 1e-3))
            weight_decay = float(training_cfg.get("weight_decay", 0.0))
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_cfg = training_cfg.get("lr_scheduler")
        if scheduler_cfg:
            scheduler_spec = _build_scheduler(optimizer, scheduler_cfg, training_cfg)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=5,
            )
            scheduler_spec = {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_spec,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_batch(self, batch: Any):
        if isinstance(batch, Mapping):
            if "inputs" not in batch:
                raise KeyError("Batch mapping must contain an 'inputs' key.")
            return batch["inputs"], batch.get("targets")
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise TypeError("Batch must be a mapping with 'inputs'/'targets' or a tuple of length 2.")


def _to_dict(cfg_like: Any) -> Dict[str, Any]:
    if cfg_like is None:
        return {}
    if OmegaConf is not None and isinstance(cfg_like, DictConfig):
        return OmegaConf.to_container(cfg_like, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg_like, Mapping):
        return dict(cfg_like)
    if hasattr(cfg_like, "__dict__"):
        return dict(vars(cfg_like))
    return {}


def _build_optimizer(parameters, cfg: Mapping[str, Any], fallback: Mapping[str, Any]):
    name = str(cfg.get("name", "adamw")).lower()
    lr = float(cfg.get("lr", cfg.get("learning_rate", fallback.get("lr", 1e-3))))
    weight_decay = float(cfg.get("weight_decay", fallback.get("weight_decay", 0.0)))
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    eps = float(cfg.get("eps", 1e-8))

    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        nesterov = bool(cfg.get("nesterov", False))
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    raise ValueError(f"Unsupported optimizer '{name}'.")


def _build_scheduler(optimizer, cfg: Mapping[str, Any], training_cfg: Mapping[str, Any]):
    name = str(cfg.get("name", "")).lower()
    interval = cfg.get("interval", "epoch")
    monitor = cfg.get("monitor", "val/loss")

    if name in {"cosine_warmup", "cosine_annealing_warmup"}:
        max_epochs = int(cfg.get("max_epochs", training_cfg.get("max_epochs", 1)))
        warmup_epochs = int(cfg.get("warmup_epochs", 0))
        min_lr = float(cfg.get("min_lr", 0.0))
        warmup_start_lr = float(cfg.get("warmup_start_lr", cfg.get("start_lr", 0.0)))

        base_lr = float(cfg.get("base_lr", cfg.get("lr", cfg.get("learning_rate", optimizer.param_groups[0]["lr"]))))
        return {
            "scheduler": _cosine_warmup_scheduler(
                optimizer,
                max_epochs=max_epochs,
                warmup_epochs=warmup_epochs,
                base_lr=base_lr,
                warmup_start_lr=warmup_start_lr,
                min_lr=min_lr,
            ),
            "interval": interval,
            "monitor": monitor,
        }

    if name in {"reduce_on_plateau", "plateau"}:
        mode = cfg.get("mode", "min")
        factor = float(cfg.get("factor", 0.1))
        patience = int(cfg.get("patience", 5))
        threshold = float(cfg.get("threshold", 1e-4))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
        )
        return {
            "scheduler": scheduler,
            "monitor": monitor,
            "interval": interval,
        }

    raise ValueError(f"Unsupported lr_scheduler '{name}'.")


def _cosine_warmup_scheduler(
    optimizer,
    *,
    max_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    warmup_start_lr: float,
    min_lr: float,
):
    max_epochs = max(1, int(max_epochs))
    warmup_epochs = max(0, int(warmup_epochs))
    base_lr = float(base_lr)
    warmup_start_lr = float(warmup_start_lr)
    min_lr = float(min_lr)

    def lr_lambda(epoch: int) -> float:
        if base_lr <= 0:
            return 1.0
        if warmup_epochs > 0 and epoch < warmup_epochs:
            progress = (epoch + 1) / float(warmup_epochs)
            start_factor = warmup_start_lr / base_lr
            return start_factor + (1.0 - start_factor) * progress
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        min_factor = min_lr / base_lr if base_lr > 0 else 0.0
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
