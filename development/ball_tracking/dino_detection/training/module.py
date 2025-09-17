from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf


@dataclass
class LossConfig:
    name: str = "bce"


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2


@dataclass
class SchedulerConfig:
    warmup_epochs: int = 1


@dataclass
class LitModuleConfig:
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @classmethod
    def from_config(cls, cfg_like: Any) -> "LitModuleConfig":
        cfg_dict = _to_dict(cfg_like)
        loss_dict = _to_dict(cfg_dict.get("loss", {}))
        optimizer_dict = _to_dict(cfg_dict.get("optimizer", {}))
        scheduler_dict = _to_dict(cfg_dict.get("scheduler", {}))

        return cls(
            loss=LossConfig(name=str(loss_dict.get("name", LossConfig.name))),
            optimizer=OptimizerConfig(
                lr=float(optimizer_dict.get("lr", OptimizerConfig.lr)),
                weight_decay=float(optimizer_dict.get("weight_decay", OptimizerConfig.weight_decay)),
            ),
            scheduler=SchedulerConfig(
                warmup_epochs=int(scheduler_dict.get("warmup_epochs", SchedulerConfig.warmup_epochs))
            ),
        )


class HeatmapLitModule(pl.LightningModule):
    def __init__(self, model: nn.Module, cfg: LitModuleConfig, *, max_epochs: int) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.max_epochs = max_epochs
        self.save_hyperparameters({
            "lit_module": asdict(cfg),
            "max_epochs": max_epochs,
        })

        if cfg.loss.name.lower() == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss type: {cfg.loss.name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self.model(x)
        if logits.shape[-2:] != y.shape[-2:]:
            y = nn.functional.interpolate(y, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        loss = self.criterion(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self.model(x)
        if logits.shape[-2:] != y.shape[-2:]:
            y = nn.functional.interpolate(y, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        loss = self.criterion(logits, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )

        warmup_epochs = self.cfg.scheduler.warmup_epochs
        if warmup_epochs > 0 and self.max_epochs > 0:

            def lr_lambda(epoch: int) -> float:
                if epoch < warmup_epochs:
                    return float(epoch + 1) / float(max(1, warmup_epochs))
                progress = (epoch - warmup_epochs) / max(1, self.max_epochs - warmup_epochs)
                return 0.5 * (1.0 + math.cos(progress * math.pi))

            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return opt


def _to_dict(cfg_like: Any) -> Mapping[str, Any]:
    if isinstance(cfg_like, DictConfig):
        return OmegaConf.to_container(cfg_like, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg_like, Mapping):
        return cfg_like
    return {}


def build_lit_module(cfg_like: Any, model: nn.Module, *, max_epochs: int) -> HeatmapLitModule:
    lit_cfg = LitModuleConfig.from_config(cfg_like)
    return HeatmapLitModule(model, lit_cfg, max_epochs=max_epochs)


__all__ = [
    "HeatmapLitModule",
    "LitModuleConfig",
    "LossConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "build_lit_module",
]
