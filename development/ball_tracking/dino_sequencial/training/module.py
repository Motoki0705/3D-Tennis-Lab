from __future__ import annotations
from typing import Any
import math
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from ..model.net import SequenceHeatmapNet


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4


@dataclass
class SchedulerConfig:
    warmup_epochs: int = 5


@dataclass
class LitModuleConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


class HeatmapLitModule(pl.LightningModule):
    def __init__(self, net: SequenceHeatmapNet, cfg: LitModuleConfig, max_epochs: int) -> None:
        super().__init__()
        self.net = net
        self.cfg = cfg
        self.max_epochs = max_epochs
        self.save_hyperparameters(ignore=["net"])
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        img_seq, target_heatmap_seq = batch
        pred_heatmap_seq = self(img_seq)
        loss = self.loss_fn(pred_heatmap_seq, target_heatmap_seq)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        img_seq, target_heatmap_seq = batch
        pred_heatmap_seq = self(img_seq)
        loss = self.loss_fn(pred_heatmap_seq, target_heatmap_seq)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.cfg.optimizer.learning_rate)

        warmup_epochs = self.cfg.scheduler.warmup_epochs
        if warmup_epochs > 0 and self.max_epochs > warmup_epochs:

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


def build_lit_module(model: SequenceHeatmapNet, lit_module_cfg: dict, max_epochs: int) -> HeatmapLitModule:
    optimizer_cfg = OptimizerConfig(**lit_module_cfg.get("optimizer", {}))
    scheduler_cfg = SchedulerConfig(**lit_module_cfg.get("scheduler", {}))
    cfg = LitModuleConfig(optimizer=optimizer_cfg, scheduler=scheduler_cfg)
    return HeatmapLitModule(net=model, cfg=cfg, max_epochs=max_epochs)


__all__ = ["HeatmapLitModule", "build_lit_module", "LitModuleConfig"]
