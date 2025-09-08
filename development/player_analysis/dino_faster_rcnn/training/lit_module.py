from __future__ import annotations

from typing import Any, Dict

import torch
import pytorch_lightning as pl
from torch.optim import AdamW


class DetectionLitModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float = 1e-4, weight_decay: float = 1e-4):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])  # avoid saving full model twice
        self.example_input_array = None

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        images, targets = batch["images"], batch["targets"]
        losses: Dict[str, torch.Tensor] = self.model(images, targets)  # type: ignore
        loss = sum(losses.values())
        for k, v in losses.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=(k == "loss_classifier"))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        images, targets = batch["images"], batch["targets"]
        losses: Dict[str, torch.Tensor] = self.model(images, targets)  # Faster R-CNN returns losses in train mode
        loss = sum(losses.values())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
