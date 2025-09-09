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
        batch_size = len(images)
        losses: Dict[str, torch.Tensor] = self.model(images, targets)  # type: ignore
        loss = sum(losses.values())
        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=(k == "loss_classifier"),
                batch_size=batch_size,
            )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        images, targets = batch["images"], batch["targets"]
        batch_size = len(images)

        # Get predictions (model is in eval mode by default in validation_step)
        with torch.no_grad():
            predictions = self.model(images)

        # Calculate loss - temporarily switch to train mode
        self.model.train()
        losses: Dict[str, torch.Tensor] = self.model(images, targets)
        self.model.eval()  # Switch back to eval mode

        loss = sum(losses.values())
        self.log("val-loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # Return data for visualization callback
        # Detach and move to CPU to avoid memory leaks in the callback
        return {
            "images": [img.cpu() for img in images],
            "targets": [{k: v.cpu() for k, v in t.items()} for t in targets],
            "predictions": [{k: v.cpu() for k, v in p.items()} for p in predictions],
        }

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
