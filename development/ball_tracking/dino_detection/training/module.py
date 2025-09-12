from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..model.net import build_model


class HeatmapLitModule(pl.LightningModule):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg

        # Model
        self.model = build_model(cfg.get("model", {}))

        # Loss
        loss_type = cfg.get("training", {}).get("loss", "bce")
        if loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer params
        tcfg = cfg.get("training", {})
        self.lr = float(tcfg.get("lr", 3e-4))
        self.weight_decay = float(tcfg.get("weight_decay", 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # inference use
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch  # x: (B,3,H,W), y: (B,1,h,w)
        logits = self.model(x)
        # If output and target sizes mismatch (due to rounding), resize target
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
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        tcfg = self.cfg.get("training", {})
        max_epochs = int(tcfg.get("max_epochs", 30))
        warmup_epochs = int(tcfg.get("warmup_epochs", 1))
        if warmup_epochs > 0:
            # Simple linear warmup into cosine decay
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return float(epoch + 1) / float(max(1, warmup_epochs))
                # cosine from warmup_epochs..max_epochs
                progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))

            sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sch,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return opt
