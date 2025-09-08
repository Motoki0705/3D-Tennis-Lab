from __future__ import annotations

from typing import Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from ..model.factory import build_sequence_model
from ..utils.normalization import FeatureStats, Standardizer


def _select_io(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Unsupported batch format. Expected (input_sequence, target_vector)")


class SequenceLitModule(pl.LightningModule):
    """LightningModule with normalization, optional delta mode and consistency loss."""

    def __init__(self, config: Any, feature_stats: Optional[FeatureStats] = None):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = config

        self.model = build_sequence_model(self.cfg)
        self.criterion = nn.MSELoss()

        self.standardizer = Standardizer(feature_stats) if feature_stats is not None else None
        # Loss weights
        self.lambda_consistency = float(getattr(self.cfg.training, "consistency_loss_weight", 0.0))
        self.predict_mode = getattr(self.cfg.training, "predict_mode", "absolute").lower()  # 'absolute'|'delta'

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def _prepare(
        self, input_seq: torch.Tensor, target_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Optionally standardize features
        if self.standardizer is not None:
            D = input_seq.shape[-1]
            inp = self.standardizer.normalize(input_seq.view(-1, D)).view_as(input_seq)
            tgt = self.standardizer.normalize(target_vec)
        else:
            inp, tgt = input_seq, target_vec

        last_state = inp[:, -1, :]  # (B, D)
        if self.predict_mode == "delta":
            # Predict residual over last state
            pred = self.model(inp)
            out = last_state + pred
        else:
            out = self.model(inp)
        return out, tgt, last_state

    def _consistency_loss(self, pred: torch.Tensor, last_state: torch.Tensor) -> torch.Tensor:
        # pred, last_state: (B, 6) with [x, y, vx, vy, ax, ay]
        if pred.size(-1) < 6:
            return torch.tensor(0.0, device=pred.device)
        x, y, vx, vy, ax, ay = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4], pred[:, 5]
        x0, y0, vx0, vy0, ax0, ay0 = (
            last_state[:, 0],
            last_state[:, 1],
            last_state[:, 2],
            last_state[:, 3],
            last_state[:, 4],
            last_state[:, 5],
        )
        # Velocity consistency: vx ~ x - x0; vy ~ y - y0
        c_vel = (vx - (x - x0)).pow(2).mean() + (vy - (y - y0)).pow(2).mean()
        # Acceleration consistency: ax ~ vx - vx0; ay ~ vy - vy0
        c_acc = (ax - (vx - vx0)).pow(2).mean() + (ay - (vy - vy0)).pow(2).mean()
        return c_vel + c_acc

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        input_seq, target_vec = _select_io(batch)
        pred_vec, tgt, last_state = self._prepare(input_seq, target_vec)
        loss_main = self.criterion(pred_vec, tgt)
        loss_cons = self._consistency_loss(pred_vec, last_state) if self.lambda_consistency > 0 else 0.0
        loss = loss_main + self.lambda_consistency * loss_cons
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_main", loss_main, on_step=True, on_epoch=True)
        if isinstance(loss_cons, torch.Tensor):
            self.log("train/loss_consistency", loss_cons, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        input_seq, target_vec = _select_io(batch)
        pred_vec, tgt, last_state = self._prepare(input_seq, target_vec)
        loss_main = self.criterion(pred_vec, tgt)
        loss_cons = self._consistency_loss(pred_vec, last_state) if self.lambda_consistency > 0 else 0.0
        loss = loss_main + self.lambda_consistency * loss_cons
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # Metrics (MAE per component)
        mae = (pred_vec - tgt).abs().mean(dim=0)
        for i, name in enumerate(["x", "y", "vx", "vy", "ax", "ay"][: pred_vec.size(-1)]):
            self.log(f"val/mae_{name}", mae[i], on_epoch=True)
        return loss

    def configure_optimizers(self):
        cfg_train = self.cfg.training
        optimizer = optim.AdamW(
            self.parameters(),
            lr=float(cfg_train.lr),
            weight_decay=float(cfg_train.weight_decay),
            betas=tuple(cfg_train.betas),
        )

        warmup_epochs = int(cfg_train.warmup_epochs)
        max_epochs = int(self.cfg.training.max_epochs)

        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs - warmup_epochs, eta_min=float(cfg_train.eta_min)
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=float(cfg_train.eta_min)
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss",
            },
        }
