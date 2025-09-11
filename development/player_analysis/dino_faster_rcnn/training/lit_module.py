from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class DetectionLitModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        lr_scheduler_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])  # avoid saving full model twice
        self.example_input_array = None
        self._optimizer_cfg = optimizer_cfg or {}
        self._lr_scheduler_cfg = lr_scheduler_cfg or {}

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
        # Log current LR (first param group)
        if self.trainer is not None and self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0].get("lr", None)
            if lr is not None:
                self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False, batch_size=batch_size)
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

    def _build_optimizer(self, params) -> Optimizer:
        name = str(self._optimizer_cfg.get("name", "adamw")).lower()
        lr = float(self._optimizer_cfg.get("lr", self.hparams.lr))
        weight_decay = float(self._optimizer_cfg.get("weight_decay", self.hparams.weight_decay))

        if name in ("adamw", "adamw_torch"):
            betas = tuple(self._optimizer_cfg.get("betas", (0.9, 0.999)))
            eps = float(self._optimizer_cfg.get("eps", 1e-8))
            return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        elif name == "sgd":
            momentum = float(self._optimizer_cfg.get("momentum", 0.9))
            nesterov = bool(self._optimizer_cfg.get("nesterov", True))
            return SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        else:
            # default fallback
            return AdamW(params, lr=lr, weight_decay=weight_decay)

    def _build_scheduler(self, optimizer: Optimizer):
        cfg = {k: v for k, v in self._lr_scheduler_cfg.items()}  # shallow copy
        name = str(cfg.get("name", "cosine_warmup")).lower()
        interval = str(cfg.get("interval", "epoch")).lower()

        if name in ("cosine_warmup", "warmup_cosine"):
            warmup_epochs = int(cfg.get("warmup_epochs", 0))
            warmup_start_lr = float(cfg.get("warmup_start_lr", 0.0))
            min_lr = float(cfg.get("min_lr", 0.0))
            max_epochs = int(cfg.get("max_epochs", 0))
            if max_epochs <= 0 and self.trainer is not None and self.trainer.max_epochs is not None:
                max_epochs = int(self.trainer.max_epochs)

            schedulers = []
            milestones = []

            base_lr = optimizer.param_groups[0]["lr"]
            if warmup_epochs > 0:
                start_factor = warmup_start_lr / max(base_lr, 1e-12)
                start_factor = max(0.0, min(1.0, start_factor))
                warmup = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs)
                schedulers.append(warmup)
                milestones.append(warmup_epochs)

            t_max = max(1, max_epochs - warmup_epochs) if max_epochs else 1
            cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
            if warmup_epochs > 0:
                sched = SequentialLR(optimizer, schedulers=[schedulers[0], cosine], milestones=milestones)
            else:
                sched = cosine

            return {
                "scheduler": sched,
                "interval": interval,  # "epoch" or "step"
                "frequency": 1,
                "monitor": None,
            }

        # No scheduler
        return None

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params)

        sched_cfg = self._build_scheduler(optimizer)
        if sched_cfg is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": sched_cfg,
            }
