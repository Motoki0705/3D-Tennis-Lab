from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

import pytorch_lightning as pl
import torch
import torch.optim as optim

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - OmegaConf not installed
    DictConfig = ()  # type: ignore
    OmegaConf = None  # type: ignore

logger = logging.getLogger(__name__)


def _to_dict(cfg_like: Any) -> Dict[str, Any]:
    if cfg_like is None:
        return {}
    if OmegaConf is not None and isinstance(cfg_like, DictConfig):  # type: ignore[arg-type]
        container = OmegaConf.to_container(cfg_like, resolve=True)
        return dict(container) if isinstance(container, Mapping) else {}
    if isinstance(cfg_like, Mapping):
        return dict(cfg_like)
    if hasattr(cfg_like, "__dict__"):
        return dict(vars(cfg_like))
    return {}


class DinoDetrLitModule(pl.LightningModule):
    """Lightning module wrapper for the DINO-DETR detector."""

    def __init__(
        self,
        *,
        cfg: Any,
        model,
        loss_fn,
        metric_fns: Mapping[str, Any] | None = None,
        postprocessors: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg
        self.model = model
        self.criterion = loss_fn
        self.metric_fns = dict(metric_fns or {})
        self.postprocessors = dict(postprocessors or {})

        training_cfg = _to_dict(getattr(cfg, "training", {}))
        self._training_cfg = training_cfg
        optimizer_cfg = _to_dict(training_cfg.get("optimizer"))

        lr_default = optimizer_cfg.get("lr", optimizer_cfg.get("learning_rate", 1.0e-4))
        self.lr = float(lr_default)
        self.weight_decay = float(optimizer_cfg.get("weight_decay", 1.0e-4))
        betas = optimizer_cfg.get("betas", (0.9, 0.999))
        if isinstance(betas, (list, tuple)) and len(betas) >= 2:
            self.betas = (float(betas[0]), float(betas[1]))
        else:
            self.betas = (0.9, 0.999)

        self.max_epochs = int(training_cfg.get("max_epochs", 50))
        self.warmup_epochs = int(training_cfg.get("warmup_epochs", 0))
        self.cosine_eta_min = float(training_cfg.get("eta_min", 1.0e-6))

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def forward(self, images: List[torch.Tensor]):  # type: ignore[override]
        return self.model(images)

    def training_step(self, batch, batch_idx: int):
        images, targets = batch
        outputs = self.model(images)
        norm_targets = self._normalise_targets(images, targets)
        loss_dict = self.criterion(outputs, norm_targets)
        total_loss = self._sum_and_log_losses(
            loss_dict,
            prefix="train",
            batch_size=len(images),
            on_step=True,
            on_epoch=True,
        )
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        self._log_metrics(outputs, targets, prefix="train", on_step=True)
        return total_loss

    def validation_step(self, batch, batch_idx: int):
        images, targets = batch
        outputs = self.model(images)
        norm_targets = self._normalise_targets(images, targets)
        loss_dict = self.criterion(outputs, norm_targets)
        val_loss = self._sum_and_log_losses(
            loss_dict,
            prefix="val",
            batch_size=len(images),
            on_step=False,
            on_epoch=True,
        )
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(images))
        self._log_metrics(outputs, targets, prefix="val", on_step=False)
        return val_loss

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=self.betas)

        warmup_epochs = max(0, int(self.warmup_epochs))
        max_epochs = max(1, int(self.max_epochs))
        cosine_eta_min = float(self.cosine_eta_min)

        schedulers = []
        milestones = []
        if warmup_epochs > 0:
            warmup = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            schedulers.append(warmup)
            milestones.append(warmup_epochs)

        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, max_epochs - warmup_epochs),
            eta_min=cosine_eta_min,
        )
        sched = (
            optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers + [cosine], milestones=milestones)
            if schedulers
            else cosine
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise_targets(self, images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        normalised: List[Dict[str, torch.Tensor]] = []
        for img, tgt in zip(images, targets):
            height, width = img.shape[-2], img.shape[-1]
            boxes = tgt.get("boxes")
            if torch.is_tensor(boxes) and boxes.numel() > 0:
                x_min, y_min, x_max, y_max = boxes.unbind(-1)
                cx = (x_min + x_max) / 2.0 / width
                cy = (y_min + y_max) / 2.0 / height
                w = (x_max - x_min) / width
                h = (y_max - y_min) / height
                boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1)
            else:
                device = img.device
                boxes_cxcywh = torch.zeros((0, 4), dtype=torch.float32, device=device)

            labels = tgt.get("labels")
            if torch.is_tensor(labels):
                labels_t = labels.to(torch.int64)
            else:
                labels_t = torch.zeros((0,), dtype=torch.int64, device=img.device)
            normalised.append({"boxes": boxes_cxcywh, "labels": labels_t})
        return normalised

    def _sum_and_log_losses(
        self, loss_dict, *, prefix: str, batch_size: int, on_step: bool, on_epoch: bool
    ) -> torch.Tensor:
        total_loss: torch.Tensor | None = None
        for name, value in loss_dict.items():
            if not torch.is_tensor(value):
                continue
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                batch_size=batch_size,
            )
            total_loss = value if total_loss is None else (total_loss + value)
        if total_loss is None:
            try:
                param = next(self.model.parameters())
                total_loss = param.new_zeros(())
            except StopIteration:
                total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        return total_loss

    def _log_metrics(self, outputs, targets, *, prefix: str, on_step: bool) -> None:
        for name, fn in self.metric_fns.items():
            try:
                value = fn(outputs, targets)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Metric '%s' failed: %s", name, exc)
                continue
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
            )


__all__ = ["DinoDetrLitModule"]
