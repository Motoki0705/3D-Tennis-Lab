from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
import torch.optim as optim
import pytorch_lightning as pl

from typing import NamedTuple


logger = logging.getLogger(__name__)


class _DetrArgs(NamedTuple):
    # backbone / transformer
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    dim_feedforward: int = 2048
    enc_layers: int = 6
    dec_layers: int = 6
    pre_norm: bool = False

    # matcher costs
    set_cost_class: float = 1.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0

    # loss weights
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    mask_loss_coef: float = 1.0
    dice_loss_coef: float = 1.0

    # model settings
    num_queries: int = 100
    aux_loss: bool = True
    masks: bool = False
    frozen_weights: any = None
    lr_backbone: float = 0.0

    # misc
    dataset_file: str = "coco"
    device: str = "cuda"


class DinoDetrLitModule(pl.LightningModule):
    """
    LightningModule for training a DINO-DETR-like detector.

    This module expects:
      - inputs: list[Tensor(3,H,W)] in [0,1]
      - targets: list[dict] with keys 'boxes' (Nx4, xyxy), 'labels' (N)

    The underlying model is provided by `build_detection_model`, which by default
    attempts to use torchvision's DETR to emulate similar training behavior.
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        # Require local DETR implementation (model/detr.py)
        self.criterion = None
        self.postprocessors = None
        self.use_local_detr = True
        from ..model.detr import build as build_local_detr

        mcfg = getattr(self.cfg, "model", {})
        args = _DetrArgs(
            hidden_dim=int(getattr(mcfg, "hidden_dim", 256)),
            dropout=float(getattr(mcfg, "dropout", 0.1)),
            nheads=int(getattr(mcfg, "nheads", 8)),
            dim_feedforward=int(getattr(mcfg, "dim_feedforward", 2048)),
            enc_layers=int(getattr(mcfg, "enc_layers", 6)),
            dec_layers=int(getattr(mcfg, "dec_layers", 6)),
            pre_norm=bool(getattr(mcfg, "pre_norm", False)),
            set_cost_class=float(getattr(mcfg, "set_cost_class", 1.0)),
            set_cost_bbox=float(getattr(mcfg, "set_cost_bbox", 5.0)),
            set_cost_giou=float(getattr(mcfg, "set_cost_giou", 2.0)),
            bbox_loss_coef=float(getattr(mcfg, "bbox_loss_coef", 5.0)),
            giou_loss_coef=float(getattr(mcfg, "giou_loss_coef", 2.0)),
            mask_loss_coef=float(getattr(mcfg, "mask_loss_coef", 1.0)),
            dice_loss_coef=float(getattr(mcfg, "dice_loss_coef", 1.0)),
            num_queries=int(getattr(mcfg, "num_queries", 100)),
            aux_loss=bool(getattr(mcfg, "aux_loss", True)),
            masks=bool(getattr(mcfg, "masks", False)),
            frozen_weights=getattr(mcfg, "frozen_weights", None),
            lr_backbone=float(getattr(mcfg, "lr_backbone", 0.0)),
            dataset_file=str(getattr(mcfg, "dataset_file", "coco")),
            device=str(getattr(self.cfg, "device", "cuda")),
        )
        # Extra: DINOv3 backbone config (repo/model name)
        # These are read in build_dino_backbone via args
        setattr(args, "dino_repo_dir", str(getattr(mcfg, "dino_repo_dir", "third_party/dinov3")))
        setattr(args, "dino_model_name", str(getattr(mcfg, "dino_model_name", "dinov3_vitl16")))

        model, criterion, postprocessors = build_local_detr(args)
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors

        # Optimizer params from model config
        self.lr = float(getattr(self.cfg.model, "lr", 1e-4))
        self.weight_decay = float(getattr(self.cfg.model, "weight_decay", 1e-4))
        betas = getattr(self.cfg.model, "betas", (0.9, 0.999))
        self.betas = (float(betas[0]), float(betas[1])) if isinstance(betas, (list, tuple)) else (0.9, 0.999)

    def forward(self, images: List[torch.Tensor]):  # type: ignore[override]
        return self.model(images)

    def training_step(self, batch, batch_idx: int):
        images, targets = batch
        # Model may return dict of losses when in train mode with targets provided
        if self.use_local_detr and self.criterion is not None:
            outputs = self.model(images)  # model expects list or NestedTensor internally
            # Convert targets boxes to normalized cxcywh per image size
            norm_targets: List[Dict[str, torch.Tensor]] = []
            for img, tgt in zip(images, targets):
                H, W = img.shape[-2], img.shape[-1]
                boxes = tgt.get("boxes")
                if torch.is_tensor(boxes) and boxes.numel() > 0:
                    x_min, y_min, x_max, y_max = boxes.unbind(-1)
                    cx = (x_min + x_max) / 2.0 / W
                    cy = (y_min + y_max) / 2.0 / H
                    w = (x_max - x_min) / W
                    h = (y_max - y_min) / H
                    boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1)
                else:
                    boxes_cxcywh = torch.zeros((0, 4), device=img.device, dtype=torch.float32)
                labels = tgt.get("labels")
                labels = (
                    labels.to(torch.int64)
                    if torch.is_tensor(labels)
                    else torch.zeros((0,), dtype=torch.int64, device=img.device)
                )
                norm_targets.append({"boxes": boxes_cxcywh, "labels": labels})

            loss_dict = self.criterion(outputs, norm_targets)
            loss = None
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=False)
                    loss = v if loss is None else (loss + v)
            if loss is None:
                loss = torch.tensor(0.0, device=images[0].device)
        else:
            out = self.model(images, targets)
            if isinstance(out, dict) and out:
                loss = None
                for k, v in out.items():
                    if torch.is_tensor(v):
                        self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=(k == "loss"))
                        loss = v if loss is None else (loss + v)
                if loss is None:
                    loss = torch.tensor(0.0, device=images[0].device)
            else:
                loss = torch.tensor(0.0, device=images[0].device)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        images, targets = batch
        if self.use_local_detr and self.criterion is not None:
            outputs = self.model(images)
            norm_targets: List[Dict[str, torch.Tensor]] = []
            for img, tgt in zip(images, targets):
                H, W = img.shape[-2], img.shape[-1]
                boxes = tgt.get("boxes")
                if torch.is_tensor(boxes) and boxes.numel() > 0:
                    x_min, y_min, x_max, y_max = boxes.unbind(-1)
                    cx = (x_min + x_max) / 2.0 / W
                    cy = (y_min + y_max) / 2.0 / H
                    w = (x_max - x_min) / W
                    h = (y_max - y_min) / H
                    boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1)
                else:
                    boxes_cxcywh = torch.zeros((0, 4), device=img.device, dtype=torch.float32)
                labels = tgt.get("labels")
                labels = (
                    labels.to(torch.int64)
                    if torch.is_tensor(labels)
                    else torch.zeros((0,), dtype=torch.int64, device=img.device)
                )
                norm_targets.append({"boxes": boxes_cxcywh, "labels": labels})

            loss_dict = self.criterion(outputs, norm_targets)
            val_loss = None
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False)
                    val_loss = v if val_loss is None else (val_loss + v)
            if val_loss is None:
                val_loss = torch.tensor(0.0, device=images[0].device)
        else:
            out = self.model(images, targets)
            val_loss = None
            if isinstance(out, dict) and out:
                for k, v in out.items():
                    if torch.is_tensor(v):
                        self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=(k == "loss"))
                        val_loss = v if val_loss is None else (val_loss + v)
            if val_loss is None:
                val_loss = torch.tensor(0.0, device=images[0].device)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=self.betas)

        tcfg = self.cfg.training
        max_epochs = int(getattr(tcfg, "max_epochs", 50))
        warmup_epochs = int(getattr(tcfg, "warmup_epochs", 0))
        cosine_eta_min = float(getattr(tcfg, "eta_min", 1e-6))

        scheds = []
        milestones = []
        if warmup_epochs > 0:
            warmup = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )
            scheds.append(warmup)
            milestones.append(warmup_epochs)

        T_max = max(1, max_epochs - warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=cosine_eta_min)
        if scheds:
            sch = optim.lr_scheduler.SequentialLR(optimizer, schedulers=scheds + [cosine], milestones=milestones)
        else:
            sch = cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss",
            },
        }


__all__ = ["DinoDetrLitModule"]
