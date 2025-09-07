# pl_rtdetr_module.py
from __future__ import annotations
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from ..model_adapters.rtdetr_hf_adapter import HFRTDetrWrapper


def _sum_losses(losses) -> torch.Tensor:
    if isinstance(losses, torch.Tensor):
        return losses
    if isinstance(losses, dict):
        vals = [v for k, v in losses.items() if isinstance(v, torch.Tensor) and ("loss" in k or k.startswith("l_"))]
        if not vals:
            vals = [v for v in losses.values() if isinstance(v, torch.Tensor)]
        if not vals:
            raise ValueError("Loss dict is empty or has no tensor values")
        return torch.stack([v.float() for v in vals]).sum()
    if isinstance(losses, (list, tuple)):
        if not losses:
            raise ValueError("Empty loss list")
        return torch.stack([l.float() for l in losses]).sum()
    raise TypeError("Unsupported loss container type")


def _xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    x, y, w, h = xywh.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def _cxcywh_norm_to_xyxy_abs(cxcywh: torch.Tensor, size_hw: torch.Tensor) -> torch.Tensor:
    H, W = size_hw.to(cxcywh.device, cxcywh.dtype)
    cx, cy, w, h = cxcywh.unbind(-1)
    x1 = (cx - 0.5 * w) * W
    y1 = (cy - 0.5 * h) * H
    x2 = (cx + 0.5 * w) * W
    y2 = (cy + 0.5 * h) * H
    return torch.stack([x1, y1, x2, y2], dim=-1)


class RTDetrLightning(pl.LightningModule):
    def __init__(self, cfg: Any):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        num_classes = int(getattr(self.cfg.model, "num_classes", 1))
        self.detector: nn.Module = HFRTDetrWrapper(num_classes=num_classes, cfg=self.cfg)

        # Optim cfg
        tr = getattr(self.cfg, "training", {})
        self.lr = float(getattr(tr, "lr", 2e-4))
        self.weight_decay = float(getattr(tr, "weight_decay", 1e-4))
        betas = getattr(tr, "betas", (0.9, 0.999))
        self.betas = (float(betas[0]), float(betas[1])) if isinstance(betas, (list, tuple)) else (0.9, 0.999)
        self.max_epochs = int(getattr(tr, "max_epochs", 50))
        self.warmup_epochs = int(getattr(tr, "warmup_epochs", 1))
        self.eta_min = float(getattr(tr, "eta_min", 1e-6))
        self.log_each_loss = bool(getattr(tr, "log_each_loss", True))

        # Metrics
        self.enable_map = bool(getattr(getattr(self.cfg, "validation", {}), "enable_map", True))
        if self.enable_map:
            self.map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

        # Data format flags from adapter
        data_cfg = getattr(self.cfg, "data", {})
        self.boxes_format = str(getattr(data_cfg, "boxes_format", "xyxy")).lower()
        self.labels_are_one_based = bool(getattr(data_cfg, "labels_are_one_based", False))

    # ---------------- Training ----------------
    def training_step(self, batch: Tuple[List[torch.Tensor], List[dict]], batch_idx: int) -> torch.Tensor:
        images, targets = batch
        losses = self.detector(images, targets)
        loss = _sum_losses(losses)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        if self.log_each_loss:
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=False, batch_size=len(images))
        return loss

    # --------------- Validation ---------------
    @torch.no_grad()
    def validation_step(self, batch: Tuple[List[torch.Tensor], List[dict]], batch_idx: int):
        images, targets = batch

        # 1) Loss using labels (training path)
        losses = self.detector(images, targets)
        val_loss = _sum_losses(losses)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(images))
        if self.log_each_loss:
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(images))

        # 2) mAP via inference + post-process
        if self.enable_map:
            proc = self.detector.processor
            dev = next(self.detector.parameters()).device
            do_rescale = not bool(getattr(self.detector, "images_are_0_1", True))
            enc = proc(images=images, return_tensors="pt", do_rescale=do_rescale)
            enc = {k: v.to(dev) for k, v in enc.items()}
            outputs = self.detector.model(**enc)

            target_sizes = torch.tensor([img.shape[-2:] for img in images], device=dev)
            preds = proc.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)

            preds_tm, targets_tm = [], []
            for i, tgt in enumerate(targets):
                # prepare GT (xyxy abs)
                boxes = tgt["boxes"]
                boxes = (
                    boxes.to(dev, dtype=torch.float32)
                    if isinstance(boxes, torch.Tensor)
                    else torch.as_tensor(boxes, device=dev, dtype=torch.float32)
                )
                if self.boxes_format in {"xywh", "coco"}:
                    boxes_xyxy = _xywh_to_xyxy(boxes)
                elif self.boxes_format == "xyxy":
                    boxes_xyxy = boxes
                elif self.boxes_format == "cxcywh_norm":
                    size = tgt.get("size", tgt.get("orig_size"))
                    if not isinstance(size, torch.Tensor):
                        size = torch.as_tensor(size, dtype=torch.float32, device=dev)
                    boxes_xyxy = _cxcywh_norm_to_xyxy_abs(boxes, size)
                else:
                    raise ValueError(f"Unsupported boxes_format: {self.boxes_format}")

                labels = tgt["labels"]
                labels = (
                    labels.to(dev, dtype=torch.long)
                    if isinstance(labels, torch.Tensor)
                    else torch.as_tensor(labels, device=dev, dtype=torch.long)
                )
                if self.labels_are_one_based:
                    labels = labels - 1

                targets_tm.append({"boxes": boxes_xyxy, "labels": labels})

                # predictions
                p = preds[i]
                preds_tm.append({
                    "boxes": p["boxes"].to(dev),
                    "scores": p["scores"].to(dev),
                    "labels": p["labels"].to(dev),
                })

            self.map_metric.update(preds_tm, targets_tm)

        return val_loss

    def on_validation_epoch_end(self):
        if self.enable_map:
            metrics = self.map_metric.compute()
            self.log_dict(
                {
                    "val/mAP": metrics.get("map", torch.tensor(0.0)),
                    "val/mAP50": metrics.get("map_50", torch.tensor(0.0)),
                    "val/mAP75": metrics.get("map_75", torch.tensor(0.0)),
                    "val/mAR_100": metrics.get("mar_100", torch.tensor(0.0)),
                },
                prog_bar=True,
                on_epoch=True,
            )
            self.map_metric.reset()

    # --------------- Optimizers ---------------
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=self.betas)

        scheds, milestones = [], []
        if self.warmup_epochs > 0:
            warmup = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_epochs
            )
            scheds.append(warmup)
            milestones.append(self.warmup_epochs)

        t_max = max(1, self.max_epochs - self.warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=self.eta_min)
        scheduler = (
            optim.lr_scheduler.SequentialLR(optimizer, schedulers=scheds + [cosine], milestones=milestones)
            if scheds
            else cosine
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val/loss"},
        }
