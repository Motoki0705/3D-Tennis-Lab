from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

import torch

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - OmegaConf not installed
    DictConfig = ()  # type: ignore
    OmegaConf = None  # type: ignore

logger = logging.getLogger(__name__)

from development.core.lightning.base_lit_module import BaseLitModule


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


class DinoDetrLitModule(BaseLitModule):
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
        super().__init__(config=cfg, model=model, loss_fn=loss_fn, metric_fns=dict(metric_fns or {}))
        self.postprocessors = dict(postprocessors or {})

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
        images, targets = batch  # images: Tensor[N,3,H,W], targets: list[dict]

        # -------------------------
        # forward
        # -------------------------
        outputs = self.model(images)

        # -------------------------
        # compute loss
        # -------------------------
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

        # metrics (必要なら)
        self._log_metrics(outputs, targets, prefix="val", on_step=False)

        # -------------------------
        # PostProcess → 画像スケール xyxy に変換
        # -------------------------
        target_sizes = torch.tensor(
            [[img.shape[-2], img.shape[-1]] for img in images],
            dtype=torch.float32,
            device=images.device,
        )
        results = self.postprocessors["bbox"](outputs, target_sizes)  # list[dict]

        # -------------------------
        # 可視化用: Top-K 固定長化
        # -------------------------
        Kp, Kg = 50, 50  # 予測/GT の上限数
        pred_boxes, pred_labels, pred_scores = [], [], []
        for det in results:
            b, l, s = det["boxes"], det["labels"], det["scores"]
            # Top-K
            idx = torch.argsort(s, descending=True)[:Kp]
            b, l, s = b[idx], l[idx], s[idx]
            # パディング
            pad = Kp - b.size(0)
            if pad > 0:
                b = torch.cat([b, b.new_zeros(pad, 4)], dim=0)
                l = torch.cat([l, l.new_full((pad,), -1)], dim=0)
                s = torch.cat([s, s.new_zeros(pad)], dim=0)
            pred_boxes.append(b)
            pred_labels.append(l)
            pred_scores.append(s)
        pred_boxes = torch.stack(pred_boxes, 0)  # [N,Kp,4]
        pred_labels = torch.stack(pred_labels, 0)  # [N,Kp]
        pred_scores = torch.stack(pred_scores, 0)  # [N,Kp]

        gt_boxes = []
        for t in targets:
            b = t["boxes"]
            if b.size(0) > Kg:
                b = b[:Kg]
            elif b.size(0) < Kg:
                pad = Kg - b.size(0)
                b = torch.cat([b, b.new_zeros(pad, 4)], dim=0)
            gt_boxes.append(b)
        gt_boxes = torch.stack(gt_boxes, 0)  # [N,Kg,4]

        # -------------------------
        # return dict for renderer
        # -------------------------
        return {
            "images": images.clamp(0, 1),  # [N,3,H,W]
            "pred_bboxes": pred_boxes,  # [N,Kp,4] xyxy
            "class_labels": pred_labels,  # [N,Kp]
            "pred_scores": pred_scores,  # [N,Kp]
            "target_bboxes": gt_boxes,  # [N,Kg,4]
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
