from __future__ import annotations

import logging
from typing import Any, List

import torch
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class TensorBoardDetectionImageLogger(pl.Callback):
    """
    Simple TensorBoard image logger for object detection.

    - Logs a few images with GT and predicted boxes every N steps.
    - Uses torchvision.utils.draw_bounding_boxes if available; otherwise skips drawing.
    - Expects batch to be (images, targets) where:
        images: List[Tensor(C,H,W)] or Tensor(B,3,H,W)
        targets: List[Dict] with 'boxes' (Nx4), 'labels' (N)
    """

    def __init__(self, every_n_steps: int = 200, num_samples: int = 2, mode: str = "val") -> None:
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self.num_samples = int(num_samples)
        self.mode = str(mode)

        try:
            from torchvision.utils import draw_bounding_boxes  # noqa: F401

            self._has_tv_draw = True
        except Exception:
            self._has_tv_draw = False

    @staticmethod
    def _select_io(batch: Any):
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        if isinstance(batch, dict) and "images" in batch and "targets" in batch:
            return batch["images"], batch["targets"]
        raise ValueError("Unsupported batch format for detection logger.")

    @staticmethod
    def _as_list_images(x: Any) -> List[torch.Tensor]:
        if isinstance(x, list):
            return x
        if torch.is_tensor(x):
            if x.dim() == 4:
                return [x[i] for i in range(x.size(0))]
            if x.dim() == 3:
                return [x]
        raise ValueError("Unexpected images type/shape for logger")

    @torch.no_grad()
    def _log_batch(self, pl_module: pl.LightningModule, batch: Any, global_step: int, tag_prefix: str):
        if not self._has_tv_draw:
            logger.debug("torchvision draw_bounding_boxes not available; skipping image logging.")
            return
        logger_tb = getattr(pl_module.logger, "experiment", None)
        if logger_tb is None:
            return

        try:
            from torchvision.utils import draw_bounding_boxes
        except Exception:
            return

        images, targets = self._select_io(batch)
        images_l = self._as_list_images(images)
        B = len(images_l)
        n = min(self.num_samples, B)

        # Forward for predictions
        pl_module.eval()
        preds = pl_module(images_l)  # type: ignore[arg-type]
        pl_module.train()

        for i in range(n):
            img = (images_l[i].detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8)
            if img.dim() == 3 and img.size(0) == 3:
                pass
            elif img.dim() == 3 and img.size(2) == 3:
                img = img.permute(2, 0, 1)
            else:
                continue

            gt_boxes = targets[i].get("boxes") if isinstance(targets, list) else None
            gt_boxes = gt_boxes.detach().cpu() if torch.is_tensor(gt_boxes) else None

            pred_i = preds[i] if isinstance(preds, list) else preds
            pred_boxes = pred_i.get("boxes") if isinstance(pred_i, dict) else None
            pred_scores = pred_i.get("scores") if isinstance(pred_i, dict) else None
            pred_boxes = pred_boxes.detach().cpu() if torch.is_tensor(pred_boxes) else None
            scores_str = None
            if torch.is_tensor(pred_scores):
                topk = min(5, pred_scores.numel())
                scores_str = ", ".join([f"{float(s):.2f}" for s in pred_scores[:topk].tolist()])

            img_draw = img
            if gt_boxes is not None and gt_boxes.numel() > 0:
                img_draw = draw_bounding_boxes(img_draw, gt_boxes, colors="green", width=2)
            if pred_boxes is not None and pred_boxes.numel() > 0:
                img_draw = draw_bounding_boxes(img_draw, pred_boxes, colors="red", width=2)

            logger_tb.add_image(f"{tag_prefix}/detections/{i}", img_draw, global_step)
            if scores_str:
                logger_tb.add_text(f"{tag_prefix}/scores/{i}", scores_str, global_step)

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        if self.mode != "train":
            return
        if self.every_n_steps <= 0 or (trainer.global_step % self.every_n_steps) != 0:
            return
        self._log_batch(pl_module, batch, trainer.global_step, tag_prefix="train")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.mode != "val":
            return
        if self.every_n_steps <= 0 or (trainer.global_step % self.every_n_steps) != 0:
            return
        self._log_batch(pl_module, batch, trainer.global_step, tag_prefix="val")


__all__ = ["TensorBoardDetectionImageLogger"]
