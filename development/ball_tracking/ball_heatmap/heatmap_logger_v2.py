from __future__ import annotations

from typing import List

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torchvision.utils import make_grid, draw_keypoints


class HeatmapLoggerV2(Callback):
    def __init__(self, num_samples: int = 8, draw_multiscale: bool = True):
        super().__init__()
        self.num_samples = num_samples
        self.draw_multiscale = draw_multiscale
        self._buffer = None

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx):
        if batch_idx > 0:
            return
        # outputs contains lists of tensors per-scale as returned by BallLitModule
        self._buffer = outputs

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if not trainer.logger or not self._buffer:
            return
        writer = trainer.logger.experiment
        images: torch.Tensor = self._buffer["images"][: self.num_samples]
        img_grid = make_grid(images, normalize=True)
        writer.add_image("val/input", img_grid, global_step=pl_module.current_epoch)

        pred_scales: List[torch.Tensor] = self._buffer["pred_heatmaps"]
        tgt_scales: List[torch.Tensor] = self._buffer["target_heatmaps"]
        pred_offs: List[torch.Tensor] = self._buffer.get("pred_offsets", [])
        gt_xy: torch.Tensor = self._buffer.get("gt_coords_img")

        for i, (pred, tgt) in enumerate(zip(pred_scales, tgt_scales)):
            pred = pred[: self.num_samples]
            tgt = tgt[: self.num_samples]
            # Each is [N,1,H,W]
            writer.add_image(f"val/pred_scale{i}", make_grid(pred, normalize=True), global_step=pl_module.current_epoch)
            writer.add_image(f"val/tgt_scale{i}", make_grid(tgt, normalize=True), global_step=pl_module.current_epoch)
        # Overlay predictions vs GT on inputs (highest res)
        try:
            if len(pred_scales) and isinstance(pred_offs, list) and len(pred_offs):
                hmap = pred_scales[-1]
                offs = pred_offs[-1]
                b = min(hmap.shape[0], self.num_samples)
                stride = int(pl_module.config.model.deep_supervision_strides[-1])
                # Compute predicted coords in image space
                flat = hmap[:b].view(b, -1)
                idx = flat.argmax(dim=-1)
                _, Ws = hmap.shape[-2:]
                ys = (idx // Ws).float()
                xs = (idx % Ws).float()
                dx = offs[:b, 0].contiguous().view(b, -1).gather(1, idx.view(-1, 1)).squeeze(1)
                dy = offs[:b, 1].contiguous().view(b, -1).gather(1, idx.view(-1, 1)).squeeze(1)
                x_img = (xs + dx) * float(stride)
                y_img = (ys + dy) * float(stride)
                pred_xy = torch.stack([x_img, y_img], dim=-1)  # [b,2]

                if gt_xy is not None:
                    gt = gt_xy[:b].clone()
                else:
                    gt = pred_xy.clone()

                # Prepare images in [0,255]
                imgs = (images[:b].detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8)
                # draw keypoints expects [N, C, H, W] and keypoints [N, K, 2]
                kps_pred = pred_xy.view(b, 1, 2)
                kps_gt = gt.view(b, 1, 2)
                over1 = draw_keypoints(imgs, kps_pred, colors=(255, 0, 0))
                over2 = draw_keypoints(over1, kps_gt, colors=(0, 255, 0))
                writer.add_image(
                    "val/overlay_pred_gt", make_grid(over2, normalize=False), global_step=pl_module.current_epoch
                )
        except Exception:
            # keep logger robust
            pass
        self._buffer = None
