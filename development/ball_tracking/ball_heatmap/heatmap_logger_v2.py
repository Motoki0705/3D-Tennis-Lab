# development/ball_tracking/ball_heatmap/callbacks/heatmap_logger_v2.py
from __future__ import annotations
from typing import List, Optional
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
        self._buffer = outputs  # keep CPU tensors as prepared by validation_step

    def _coords_from_hmap_and_offset_img(self, hmap: torch.Tensor, offs: torch.Tensor, stride: int) -> torch.Tensor:
        # hmap: [B,1,Hs,Ws], offs: [B,2,Hs,Ws] -> [B,2] in image px
        b, _, _Hs, Ws = hmap.shape
        flat = hmap.view(b, -1)
        idx = flat.argmax(dim=-1)  # [B]
        ys = (idx // Ws).float()  # [B]
        xs = (idx % Ws).float()  # [B]
        dx = offs[:, 0].contiguous().view(b, -1).gather(1, idx.view(-1, 1)).squeeze(1)
        dy = offs[:, 1].contiguous().view(b, -1).gather(1, idx.view(-1, 1)).squeeze(1)
        x_img = (xs + dx) * float(stride)
        y_img = (ys + dy) * float(stride)
        return torch.stack([x_img, y_img], dim=-1)  # [B,2]

    def _draw_overlay(
        self,
        images: torch.Tensor,
        pred_xy_img: torch.Tensor,
        gt_xy_img: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        images: [B,3,H,W] in [0,1] or arbitrary (will normalize)
        pred_xy_img: [B,2] (x,y) in image px
        gt_xy_img:   [B,2] or None
        valid:       [B] bool/0-1 or None
        returns CHW grid tensor ready for add_image
        """
        B = images.shape[0]
        # 0..1 正規化してから描画（TensorBoardで見やすく）
        imgs = images.clone()
        if imgs.dtype != torch.float32:
            imgs = imgs.float()
        # ざっくり per-image minmax 正規化（画質にこだわらない）
        imgs = (imgs - imgs.amin(dim=(1, 2, 3), keepdim=True)) / (
            imgs.amax(dim=(1, 2, 3), keepdim=True) - imgs.amin(dim=(1, 2, 3), keepdim=True) + 1e-6
        )

        # torchvision.draw_keypoints は [N,K,2] を期待
        pred_kp = pred_xy_img.view(B, 1, 2)
        imgs = draw_keypoints(imgs, pred_kp, colors=(255, 0, 0), radius=3, width=2)  # 赤：pred

        if gt_xy_img is not None:
            if valid is not None:
                mask = (valid.view(-1) > 0).cpu()
            else:
                mask = torch.ones(B, dtype=torch.bool)
            if mask.any():
                gt_kp = gt_xy_img[mask].view(-1, 1, 2)
                imgs[mask] = draw_keypoints(imgs[mask], gt_kp, colors=(0, 255, 0), radius=3, width=2)  # 緑：GT

        grid = make_grid(imgs, nrow=min(B, self.num_samples))
        return grid

    def on_validation_epoch_end(self, trainer, pl_module):
        if not getattr(trainer, "logger", None) or self._buffer is None:
            return
        writer = trainer.logger.experiment

        # 入力画像
        images: torch.Tensor = self._buffer["images"][: self.num_samples]  # [B,3,H,W]
        writer.add_image("val/input", make_grid(images, normalize=True), global_step=pl_module.current_epoch)

        # マルチスケールの予測/GT ヒートマップの可視化（既存）
        if self.draw_multiscale and "pred_heatmaps" in self._buffer and "target_heatmaps" in self._buffer:
            pred_list: List[torch.Tensor] = self._buffer["pred_heatmaps"]
            tgt_list: List[torch.Tensor] = self._buffer["target_heatmaps"]
            for i, (ph, th) in enumerate(zip(pred_list, tgt_list)):
                writer.add_image(
                    f"val/pred_scale{i}",
                    make_grid(ph[: self.num_samples], normalize=True),
                    global_step=pl_module.current_epoch,
                )
                writer.add_image(
                    f"val/tgt_scale{i}",
                    make_grid(th[: self.num_samples], normalize=True),
                    global_step=pl_module.current_epoch,
                )

        # ★ ここから overlay の追加：pred(ヒートマップ+オフセット) vs GT
        have_offs = "pred_offsets" in self._buffer and len(self._buffer["pred_offsets"]) > 0
        # valid_mask is optional; if absent, treat all as valid
        have_gt = "gt_coords_img" in self._buffer
        if have_offs and have_gt and "pred_heatmaps" in self._buffer and len(self._buffer["pred_heatmaps"]) > 0:
            # 最終スケール（通常は最高解像度）を使用
            pred_hmap_hi: torch.Tensor = self._buffer["pred_heatmaps"][-1]
            pred_off_hi: torch.Tensor = self._buffer["pred_offsets"][-1]
            # stride 推定：モジュールが持つ値を優先。なければ 1。
            stride = 1
            if hasattr(pl_module, "config") and hasattr(pl_module.config.model, "deep_supervision_strides"):
                stride = int(pl_module.config.model.deep_supervision_strides[-1])
            elif hasattr(pl_module, "stride"):
                stride = int(pl_module.stride)

            # [B,2] 画像座標
            pred_xy_img = self._coords_from_hmap_and_offset_img(pred_hmap_hi, pred_off_hi, stride)
            gt_xy_img = self._buffer["gt_coords_img"]
            valid = self._buffer.get("valid_mask", None)
            if valid is not None:
                valid = valid.view(-1)

            # サンプル数をそろえる
            if valid is not None:
                B = min(images.shape[0], pred_xy_img.shape[0], gt_xy_img.shape[0], valid.shape[0])
                grid = self._draw_overlay(images[:B], pred_xy_img[:B], gt_xy_img[:B], valid[:B])
            else:
                B = min(images.shape[0], pred_xy_img.shape[0], gt_xy_img.shape[0])
                grid = self._draw_overlay(images[:B], pred_xy_img[:B], gt_xy_img[:B], None)

            writer.add_image("val/overlay_pred_gt", grid, global_step=pl_module.current_epoch)

        # clear buffer after logging
        self._buffer = None
