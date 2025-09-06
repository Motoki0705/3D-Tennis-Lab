from __future__ import annotations

import logging
from typing import Any, Tuple

import torch
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class TensorBoardHeatmapLogger(pl.Callback):
    """
    Logs:
      - the last RGB frame with GT/PRED peak points
      - standalone GT/PRED heatmaps (grayscale, 3ch)
    to TensorBoard every N steps.

    Expects batch to be (x, y) or dict with keys, where:
      - x: (B, C, T, H, W) or (B, 3*T, H, W)
      - y: (B, 1, H', W')
    """

    def __init__(
        self,
        every_n_steps: int = 200,
        num_samples: int = 2,
        mode: str = "val",  # "train" or "val"
        # new options for point rendering
        marker_radius: int = 4,
        pred_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),  # red
        gt_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),  # green
    ) -> None:
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self.num_samples = int(num_samples)
        self.mode = str(mode)
        self.marker_radius = int(marker_radius)
        self.pred_color = tuple(float(c) for c in pred_color)
        self.gt_color = tuple(float(c) for c in gt_color)

    # ------------------------------ helpers ------------------------------
    @staticmethod
    def _select_io(batch: Any):
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            if "inputs" in batch and "targets" in batch:
                return batch["inputs"], batch["targets"]
            if "x" in batch and "y" in batch:
                return batch["x"], batch["y"]
        raise ValueError("Unsupported batch format. Expected (x,y) or dict with inputs/targets.")

    @staticmethod
    def _prep_x_for_model(pl_module: pl.LightningModule, x: torch.Tensor) -> torch.Tensor:
        if hasattr(pl_module, "_prepare_x"):
            return pl_module._prepare_x(x)  # type: ignore[attr-defined]
        return x

    @staticmethod
    def _to_image(x3t: torch.Tensor, T: int) -> torch.Tensor:
        # x3t: (3*T, H, W) -> take last RGB frame: (3, H, W)
        assert x3t.dim() == 3 and x3t.size(0) % 3 == 0
        C3 = x3t.size(0)
        t = C3 // 3
        t_idx = max(0, min(T - 1, t - 1))
        frame = x3t[3 * t_idx : 3 * (t_idx + 1), :, :]
        return frame

    @staticmethod
    def _denorm(
        img_chw: torch.Tensor, mean: list[float] | tuple[float, ...], std: list[float] | tuple[float, ...]
    ) -> torch.Tensor:
        m = torch.tensor(mean, dtype=img_chw.dtype, device=img_chw.device).view(3, 1, 1)
        s = torch.tensor(std, dtype=img_chw.dtype, device=img_chw.device).view(3, 1, 1)
        x = img_chw * s + m
        return x.clamp(0.0, 1.0)

    @staticmethod
    def _norm_hm(hm: torch.Tensor) -> torch.Tensor:
        hm = hm.detach().float()
        vmin = torch.minimum(hm.min(), torch.tensor(0.0, device=hm.device))
        vmax = torch.maximum(hm.max(), torch.tensor(1e-6, device=hm.device))
        hm = (hm - vmin) / (vmax - vmin)
        return hm

    @staticmethod
    def _to_grayscale_3ch(hm: torch.Tensor) -> torch.Tensor:
        # hm: (H, W) in [0,1] -> (3,H,W) grayscale
        return hm.expand(3, *hm.shape[-2:])

    @staticmethod
    def _peak_xy(hm: torch.Tensor) -> Tuple[int, int]:
        """
        hm: (H, W) -> returns (x, y) int pixel coords in heatmap resolution.
        """
        H, W = hm.shape[-2], hm.shape[-1]
        idx = torch.argmax(hm.view(-1)).item()
        y, x = divmod(idx, W)
        return int(x), int(y)

    @staticmethod
    def _scale_xy(x: int, y: int, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> Tuple[int, int]:
        """
        Scale (x,y) from src (H,W) to dst (H,W). Nearest-pixel mapping.
        """
        srcH, srcW = src_hw
        dstH, dstW = dst_hw
        # avoid zero division
        sx = (dstW - 1) / max(1, srcW - 1)
        sy = (dstH - 1) / max(1, srcH - 1)
        return int(round(x * sx)), int(round(y * sy))

    @staticmethod
    def _draw_filled_circle(
        img: torch.Tensor, x: int, y: int, color: Tuple[float, float, float], radius: int
    ) -> torch.Tensor:
        """
        Draw a filled circle on img (3,H,W) in-place-like (returns a new tensor sharing storage).
        color in [0,1]; radius in pixels.
        """
        C, H, W = img.shape
        if radius <= 0:
            radius = 1
        x0 = max(0, x - radius)
        x1 = min(W - 1, x + radius)
        y0 = max(0, y - radius)
        y1 = min(H - 1, y + radius)
        if x0 > x1 or y0 > y1:
            return img

        yy = torch.arange(y0, y1 + 1, device=img.device).view(-1, 1)
        xx = torch.arange(x0, x1 + 1, device=img.device).view(1, -1)
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2  # (h, w) boolean
        # blend: simple overwrite toward color
        for c in range(3):
            region = img[c, y0 : y1 + 1, x0 : x1 + 1]
            region = torch.where(mask, torch.tensor(color[c], device=img.device, dtype=img.dtype), region)
            img[c, y0 : y1 + 1, x0 : x1 + 1] = region
        return img

    @torch.no_grad()
    def _log_batch(self, pl_module: pl.LightningModule, batch: Any, global_step: int, tag_prefix: str):
        logger_tb = getattr(pl_module.logger, "experiment", None)
        if logger_tb is None:
            logger.warning("TensorBoard logger is not available. Skipping image logging.")
            return

        x, y_true = self._select_io(batch)
        x_in = self._prep_x_for_model(pl_module, x)
        # Forward
        y_out = pl_module(x_in)
        scale = pl_module.cfg.model.out_scales[0]
        y_pred = y_out[scale]

        # Reduce to single-channel heatmap
        if y_pred.dim() == 4:
            y_pred_viz = y_pred[:, 0]
        else:
            y_pred_viz = y_pred
        y_true_viz = y_true[:, 0]

        B = x.size(0)
        num = min(self.num_samples, B)

        # inverse-norm params
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        try:
            if hasattr(pl_module.cfg, "data") and hasattr(pl_module.cfg.data, "normalize"):
                mean = list(getattr(pl_module.cfg.data.normalize, "mean", mean))
                std = list(getattr(pl_module.cfg.data.normalize, "std", std))
        except Exception:
            pass

        for i in range(num):
            # ---- choose last frame
            if x.dim() == 5:
                _, C, T, H, W = x.shape
                x3t = x[i].reshape(C * T, H, W)
                frame = self._to_image(x3t, T)
            else:
                _, C3T, H, W = x.shape
                T = C3T // 3
                frame = self._to_image(x[i], T)

            frame_denorm = self._denorm(frame, mean, std)  # (3,Hf,Wf)
            Hf, Wf = frame_denorm.shape[-2], frame_denorm.shape[-1]

            # ---- normalize heatmaps to [0,1]
            hm_pred = self._norm_hm(y_pred_viz[i])  # (Hp, Wp)
            hm_true = self._norm_hm(y_true_viz[i])  # (Hg, Wg)

            # ---- find peaks in heatmap space
            xp, yp = self._peak_xy(hm_pred)  # in pred map coords
            xg, yg = self._peak_xy(hm_true)  # in gt map coords

            # ---- scale to frame coords
            Hp, Wp = hm_pred.shape[-2], hm_pred.shape[-1]
            Hg, Wg = hm_true.shape[-2], hm_true.shape[-1]
            xp_f, yp_f = self._scale_xy(xp, yp, (Hp, Wp), (Hf, Wf))
            xg_f, yg_f = self._scale_xy(xg, yg, (Hg, Wg), (Hf, Wf))

            # ---- draw points on the frame
            frame_pts = frame_denorm.clone()
            frame_pts = self._draw_filled_circle(frame_pts, xp_f, yp_f, self.pred_color, self.marker_radius)
            frame_pts = self._draw_filled_circle(frame_pts, xg_f, yg_f, self.gt_color, self.marker_radius)

            # ---- log images
            logger_tb.add_image(f"{tag_prefix}/frame_points/{i}", frame_pts.clamp(0, 1), global_step)

            # standalone heatmaps (grayscale 3ch), still helpful for debugging
            logger_tb.add_image(f"{tag_prefix}/heatmap_pred/{i}", self._to_grayscale_3ch(hm_pred), global_step)
            logger_tb.add_image(f"{tag_prefix}/heatmap_gt/{i}", self._to_grayscale_3ch(hm_true), global_step)

    # ------------------------------ hooks ------------------------------
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        if self.mode != "train":
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
        self._log_batch(pl_module, batch, trainer.global_step, tag_prefix="val")


__all__ = ["TensorBoardHeatmapLogger"]
