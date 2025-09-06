from __future__ import annotations

import logging
from typing import Any

import torch
import pytorch_lightning as pl


logger = logging.getLogger(__name__)


class TensorBoardHeatmapLogger(pl.Callback):
    """
    Logs GT and prediction heatmaps to TensorBoard every N steps.

    Expects batch to be (x, y) or dict with keys, where:
      - x: (B, C, T, H, W) or (B, 3*T, H, W)
      - y: (B, 1, H', W') predicted heatmap shape matches model output (scale 0)
    """

    def __init__(
        self,
        every_n_steps: int = 200,
        num_samples: int = 2,
        mode: str = "val",  # "train" or "val"
        log_overlay: bool = True,
    ) -> None:
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self.num_samples = int(num_samples)
        self.mode = str(mode)
        self.log_overlay = bool(log_overlay)

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
        # Reuse module's helper if present
        if hasattr(pl_module, "_prepare_x"):
            return pl_module._prepare_x(x)  # type: ignore[attr-defined]
        # Fallback: accept (B, 3*T, H, W)
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
        # img_chw: (3,H,W) normalized; returns [0,1] clamped
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

    @torch.no_grad()
    def _log_batch(self, pl_module: pl.LightningModule, batch: Any, global_step: int, tag_prefix: str):
        logger_tb = getattr(pl_module.logger, "experiment", None)
        if logger_tb is None:
            logger.warning("TensorBoard logger is not available. Skipping image logging.")
            return

        x, y_true = self._select_io(batch)
        # Prepare x for model input
        x_in = self._prep_x_for_model(pl_module, x)
        # Forward
        y_out = pl_module(x_in)
        scale = pl_module.cfg.model.out_scales[0]
        y_pred = y_out[scale]

        # Shapes: y_true (B, 1, H, W), y_pred (B, C, H, W) or (B, 1, H, W)
        # Reduce to single-channel heatmap
        if y_pred.dim() == 4 and y_pred.size(1) > 1:
            y_pred_viz = y_pred[:, 0]
        elif y_pred.dim() == 4:
            y_pred_viz = y_pred[:, 0]
        else:
            y_pred_viz = y_pred

        y_true_viz = y_true[:, 0]

        # Select samples
        B = x.size(0)
        num = min(self.num_samples, B)
        # decide mean/std for inverse normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # allow override from cfg if available
        try:
            if hasattr(pl_module.cfg, "data") and hasattr(pl_module.cfg.data, "normalize"):
                mean = list(getattr(pl_module.cfg.data.normalize, "mean", mean))
                std = list(getattr(pl_module.cfg.data.normalize, "std", std))
        except Exception:
            pass

        for i in range(num):
            # Choose last frame from input
            if x.dim() == 5:
                Bx, C, T, H, W = x.shape
                x3t = x[i].reshape(C * T, H, W)
                frame = self._to_image(x3t, T)
            else:
                # (B, 3*T, H, W)
                _, C3T, H, W = x.shape
                T = C3T // 3
                frame = self._to_image(x[i], T)
            # inverse normalize original frame
            frame_denorm = self._denorm(frame, mean, std)

            hm_pred = self._norm_hm(y_pred_viz[i])
            hm_true = self._norm_hm(y_true_viz[i])

            # Resize heatmap to frame size if needed (for overlays)
            if hm_pred.shape[-2:] != frame_denorm.shape[-2:]:
                hm_pred_res = torch.nn.functional.interpolate(
                    hm_pred.unsqueeze(0).unsqueeze(0),
                    size=frame_denorm.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                hm_true_res = torch.nn.functional.interpolate(
                    hm_true.unsqueeze(0).unsqueeze(0),
                    size=frame_denorm.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            else:
                hm_pred_res, hm_true_res = hm_pred, hm_true

            # Overlays on inverse-normalized original image
            if self.log_overlay:
                # simple colorization: pred in red, gt in green
                pred_rgb = torch.stack(
                    [hm_pred_res, torch.zeros_like(hm_pred_res), torch.zeros_like(hm_pred_res)], dim=0
                )
                gt_rgb = torch.stack([torch.zeros_like(hm_true_res), hm_true_res, torch.zeros_like(hm_true_res)], dim=0)
                overlay_pred = (frame_denorm + pred_rgb) / 2.0
                overlay_gt = (frame_denorm + gt_rgb) / 2.0
                logger_tb.add_image(f"{tag_prefix}/image/{i}", frame_denorm, global_step)
                logger_tb.add_image(f"{tag_prefix}/overlay_pred/{i}", overlay_pred.clamp(0, 1), global_step)
                logger_tb.add_image(f"{tag_prefix}/overlay_gt/{i}", overlay_gt.clamp(0, 1), global_step)

            # Standalone heatmaps (grayscale 3ch)
            logger_tb.add_image(f"{tag_prefix}/heatmap_pred/{i}", self._to_grayscale_3ch(hm_pred), global_step)
            logger_tb.add_image(f"{tag_prefix}/heatmap_gt/{i}", self._to_grayscale_3ch(hm_true), global_step)

    # ------------------------------ hooks ------------------------------
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        if self.mode != "train":
            return
        if trainer.global_step % self.every_n_steps != 0:
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
        if trainer.global_step % self.every_n_steps != 0:
            return
        self._log_batch(pl_module, batch, trainer.global_step, tag_prefix="val")


__all__ = ["TensorBoardHeatmapLogger"]
