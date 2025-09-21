from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - Lightning not installed in env
    pl = None  # type: ignore


def _to_3ch(x: torch.Tensor) -> torch.Tensor:
    """Convert [H,W] or [1,H,W] to [3,H,W] grayscale (0..1)."""
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [1,H,W]
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    return x.clamp(0, 1)


def _upsample_2d(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Upsample a 2D map [H,W] to size using bilinear (returns [H',W'])."""
    x = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x.squeeze(0).squeeze(0)


@dataclass
class HeatmapLogger(pl.callbacks.Callback if pl else object):
    """
    Log ALL heatmaps (K) for both GT and Pred at a large size to TensorBoard.

    Modes:
      - per_channel  : each heatmap is logged as a separate large image
      - chunked_grid : heatmaps are grouped into chunks and logged as large grids

    Expected outputs in validation/test steps:
      outputs = {
        "images":          [N, C, H, W] (optional; required if log_input=True),
        "pred_heatmaps":   [N, K, H, W],
        "target_heatmaps": [N, K, H, W],
      }
    """

    mode: str = "per_channel"  # "per_channel" | "chunked_grid"
    max_samples: int = 2  # how many samples from the first batch to log
    upsample_to: int = 512  # final H=W size for single heatmap images
    grid_nrow: int = 4  # for chunked_grid mode
    chunk_size: int = 16  # number of K per grid in chunked_grid mode
    normalize_each: bool = True  # normalize each heatmap independently
    stage_prefix: str = "Val"  # "Val" / "Test"
    log_input: bool = True  # also log the input image
    image_tag: str = "Input"
    pred_tag: str = "Pred"
    target_tag: str = "GT"
    every_n_epochs: int = 1  # log every n epochs

    # internal buffers
    _ready: bool = True
    _buffer: Dict[str, torch.Tensor] | None = None

    def on_validation_epoch_start(self, trainer, pl_module):
        epoch = int(getattr(pl_module, "current_epoch", 0))
        if epoch >= 0 and (epoch % self.every_n_epochs) != 0:
            self._ready = False
            return
        self._ready = True
        self._buffer = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self._ready or batch_idx != 0:
            return
        if not isinstance(outputs, dict):
            return
        keys = ["pred_heatmaps", "target_heatmaps"]
        if not all(k in outputs for k in keys):
            return

        pred = outputs["pred_heatmaps"].detach()
        targ = outputs["target_heatmaps"].detach()
        images = outputs.get("images", None)
        if images is not None:
            images = images.detach()

        # Slice to max_samples
        n = min(pred.size(0), self.max_samples)
        pred = pred[:n]
        targ = targ[:n]
        if images is not None:
            images = images[:n]

        self._buffer = {
            "pred": pred.cpu(),
            "targ": targ.cpu(),
        }
        if images is not None:
            self._buffer["images"] = images.cpu()
        self._ready = False  # only first batch

    def _log_image(self, writer, tag: str, img: torch.Tensor, step: int):
        # img: [3,H,W], 0..1
        writer.add_image(tag, img, step, dataformats="CHW")

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_each:
            mn, mx = x.min(), x.max()
            if (mx - mn) > 1e-8:
                x = (x - mn) / (mx - mn)
            else:
                x = torch.zeros_like(x)
        else:
            x = x.clamp(0, 1)
        return x

    def _log_per_channel(self, writer, step: int):
        assert self._buffer is not None
        pred = self._buffer["pred"]  # [N,K,H,W]
        targ = self._buffer["targ"]  # [N,K,H,W]
        images = self._buffer.get("images")  # [N,C,H,W] or None
        N, K, H, W = pred.shape
        size = (self.upsample_to, self.upsample_to)

        # Optional: log input images large
        if self.log_input and images is not None:
            for i in range(N):
                img = images[i]
                if img.dim() == 3 and img.size(0) == 1:
                    img = img.repeat(3, 1, 1)
                # upscale input if needed
                if img.shape[-2:] != size:
                    img = F.interpolate(img.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
                self._log_image(writer, f"{self.stage_prefix}/{self.image_tag}/sample_{i:02d}", img, step)

        # Log every K as a separate big image
        for i in range(N):
            for k in range(K):
                hp = self._norm(pred[i, k])  # [H,W]
                ht = self._norm(targ[i, k])
                hp = _upsample_2d(hp, size)
                ht = _upsample_2d(ht, size)
                self._log_image(
                    writer, f"{self.stage_prefix}/{self.pred_tag}/sample_{i:02d}/kp_{k:03d}", _to_3ch(hp), step
                )
                self._log_image(
                    writer, f"{self.stage_prefix}/{self.target_tag}/sample_{i:02d}/kp_{k:03d}", _to_3ch(ht), step
                )

    def _log_chunked_grid(self, writer, step: int):
        assert self._buffer is not None
        pred = self._buffer["pred"]  # [N,K,H,W] or [B,T,C,H,W]
        targ = self._buffer["targ"]  # [N,K,H,W] or [B,T,C,H,W]
        images = self._buffer.get("images")  # [N,C,H,W] or [B,T,C,H,W]
        if pred is not None and pred.ndim == 5:
            b, t, c, h, w = pred.shape
            pred = pred.reshape(b * t, c, h, w)
        if targ is not None and targ.ndim == 5:
            b, t, c, h, w = targ.shape
            targ = targ.reshape(b * t, c, h, w)
        # ★ images も 5D→4D に正規化（最後のフレームで良ければ images = images[:, -1] でもOK）
        if images is not None and images.ndim == 5:
            b, t, c, h, w = images.shape
            images = images.reshape(b * t, c, h, w)
        N, K, H, W = pred.shape

        # Optionally log inputs (once)
        if self.log_input and images is not None:
            vis = images[: self.max_samples]  # -> [N,C,H,W]
            # 1chなら3chへ
            if vis.size(1) == 1:
                vis = vis.repeat(1, 3, 1, 1)
            writer.add_images(f"{self.stage_prefix}/{self.image_tag}", vis, step, dataformats="NCHW")

        # chunk K into groups and log big grids
        for i in range(N):
            start = 0
            chunk_idx = 0
            while start < K:
                end = min(start + self.chunk_size, K)
                # prepare tensors [M,1,H,W] -> upsample -> make_grid
                pred_chunk = []
                targ_chunk = []
                for k in range(start, end):
                    hp = self._norm(pred[i, k])
                    ht = self._norm(targ[i, k])
                    pred_chunk.append(_upsample_2d(hp, (self.upsample_to, self.upsample_to)).unsqueeze(0))
                    targ_chunk.append(_upsample_2d(ht, (self.upsample_to, self.upsample_to)).unsqueeze(0))

                pred_stack = torch.stack(pred_chunk, dim=0)  # [M,1,H',W']
                targ_stack = torch.stack(targ_chunk, dim=0)

                pred_grid = make_grid(pred_stack, nrow=self.grid_nrow, normalize=False)  # [?,Hg,Wg]（1chのことがある）
                targ_grid = make_grid(targ_stack, nrow=self.grid_nrow, normalize=False)

                # 安全に3chへ（_log_imageはCHW想定）
                self._log_image(
                    writer, f"{self.stage_prefix}/{self.pred_tag}/sample_{i:02d}/chunk_{chunk_idx:02d}", pred_grid, step
                )
                self._log_image(
                    writer,
                    f"{self.stage_prefix}/{self.target_tag}/sample_{i:02d}/chunk_{chunk_idx:02d}",
                    targ_grid,
                    step,
                )

                start = end
                chunk_idx += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._buffer is None:
            return
        logger = getattr(trainer, "logger", None)
        if logger is None or not hasattr(logger, "experiment"):
            return
        writer = logger.experiment  # TensorBoard SummaryWriter
        step = getattr(pl_module, "current_epoch", 0)

        if self.mode == "per_channel":
            self._log_per_channel(writer, step)
        elif self.mode == "chunked_grid":
            self._log_chunked_grid(writer, step)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        # clear
        self._buffer = None
