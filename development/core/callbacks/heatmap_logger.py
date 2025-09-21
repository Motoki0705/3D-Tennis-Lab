from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
except Exception:
    pl = None  # type: ignore


def _to_3ch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    return x.clamp(0, 1)


def _upsample_2d(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    x = x.unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x.squeeze(0).squeeze(0)


@dataclass
class HeatmapLogger(pl.callbacks.Callback if pl else object):
    """
    Logs Input / Pred / GT in **temporal order**.
    Expects 5D tensors from the LightningModule outputs:
      images: [B, T, C, H, W] (optional; required if log_input=True)
      pred_heatmaps:   [B, T, 1, H, W] or [B, T, K, H, W]
      target_heatmaps: [B, T, 1, H, W] or [B, T, K, H, W]
    """

    mode: str = "per_channel"  # kept for compatibility, currently logs per-frame images
    num_samples: int = 2  # number of sequences sampled per epoch
    upsample_to: int = 512
    normalize_each: bool = True
    stage_prefix: str = "Val"
    log_input: bool = True
    image_tag: str = "Input"
    pred_tag: str = "Pred"
    target_tag: str = "GT"
    every_n_epochs: int = 1
    rng_seed: int = 1234  # base seed; epochでオフセット

    # internal
    _ready: bool = True
    _buffer: Optional[Dict[str, torch.Tensor]] = None
    _epoch: int = 0

    # ---------------- Hooks ----------------
    def on_validation_epoch_start(self, trainer, pl_module):
        self._epoch = int(getattr(pl_module, "current_epoch", 0))
        # allow sanity check (epoch=-1); otherwise honor every_n_epochs
        if self._epoch >= 0 and (self._epoch % self.every_n_epochs) != 0:
            self._ready = False
            return
        self._ready = True
        self._buffer = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # 1バッチ目だけ拾う（コスト・重複回避）
        if not self._ready or batch_idx != 0:
            return
        if not isinstance(outputs, dict):
            return
        if ("pred_heatmaps" not in outputs) or ("target_heatmaps" not in outputs):
            return

        pred5 = outputs["pred_heatmaps"].detach()  # [B,T,K,H,W] or [B,T,1,H,W]
        targ5 = outputs["target_heatmaps"].detach()
        imgs5 = outputs.get("images", None)
        if imgs5 is not None:
            imgs5 = imgs5.detach()  # [B,T,C,H,W]

        # 形状チェック（Tは同じ想定）
        B = int(pred5.shape[0])
        # torch.randperm は CPU Generator しか受け付けない
        g = torch.Generator(device="cpu")
        g.manual_seed(self.rng_seed + max(self._epoch, 0))
        if self.num_samples >= B:
            sel = torch.arange(B)
        else:
            sel = torch.randperm(B, generator=g)[: self.num_samples]
        sel = sel.tolist()

        # バッファに 5D のまま保存（後で時間順に1枚ずつ出力）
        buf = {
            "pred": pred5[sel].cpu(),  # [S,T,K,H,W]
            "targ": targ5[sel].cpu(),
        }
        if imgs5 is not None:
            buf["images"] = imgs5[sel].cpu()  # [S,T,C,H,W]
        self._buffer = buf
        self._ready = False

    # ---------------- Utils ----------------
    def _log_image(self, writer, tag: str, img: torch.Tensor, step: int):
        # img: [C,H,W] or [H,W] -> 3ch
        img = _to_3ch(img)
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

    # ---------------- Logging core ----------------
    def _log_temporal_sequences(self, writer, step: int):
        """
        時系列順に、各サンプル S について t=0..T-1 を順番に 1枚ずつ保存する。
        - Input   (任意)
        - Pred    （K>1のときは各Kを個別保存）
        - GT
        """
        assert self._buffer is not None
        size = (self.upsample_to, self.upsample_to)

        pred = self._buffer["pred"]  # [S,T,K,H,W]
        targ = self._buffer["targ"]  # [S,T,K,H,W]
        imgs = self._buffer.get("images")  # [S,T,C,H,W] or None

        S, T = int(pred.shape[0]), int(pred.shape[1])
        K = int(pred.shape[2])

        for s in range(S):
            for t in range(T):
                # --- Input ---
                if self.log_input and imgs is not None:
                    img = imgs[s, t]  # [C,H,W]
                    if img.dim() == 3 and img.shape[-2:] != size:
                        img = F.interpolate(img.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(
                            0
                        )
                    self._log_image(writer, f"{self.stage_prefix}/{self.image_tag}/s{s:02d}/t{t:04d}", img, step)

                # --- Pred / GT（K枚あれば全部。1枚ならそのまま） ---
                for k in range(K):
                    hp = self._norm(pred[s, t, k])  # [H,W]
                    ht = self._norm(targ[s, t, k])  # [H,W]
                    hp = _upsample_2d(hp, size)
                    ht = _upsample_2d(ht, size)

                    self._log_image(writer, f"{self.stage_prefix}/{self.pred_tag}/s{s:02d}/t{t:04d}/k{k:03d}", hp, step)
                    self._log_image(
                        writer, f"{self.stage_prefix}/{self.target_tag}/s{s:02d}/t{t:04d}/k{k:03d}", ht, step
                    )

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._buffer is None:
            return
        logger = getattr(trainer, "logger", None)
        if logger is None or not hasattr(logger, "experiment"):
            return
        writer = logger.experiment
        step = getattr(pl_module, "current_epoch", 0)

        # 時系列順ログ
        self._log_temporal_sequences(writer, step)

        # clear
        self._buffer = None
