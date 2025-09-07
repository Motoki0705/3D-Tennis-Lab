from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from ..model.lite_unet_context import build_preset_a, LiteUNetContext
from ....utils.loss import loss_registry


@dataclass
class ModelConfig:
    num_keypoints: int = 15
    variant: str = "small"  # nano | small | base
    out_stride: int = 4  # 4 or 2
    use_offset_head: bool = False
    deep_supervision: bool = False


class LiteUNetContextLitModule(pl.LightningModule):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg

        # Model config
        mcfg = getattr(cfg, "model", {})
        self.model_cfg = ModelConfig(
            num_keypoints=int(getattr(mcfg, "num_keypoints", 15)),
            variant=str(getattr(mcfg, "variant", "small")),
            out_stride=int(getattr(mcfg, "out_stride", 4)),
            use_offset_head=bool(getattr(mcfg, "use_offset_head", False)),
            deep_supervision=bool(getattr(mcfg, "deep_supervision", False)),
        )

        # Build model
        self.model: LiteUNetContext = build_preset_a(
            num_keypoints=self.model_cfg.num_keypoints,
            variant=self.model_cfg.variant,
            out_stride=self.model_cfg.out_stride,
            use_offset_head=self.model_cfg.use_offset_head,
            deep_supervision=self.model_cfg.deep_supervision,
        )

        # Loss (heatmap)
        lcfg = getattr(getattr(cfg, "training", {}), "loss", {})
        loss_name = str(getattr(lcfg, "name", "mse")).lower()
        loss_params = getattr(lcfg, "params", {}) if hasattr(lcfg, "params") else {}
        # default to mse to avoid logits/probability ambiguity
        if loss_name not in ("mse", "focal", "bce", "kldiv"):
            loss_name = "mse"
        self.loss_fn = loss_registry.get(loss_name, **(loss_params or {}))
        self.ds_weight = float(getattr(getattr(cfg, "training", {}), "deep_supervision_weight", 0.4))

        # Loss (offset)
        ocfg = getattr(getattr(cfg, "training", {}), "offset_loss", {})
        o_name = str(getattr(ocfg, "name", "l1")).lower()
        o_params = getattr(ocfg, "params", {}) if hasattr(ocfg, "params") else {}
        if o_name in ("mse", "bce", "focal", "kldiv"):
            self.offset_loss_fn = loss_registry.get(o_name, **(o_params or {}))
        elif o_name in ("l1", "mae"):
            self.offset_loss_fn = nn.L1Loss(reduction="none")
        else:
            # fallback to L2/MSE
            self.offset_loss_fn = nn.MSELoss(reduction="none")
        self.offset_weight = float(getattr(ocfg, "weight", 1.0))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        return self.model(x)

    def _select_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Returns x, y_main, y_aux(optional). Supports dict or (x,y)."""
        if isinstance(batch, dict):
            x = batch.get("inputs")
            y_main = batch.get("targets")
            aux = None
            aux_dict = batch.get("aux")
            if isinstance(aux_dict, dict) and "os8" in aux_dict:
                aux = aux_dict["os8"]
            if x is None or y_main is None:
                raise ValueError("Batch dict must have 'inputs' and 'targets'.")
            return x, y_main, aux
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1], None
        raise ValueError("Unsupported batch format for lite_cnn.")

    def _compute_loss(
        self,
        pred: Dict[str, torch.Tensor | Dict[str, torch.Tensor]],
        y: torch.Tensor,
        y_aux: Optional[torch.Tensor],
        batch: Optional[Dict[str, torch.Tensor]] = None,
    ):
        y_hat_main = pred.get("heatmap")
        if y_hat_main is None:
            raise ValueError("Model output missing 'heatmap'")
        loss_main = self.loss_fn(y_hat_main, y)

        loss = loss_main
        logs = {"loss_main": loss_main.detach()}
        if self.model_cfg.deep_supervision and y_aux is not None:
            aux_dict = pred.get("aux") if isinstance(pred, dict) else None
            y_hat_aux = aux_dict.get("os8") if isinstance(aux_dict, dict) else None
            if y_hat_aux is not None:
                loss_aux = self.loss_fn(y_hat_aux, y_aux)
                loss = loss + self.ds_weight * loss_aux
                logs["loss_aux"] = loss_aux.detach()

        # Offset loss
        if self.model_cfg.use_offset_head and isinstance(pred, dict) and "offset" in pred and isinstance(batch, dict):
            y_hat_off = pred["offset"]  # [B, 2K, H, W]
            y_off = batch.get("offsets")  # [B, 2K, H, W]
            m_off = batch.get("offset_mask")  # [B, K, H, W] or None
            if y_off is not None:
                if m_off is not None:
                    # Expand mask to 2K channels: duplicate per (dx,dy)
                    m2 = torch.repeat_interleave(m_off, repeats=2, dim=1).to(y_hat_off.dtype)
                else:
                    # If no mask provided, derive from target heatmap (peaky regions)
                    m2 = (y > 0.5).to(y_hat_off.dtype)
                    m2 = torch.repeat_interleave(m2, repeats=2, dim=1)

                diff = self.offset_loss_fn(y_hat_off, y_off)  # elementwise, shape [B,2K,H,W]
                if diff.dim() == 0:
                    # just in case reduction was applied by mistake
                    off_loss = diff
                else:
                    masked = diff * m2
                    denom = m2.sum().clamp_min(1.0)
                    off_loss = masked.sum() / denom
                loss = loss + self.offset_weight * off_loss
                logs["loss_offset"] = off_loss.detach()
        return loss, logs

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y, y_aux = self._select_batch(batch)
        out = self(x)
        loss, logs = self._compute_loss(out, y, y_aux, batch if isinstance(batch, dict) else None)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in logs.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, y_aux = self._select_batch(batch)
        out = self(x)
        loss, logs = self._compute_loss(out, y, y_aux, batch if isinstance(batch, dict) else None)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        for k, v in logs.items():
            self.log(f"val/{k}", v, on_epoch=True)

        # HeatmapImageLogger用の出力
        return {
            "loss": loss.detach(),
            "images": x.detach().cpu(),
            "pred_heatmaps": out["heatmap"].detach().cpu() if isinstance(out, dict) else None,
            "target_heatmaps": y.detach().cpu(),
        }

    def configure_optimizers(self):
        tcfg = getattr(self.cfg, "training", {})
        lr = float(getattr(tcfg, "lr", 3e-4))
        wd = float(getattr(tcfg, "weight_decay", 1e-4))
        betas = tuple(getattr(tcfg, "betas", (0.9, 0.999)))
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=betas)

        max_epochs = int(getattr(tcfg, "max_epochs", 50))
        warmup_epochs = int(getattr(tcfg, "warmup_epochs", 0))
        eta_min = float(getattr(tcfg, "eta_min", 1e-6))

        scheds = []
        milestones = []
        if warmup_epochs > 0:
            scheds.append(
                optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            )
            milestones.append(warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - warmup_epochs), eta_min=eta_min
        )
        scheduler = (
            optim.lr_scheduler.SequentialLR(optimizer, scheds + [cosine], milestones=milestones) if scheds else cosine
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val/loss"},
        }
