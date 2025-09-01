from __future__ import annotations

from typing import Dict, List

import torch
from pytorch_lightning import LightningModule

from ...utils.loss import loss_registry
from .model.ball_heatmap_model import BallHeatmapModel


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 25.0,
    pos_mask_tau: float = 0.3,
) -> torch.Tensor:
    # pred, target: [B,1,H,W]
    pos_mask = (target >= pos_mask_tau).float()
    neg_mask = 1.0 - pos_mask
    diff2 = (pred - target) ** 2
    pos_loss = (diff2 * pos_mask).sum()
    neg_loss = (diff2 * neg_mask).sum()
    # avoid div by zero
    n_pos = pos_mask.sum().clamp(min=1.0)
    n_neg = neg_mask.sum().clamp(min=1.0)
    pos_loss = pos_loss / n_pos
    neg_loss = neg_loss / n_neg
    return pos_weight * pos_loss + neg_loss


class BallLitModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.model = BallHeatmapModel(
            vit_name=config.model.vit_name,
            pretrained=config.model.pretrained,
            decoder_channels=None,  # use default
            deep_supervision_strides=list(config.model.deep_supervision_strides),
            heatmap_channels=int(config.model.heatmap_channels),
            offset_channels=int(config.model.offset_channels),
        )

        # Loss settings
        self.lambda_h = float(config.training.loss.lambda_hmap)
        self.lambda_o = float(config.training.loss.lambda_offset)
        self.lambda_c = float(getattr(config.training.loss, "lambda_coord", 0.0))
        self.deep_w: List[float] = list(config.training.loss.deep_supervision_weights)
        self.use_focal = bool(config.training.loss.focal)
        self.pos_weight = float(config.training.loss.pos_weight)
        self.pos_mask_tau = float(config.training.loss.pos_mask_tau)

        # Optional focal loss
        self.focal_loss = loss_registry.get("focal", alpha=0.25, gamma=2.0) if self.use_focal else None

        self.freeze_vit_epochs = int(config.training.freeze_vit_epochs)
        if self.freeze_vit_epochs > 0:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        # Offset supervision mask threshold
        self.offset_tau = float(config.dataset.heatmap.offset_mask_tau)

        self.ds_strides: List[int] = list(config.model.deep_supervision_strides)  # ←追加

    # ---- Optimizer with param groups ----
    def configure_optimizers(self):
        vit_params = list(self.model.encoder.parameters())
        other_params = list(self.model.decoder.parameters()) + list(self.model.heads.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": float(self.config.training.lr_head)},
                {"params": vit_params, "lr": float(self.config.training.vit_lr)},
            ],
            weight_decay=float(self.config.training.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/total_loss"}}

    # ---- Epoch hooks ----
    def on_train_epoch_start(self) -> None:
        # Unfreeze ViT after configured epochs
        if self.current_epoch == self.freeze_vit_epochs:
            for p in self.model.encoder.parameters():
                p.requires_grad = True

        # Update datamodule sampling schedule if available
        if self.trainer and self.trainer.datamodule and hasattr(self.trainer.datamodule, "update_sampling_for_epoch"):
            self.trainer.datamodule.update_sampling_for_epoch(self.current_epoch)

    # ---- Steps ----
    def training_step(self, batch, batch_idx):
        out = self._forward_and_loss(batch)
        self.log("train/total_loss", out["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_heatmap", out["loss_h"], on_epoch=True)
        self.log("train/loss_offset", out["loss_o"], on_epoch=True)

        # NEW: per-scale logs
        for stride, lh, lo, pr, mae in zip(
            self.ds_strides, out["loss_h_scales"], out["loss_o_scales"], out["pos_ratio_scales"], out["off_mae_scales"]
        ):
            self.log(f"train/loss_h.s{stride}", lh, on_epoch=True)
            self.log(f"train/loss_o.s{stride}", lo, on_epoch=True)
            self.log(f"train/pos_ratio.s{stride}", pr, on_epoch=True)
            self.log(f"train/offset_mae.s{stride}", mae, on_epoch=True)

        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._forward_and_loss(batch)
        self.log("val/total_loss", out["loss"], prog_bar=True, on_epoch=True)
        self.log("val/loss_heatmap", out["loss_h"], on_epoch=True)
        self.log("val/loss_offset", out["loss_o"], on_epoch=True)

        # NEW: per-scale logs
        for stride, lh, lo, pr, mae in zip(
            self.ds_strides, out["loss_h_scales"], out["loss_o_scales"], out["pos_ratio_scales"], out["off_mae_scales"]
        ):
            self.log(f"val/loss_h.s{stride}", lh, on_epoch=True)
            self.log(f"val/loss_o.s{stride}", lo, on_epoch=True)
            self.log(f"val/pos_ratio.s{stride}", pr, on_epoch=True)
            self.log(f"val/offset_mae.s{stride}", mae, on_epoch=True)

        # PCK metrics: (1) heatmap-only argmax, (2) argmax+offset in image space
        pred_hmap_hi = out["pred_heatmaps"][-1]
        tgt_hmap_hi = out["tgt_heatmaps"][-1]
        img_H, img_W = batch["image"].shape[-2:]
        gt_xy_img = batch["coord"].to(pred_hmap_hi.device)
        valid = batch["valid_mask"].to(pred_hmap_hi.device)
        pred_off_hi = out["pred_offsets"][-1]
        for thr in list(self.config.evaluation.pck_thresholds):
            pck_hm = self._pck(pred_hmap_hi, tgt_hmap_hi, threshold_ratio=float(thr))
            pck_off = self._pck_offset_img(
                pred_hmap_hi, pred_off_hi, gt_xy_img, valid, (img_H, img_W), threshold_ratio=float(thr)
            )
            self.log(f"val/PCK_hm@{thr}", pck_hm, on_epoch=True)
            self.log(
                f"val/PCK_off@{thr}",
                pck_off,
                on_epoch=True,
                prog_bar=(thr == self.config.evaluation.pck_thresholds[-1]),
            )

        # For HeatmapLoggerV2
        return {
            "images": batch["image"].detach().cpu(),
            "pred_heatmaps": [h.detach().cpu() for h in out["pred_heatmaps"]],
            "pred_offsets": [o.detach().cpu() for o in out["pred_offsets"]],
            "target_heatmaps": [h.detach().cpu() for h in out["tgt_heatmaps"]],
            "gt_coords_img": batch["coord"].detach().cpu(),
            "valid_mask": batch["valid_mask"].detach().cpu(),
        }

    def test_step(self, batch, batch_idx):
        out = self._forward_and_loss(batch)
        self.log("test/total_loss", out["loss"], prog_bar=True, on_epoch=True)
        self.log("test/loss_heatmap", out["loss_h"], on_epoch=True)
        self.log("test/loss_offset", out["loss_o"], on_epoch=True)
        return out["loss"]

    # ---- Core ----
    def _forward_and_loss(self, batch) -> Dict[str, torch.Tensor]:
        images: torch.Tensor = batch["image"]
        tgt_hmaps: List[torch.Tensor] = batch["heatmaps"]
        tgt_offs: List[torch.Tensor] = batch["offsets"]
        tgt_hmaps = [t.to(images.device) for t in tgt_hmaps]
        tgt_offs = [t.to(images.device) for t in tgt_offs]

        outputs = self.model(images)
        pred_hmaps: List[torch.Tensor] = outputs["heatmaps"]
        pred_offs: List[torch.Tensor] = outputs["offsets"]

        # NEW: per-scale holders
        loss_h_scales: List[torch.Tensor] = []
        loss_o_scales: List[torch.Tensor] = []
        pos_ratio_scales: List[torch.Tensor] = []
        off_mae_scales: List[torch.Tensor] = []

        loss_h_total = 0.0
        loss_o_total = 0.0

        for s, (wh, ph, wo, po, w) in enumerate(zip(tgt_hmaps, pred_hmaps, tgt_offs, pred_offs, self.deep_w)):
            # Heatmap loss
            if self.focal_loss is not None:
                bin_wh = (wh >= self.pos_mask_tau).float()
                loss_h = self.focal_loss(ph, bin_wh)
                pos_ratio = bin_wh.mean()
            else:
                loss_h = weighted_mse_loss(ph, wh, pos_weight=self.pos_weight, pos_mask_tau=self.pos_mask_tau)
                pos_ratio = (wh >= self.pos_mask_tau).float().mean()

            # Offset loss with mask from target heatmap
            mask = (wh > self.offset_tau).float()
            if mask.sum() > 0:
                l1 = (torch.abs(po - wo) * mask).sum() / mask.sum()
                # 参考: MAE(px) をログ用に
                off_mae = l1.detach()
            else:
                l1 = torch.zeros((), device=images.device)
                off_mae = torch.zeros((), device=images.device)

            # collect per-scale (unweighted) values
            loss_h_scales.append(loss_h.detach())
            loss_o_scales.append(l1.detach())
            pos_ratio_scales.append(pos_ratio.detach())
            off_mae_scales.append(off_mae)

            # deep supervision の重みで合算
            loss_h_total = loss_h_total + w * loss_h
            loss_o_total = loss_o_total + w * l1

        total = self.lambda_h * loss_h_total + self.lambda_o * loss_o_total

        return {
            "loss": total,
            "loss_h": loss_h_total.detach(),
            "loss_o": loss_o_total.detach(),
            "pred_heatmaps": pred_hmaps,
            "pred_offsets": pred_offs,
            "tgt_heatmaps": tgt_hmaps,
            # NEW: per-scale tensors (lists)
            "loss_h_scales": loss_h_scales,
            "loss_o_scales": loss_o_scales,
            "pos_ratio_scales": pos_ratio_scales,  # 正例画素の割合
            "off_mae_scales": off_mae_scales,  # offset の MAE(px)
        }

    # ---- Metrics ----
    @staticmethod
    def _coords_from_heatmap(hmap: torch.Tensor) -> torch.Tensor:
        # hmap: [B,1,H,W] or [B,C,H,W] -> get per-channel argmax; assume C==1 here
        b, c, _h, w = hmap.shape
        flat = hmap.view(b, c, -1)
        idx = flat.argmax(dim=-1)  # [B,C]
        ys = (idx // w).float()
        xs = (idx % w).float()
        coords = torch.stack([xs, ys], dim=-1)  # [B,C,2]
        return coords

    def _pck(self, pred_hmap: torch.Tensor, tgt_hmap: torch.Tensor, threshold_ratio: float = 0.05) -> torch.Tensor:
        # Expect shape [B,1,H,W]
        pred_xy = self._coords_from_heatmap(pred_hmap)
        tgt_xy = self._coords_from_heatmap(tgt_hmap)
        h, w = pred_hmap.shape[-2:]
        thr = threshold_ratio * (h**2 + w**2) ** 0.5
        d = torch.linalg.norm(pred_xy - tgt_xy, dim=-1)  # [B,1]
        tgt_max = tgt_hmap.view(tgt_hmap.shape[0], tgt_hmap.shape[1], -1).max(dim=-1).values
        vis = (tgt_max > 1e-6).float()
        correct = ((d < thr).float() * vis).sum()
        total = vis.sum().clamp(min=1.0)
        return correct / total

    # Offset-inclusive PCK in image space
    @staticmethod
    def _coords_from_hmap_and_offset_img(hmap: torch.Tensor, offs: torch.Tensor, stride: int) -> torch.Tensor:
        # hmap: [B,1,Hs,Ws], offs: [B,2,Hs,Ws]
        b, _, _h, w = hmap.shape
        flat = hmap.view(b, -1)
        idx = flat.argmax(dim=-1)
        ys = (idx // w).float()
        xs = (idx % w).float()
        dx = offs[:, 0].contiguous().view(b, -1).gather(1, idx.view(-1, 1)).squeeze(1)
        dy = offs[:, 1].contiguous().view(b, -1).gather(1, idx.view(-1, 1)).squeeze(1)
        x_img = (xs + dx) * float(stride)
        y_img = (ys + dy) * float(stride)
        return torch.stack([x_img, y_img], dim=-1)  # [B,2]

    def _pck_offset_img(
        self,
        pred_hmap: torch.Tensor,
        pred_off: torch.Tensor,
        gt_xy_img: torch.Tensor,
        valid: torch.Tensor,
        img_hw: tuple[int, int],
        threshold_ratio: float = 0.05,
    ) -> torch.Tensor:
        stride = int(self.config.model.deep_supervision_strides[-1])
        pred_xy_img = self._coords_from_hmap_and_offset_img(pred_hmap, pred_off, stride)
        H, W = img_hw
        thr = threshold_ratio * ((H**2 + W**2) ** 0.5)
        d = torch.linalg.norm(pred_xy_img - gt_xy_img, dim=-1)  # [B]
        vis = valid.float().view(-1)
        correct = ((d < thr).float() * vis).sum()
        total = vis.sum().clamp(min=1.0)
        return correct / total
