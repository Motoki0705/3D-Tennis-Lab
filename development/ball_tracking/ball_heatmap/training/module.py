from __future__ import annotations
from typing import Any, Dict, List, Optional

import torch
import pytorch_lightning as pl

from development.ball_tracking.ball_heatmap.losses.supervised import (
    heatmap_loss,
    speed_huber,
    vis_ce,
    keypoint_accuracy,
)
from development.ball_tracking.ball_heatmap.model.ball_heatmap_model import BallHeatmapModel


class BallLightningModule(pl.LightningModule):
    def __init__(self, cfg, semisup_strategy=None, adversary=None):
        super().__init__()
        self.save_hyperparameters(ignore=["semisup_strategy", "adversary"])
        self.cfg = cfg
        self.model = BallHeatmapModel(self.cfg)
        self.teacher = None
        if cfg.semisup.get("enable", False):
            # For now, mirror student as teacher; load ckpt if available
            self.teacher = BallHeatmapModel(self.cfg)
            # TODO: load teacher checkpoint from cfg.semisup.teacher_ckpt
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        self.semisup_strategy = semisup_strategy
        self.adversary = adversary

    def forward(self, x_btc: torch.Tensor, x_strong: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        # The model's forward pass requires both weak and strong inputs when SSL is enabled.
        # However, strategies might call `module(strong_aug_data)`, which only provides one tensor.
        # This is an attempt to bridge the gap.
        if self.model.use_ssl and x_strong is None:
            # This call is likely from a strategy, passing only the strongly augmented data.
            # We lack the corresponding weakly augmented data here.
            # As a workaround, we pass the strong data as the weak input as well.
            # This will produce valid tensor shapes, but the consistency loss might not be meaningful.
            # A proper fix would involve refactoring the strategy pattern to have access to both augmentations.
            preds = self.model(x_btc, x_btc)
            # The strategy expects the "non-strong" keys. We return the strong predictions under the standard keys.
            return {
                "heatmaps": preds["heatmaps_strong"],
                "speed": preds["speed_strong"],
                "vis_logits": preds["vis_logits_strong"],
                "aux_losses": preds["aux_losses"],
            }
        return self.model(x_btc, x_strong)

    def configure_optimizers(self):
        params = [
            {"params": self.model.encoder.parameters(), "lr": self.cfg.opt.lr_backbone},
            {
                "params": list(self.model.decoder.parameters()) + list(self.model.heads_per_scale.parameters()),
                "lr": self.cfg.opt.lr_heads,
            },
        ]
        opt = torch.optim.AdamW(params, weight_decay=self.cfg.opt.wd)
        return opt

    def training_step(self, batch, batch_idx):
        total_loss = 0.0
        log_dict = {}

        # Supervised loss
        # batch can be a mix of labeled and unlabeled from a CombinedLoader
        sup_batch = batch.get("labeled", batch)
        if "sup" in sup_batch:
            sup = sup_batch["sup"]
            video = sup["video"]
            targets = sup["targets"]
            preds = self(video)  # This calls forward(video, x_strong=None)

            mask_hm = targets["vis_mask_hm"].to(video.device)
            # Heatmap loss (config-selectable: mse|focal|kl)
            hm_kind = str(getattr(self.cfg.losses, "hm_type", "mse")).lower()
            focal_gamma = float(getattr(self.cfg.losses, "focal_gamma", 2.0))
            kl_tau = float(getattr(self.cfg.losses, "kl_tau", 1.0))
            loss_hm = heatmap_loss(
                preds["heatmaps"],
                [t.to(video.device) for t in targets["hm"]],
                mask_hm,
                kind=hm_kind,
                focal_gamma=focal_gamma,
                kl_tau=kl_tau,
            )
            loss_speed = speed_huber(
                preds["speed"], targets["speed"].to(video.device), targets["vis_mask_speed"].to(video.device)
            )
            loss_vis = vis_ce(preds["vis_logits"], targets["vis_state"].to(video.device))
            w = self.cfg.losses
            sup_loss = w.lambda_hm * loss_hm + w.lambda_speed * loss_speed + w.lambda_vis * loss_vis
            total_loss += sup_loss
            # Accuracy metric @ threshold (pixels)
            H, W = video.shape[-2:]
            acc_thr = float(getattr(self.cfg.losses, "acc_threshold_px", 8.0))
            acc = keypoint_accuracy(
                preds["heatmaps"],
                [t.to(video.device) for t in targets["hm"]],
                mask_hm,
                (int(H), int(W)),
                threshold_px=acc_thr,
            )
            log_dict.update({
                "sup/hm": loss_hm,
                "sup/speed": loss_speed,
                "sup/vis": loss_vis,
                "sup/acc": acc,
                "sup/loss": sup_loss,
            })

            # Per-scale metrics (heatmap loss and accuracy)
            try:
                strides = list(getattr(self.cfg.model, "deep_supervision_strides", []))
            except Exception:
                strides = []
            pred_hm_list = preds["heatmaps"]
            tgt_hm_list = [t.to(video.device) for t in targets["hm"]]
            for si, (p, t) in enumerate(zip(pred_hm_list, tgt_hm_list)):
                s_val = strides[si] if si < len(strides) else (si + 1)
                l_si = heatmap_loss([p], [t], mask_hm, kind=hm_kind, focal_gamma=focal_gamma, kl_tau=kl_tau)
                a_si = keypoint_accuracy([p], [t], mask_hm, (int(H), int(W)), threshold_px=acc_thr)
                log_dict[f"sup/hm/stride_{s_val}"] = l_si
                log_dict[f"sup/acc/stride_{s_val}"] = a_si

            if "aux_losses" in preds and preds["aux_losses"]:
                for k, v in preds["aux_losses"].items():
                    if isinstance(v, torch.Tensor) and v.requires_grad:
                        total_loss += v
                        log_dict[f"aux/{k}"] = v

        # Unlabeled semi-supervision
        if self.semisup_strategy and self.cfg.semisup.get("enable", False):
            unsup_batch = batch.get("unlabeled")
            if unsup_batch is not None:
                unsup_losses = self.semisup_strategy.training_step(self, unsup_batch, self.global_step)
                if unsup_losses:
                    for k, v in unsup_losses.items():
                        if isinstance(v, torch.Tensor) and v.requires_grad:
                            total_loss += v
                            log_dict[k] = v  # e.g., loss_unsup

        log_dict["train/loss"] = total_loss
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # This method is implicitly called by the trainer.
        # We define it to log validation metrics and prepare data for visualization callbacks.
        sup = batch["sup"]
        video = sup["video"]
        targets = sup["targets"]
        preds = self(video)

        mask_hm = targets["vis_mask_hm"].to(video.device)
        hm_kind = str(getattr(self.cfg.losses, "hm_type", "mse")).lower()
        focal_gamma = float(getattr(self.cfg.losses, "focal_gamma", 2.0))
        kl_tau = float(getattr(self.cfg.losses, "kl_tau", 1.0))
        loss_hm = heatmap_loss(
            preds["heatmaps"],
            [t.to(video.device) for t in targets["hm"]],
            mask_hm,
            kind=hm_kind,
            focal_gamma=focal_gamma,
            kl_tau=kl_tau,
        )
        loss_speed = speed_huber(
            preds["speed"], targets["speed"].to(video.device), targets["vis_mask_speed"].to(video.device)
        )
        loss_vis = vis_ce(preds["vis_logits"], targets["vis_state"].to(video.device))
        w = self.cfg.losses
        val_loss = w.lambda_hm * loss_hm + w.lambda_speed * loss_speed + w.lambda_vis * loss_vis

        H, W = video.shape[-2:]
        acc_thr = float(getattr(self.cfg.losses, "acc_threshold_px", 8.0))
        acc = keypoint_accuracy(
            preds["heatmaps"],
            [t.to(video.device) for t in targets["hm"]],
            mask_hm,
            (int(H), int(W)),
            threshold_px=acc_thr,
        )

        self.log_dict(
            {
                "val/hm": loss_hm,
                "val/speed": loss_speed,
                "val/vis": loss_vis,
                "val/acc": acc,
                "val/loss": val_loss,
            },
            on_step=False,
            on_epoch=True,
        )

        # Per-scale metrics for validation
        try:
            strides = list(getattr(self.cfg.model, "deep_supervision_strides", []))
        except Exception:
            strides = []
        pred_hm_list = preds["heatmaps"]
        tgt_hm_list = [t.to(video.device) for t in targets["hm"]]
        per_scale_logs = {}
        for si, (p, t) in enumerate(zip(pred_hm_list, tgt_hm_list)):
            s_val = strides[si] if si < len(strides) else (si + 1)
            l_si = heatmap_loss([p], [t], mask_hm, kind=hm_kind, focal_gamma=focal_gamma, kl_tau=kl_tau)
            a_si = keypoint_accuracy([p], [t], mask_hm, (int(H), int(W)), threshold_px=acc_thr)
            per_scale_logs[f"val/hm/stride_{s_val}"] = l_si
            per_scale_logs[f"val/acc/stride_{s_val}"] = a_si
        if per_scale_logs:
            self.log_dict(per_scale_logs, on_step=False, on_epoch=True)

        # Prepare data for visualization callbacks (e.g., HeatmapLogger)
        # Move tensors to CPU to avoid cluttering GPU memory.
        callback_data = {
            "images": video.cpu(),
            "pred_heatmaps": [h.cpu() for h in preds["heatmaps"]],
            "target_heatmaps": [t.cpu() for t in targets["hm"]],
            # "gt_coords_img": targets["coords_img"].cpu(),
            "valid_mask": targets["vis_mask_hm"].cpu(),
        }
        # The callback will receive this data in `on_validation_batch_end`.
        return callback_data

    def compute_semisup_losses(
        self,
        preds_s: Dict[str, Any],
        hm_pseudo: List[torch.Tensor],
        speed_pseudo: torch.Tensor,
        vis_state_pseudo: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, Any]:
        mask = mask.to(preds_s["speed"].device)
        hm_kind = str(getattr(self.cfg.losses, "hm_type", "mse")).lower()
        focal_gamma = float(getattr(self.cfg.losses, "focal_gamma", 2.0))
        kl_tau = float(getattr(self.cfg.losses, "kl_tau", 1.0))
        loss_hm = heatmap_loss(
            preds_s["heatmaps"],
            [h.to(mask.device) for h in hm_pseudo],
            mask,
            kind=hm_kind,
            focal_gamma=focal_gamma,
            kl_tau=kl_tau,
        )
        loss_speed = speed_huber(preds_s["speed"], speed_pseudo.to(mask.device), mask)
        loss_vis = vis_ce(preds_s["vis_logits"], vis_state_pseudo.to(mask.device))
        w = self.cfg.losses
        loss_unsup = w.lambda_hm * loss_hm + w.lambda_speed * loss_speed + w.lambda_vis * loss_vis
        self.log_dict(
            {
                "unsup/hm": loss_hm,
                "unsup/speed": loss_speed,
                "unsup/vis": loss_vis,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        return {"loss_unsup": loss_unsup}
