from __future__ import annotations
from typing import Any, Dict, List, Optional
import contextlib

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from development.ball_tracking.ball_heatmap.losses.composer import LossComposer
from development.ball_tracking.ball_heatmap.losses.frame import heatmap_loss, speed_huber, vis_ce
from development.ball_tracking.ball_heatmap.losses.metrics import keypoint_accuracy
from development.ball_tracking.ball_heatmap.trackers.features import soft_argmax_2d
from development.ball_tracking.ball_heatmap.model.ball_heatmap_model import BallHeatmapModel


class BallLightningModule(pl.LightningModule):
    def __init__(self, cfg, semisup_strategy=None, adversary=None):
        super().__init__()
        self.save_hyperparameters(ignore=["semisup_strategy", "adversary"])
        self.cfg = cfg
        self.model = BallHeatmapModel(self.cfg)
        self.loss_composer = LossComposer(self.cfg.losses)
        self.teacher = None
        if cfg.semisup.get("enable", False):
            self.teacher = BallHeatmapModel(self.cfg)
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        self.semisup_strategy = semisup_strategy
        self.adversary = adversary
        self.automatic_optimization = False

    def forward(self, x_btc: torch.Tensor, x_strong: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        if self.model.use_ssl and x_strong is None:
            preds = self.model(x_btc, x_btc)
            return {
                "heatmaps": preds["heatmaps_strong"],
                "speed": preds["speed_strong"],
                "vis_logits": preds["vis_logits_strong"],
                "aux_losses": preds["aux_losses"],
            }
        return self.model(x_btc, x_strong)

    def configure_optimizers(self):
        # --- Optimizers ---
        params = [
            {"params": self.model.encoder.parameters(), "lr": self.cfg.opt.lr_backbone},
            {
                "params": list(self.model.decoder.parameters()) + list(self.model.heads_per_scale.parameters()),
                "lr": self.cfg.opt.lr_heads,
            },
        ]
        opt_g = torch.optim.AdamW(params, weight_decay=self.cfg.opt.wd)

        # Helper to build (Linear warmup -> Cosine) scheduler
        def _build_cosine_scheduler(optimizer, cfg_sched, *, default_name: str, is_step_based: bool):
            # Defaults & reads
            warmup_start_factor = float(getattr(cfg_sched, "warmup_start_factor", 0.1))
            min_lr = float(getattr(cfg_sched, "min_lr", 1e-6))

            if is_step_based:
                # Step-based scheduling
                # Derive steps
                total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
                if total_steps <= 0:
                    # Fallback: epochs * steps_per_epoch
                    steps_per_epoch = max(1, getattr(self.trainer, "num_training_batches", 0))
                    total_steps = max(1, steps_per_epoch * max(1, self.trainer.max_epochs))
                # Warmup steps
                warmup_steps = getattr(cfg_sched, "warmup_steps", None)
                if warmup_steps is None:
                    warmup_epochs = int(getattr(cfg_sched, "warmup_epochs", 0))
                    steps_per_epoch = max(1, total_steps // max(1, self.trainer.max_epochs))
                    warmup_steps = max(0, warmup_epochs * steps_per_epoch)
                warmup_steps = int(warmup_steps)
                # T_max after warmup
                t_max = max(1, total_steps - warmup_steps)

                warmup = (
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=warmup_start_factor,
                        total_iters=max(1, warmup_steps),
                    )
                    if warmup_steps > 0
                    else None
                )

                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)

                if warmup is None:
                    sched = cosine
                    interval = "step"
                else:
                    sched = torch.optim.lr_scheduler.SequentialLR(
                        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
                    )
                    interval = "step"

            else:
                # Epoch-based scheduling
                max_epochs = int(getattr(self.trainer, "max_epochs", 1))
                warmup_epochs = int(getattr(cfg_sched, "warmup_epochs", 0))
                t_max = max(1, max_epochs - warmup_epochs)

                warmup = (
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=warmup_start_factor,
                        total_iters=max(1, warmup_epochs),
                    )
                    if warmup_epochs > 0
                    else None
                )

                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)

                if warmup is None:
                    sched = cosine
                else:
                    sched = torch.optim.lr_scheduler.SequentialLR(
                        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
                    )
                interval = "epoch"

            return {
                "scheduler": sched,
                "interval": interval,  # "epoch" or "step"
                "frequency": 1,
                "name": default_name,
            }

        # --- Generator scheduler ---
        sched_cfg = getattr(self.cfg.opt, "scheduler", {})
        is_step_based = str(getattr(sched_cfg, "interval", "epoch")).lower() == "step"
        sch_g = _build_cosine_scheduler(opt_g, sched_cfg, default_name="cosine_g", is_step_based=is_step_based)

        # --- With adversary (optional) ---
        if self.adversary and self.cfg.gan.get("enable", False):
            # Expect your adversary to expose its optimizer (opt_d) already.
            opt_d = self.adversary.opt_d

            # If you also want a cosine schedule for D, either:
            # 1) mirror cfg under self.cfg.gan.scheduler, or
            # 2) reuse the generator's scheduler config.
            gan_sched_cfg = getattr(self.cfg.gan, "scheduler", sched_cfg)
            is_step_based_d = str(getattr(gan_sched_cfg, "interval", "epoch")).lower() == "step"
            sch_d = _build_cosine_scheduler(
                opt_d, gan_sched_cfg, default_name="cosine_d", is_step_based=is_step_based_d
            )

            return [opt_g, opt_d], [sch_g, sch_d]

        # --- No adversary ---
        return {"optimizer": opt_g, "lr_scheduler": sch_g}

    def _construct_trajectory(self, preds: Dict = None, targets: Dict = None) -> torch.Tensor:
        if preds:
            coords = soft_argmax_2d(preds["heatmaps"][0])
            vis = torch.softmax(preds["vis_logits"], dim=-1)
            vel = preds["speed"]
        elif targets:
            coords = soft_argmax_2d(targets["hm"][0].to(self.device))
            vis = F.one_hot(targets["vis_state"].to(self.device), num_classes=3).float()
            vel = targets["speed"].to(self.device)
        else:
            raise ValueError("Either preds or targets must be provided.")

        acc = torch.diff(vel, dim=1, prepend=vel[:, :1])
        return torch.cat([coords, vis, vel, acc], dim=-1)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = (
            self.optimizers() if (self.adversary and self.cfg.gan.get("enable", False)) else (self.optimizers(), None)
        )
        total_loss = torch.tensor(0.0, device=self.device)
        log_dict = {}

        # Supervised loss on labeled data
        sup_batch = batch.get("labeled", batch)
        if "sup" in sup_batch:
            sup = sup_batch["sup"]
            video = sup["video"]
            targets = sup["targets"]
            preds = self(video)

            # Calculate supervised loss using the composer
            sup_loss, sup_logs = self.loss_composer(preds, targets, self.device)
            total_loss += sup_loss
            log_dict.update(sup_logs)

            # --- GAN Discriminator Step ---
            if self.adversary and self.cfg.gan.get("enable", False) and (batch_idx + 1) % self.cfg.gan.d_every == 0:
                trajectories_fake = self._construct_trajectory(preds=preds)
                trajectories_real = self._construct_trajectory(targets=targets)
                d_logs = self.adversary.d_step(self, trajectories_real.detach(), trajectories_fake.detach())
                self.log_dict({f"adv/{k}": v for k, v in d_logs.items()})

            # --- GAN Generator Step ---
            if self.adversary and self.cfg.gan.get("enable", False) and (batch_idx + 1) % self.cfg.gan.g_every == 0:
                trajectories_fake = self._construct_trajectory(preds=preds)
                g_loss = self.adversary.g_loss(self, trajectories_fake)
                total_loss += g_loss
                log_dict["adv/g_loss"] = g_loss

        # --- Semi-supervised loss on unlabeled data ---
        if self.semisup_strategy and self.cfg.semisup.get("enable", False):
            unsup_batch = batch.get("unlabeled")
            if unsup_batch is not None:
                unsup_losses = self.semisup_strategy.training_step(self, unsup_batch, self.global_step)
                if unsup_losses:
                    for k, v in unsup_losses.items():
                        if isinstance(v, torch.Tensor) and v.requires_grad:
                            total_loss += v
                            log_dict[k] = v

        # --- Optimizer Step for Generator ---
        opt_g.zero_grad()
        self.manual_backward(total_loss)

        # Optional: log gradient norm to help detect vanishing/exploding grads
        try:
            with torch.no_grad():
                sqsum = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        g = p.grad.detach().data
                        sqsum += (g.float().norm(2).item()) ** 2
                grad_norm = (sqsum**0.5) if sqsum > 0 else 0.0
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        except Exception:
            pass
        opt_g.step()

        log_dict["train/loss"] = total_loss
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        sup = batch["sup"]
        video = sup["video"]
        targets = sup["targets"]
        preds = self(video)

        # For validation, we can still calculate the loss directly or use the composer
        # Using composer for consistency
        val_loss, val_logs = self.loss_composer(preds, targets, self.device)

        H, W = video.shape[-2:]
        acc = keypoint_accuracy(
            preds["heatmaps"],
            [t.to(self.device) for t in targets["hm"]],
            targets["vis_mask_hm"].to(self.device),
            (int(H), int(W)),
        )

        # Prefix logs with 'val/'
        val_logs = {f"val/{k.replace('sup/', '')}": v for k, v in val_logs.items()}
        val_logs["val/acc"] = acc
        self.log_dict(val_logs, on_step=False, on_epoch=True)

        callback_data = {
            "images": video.cpu(),
            "pred_heatmaps": [h.cpu() for h in preds["heatmaps"]],
            "target_heatmaps": [t.cpu() for t in targets["hm"]],
            "valid_mask": targets["vis_mask_hm"].cpu(),
        }
        return callback_data

    def compute_semisup_losses(
        self,
        preds_s: Dict[str, Any],
        hm_pseudo: List[torch.Tensor],
        speed_pseudo: torch.Tensor,
        vis_state_pseudo: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, Any]:
        # Compute semi-supervised losses in float32 (disable autocast) to avoid underflow
        device = preds_s["speed"].device
        mask = mask.to(device)
        w = self.cfg.losses
        dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
        ac = (
            torch.autocast(device_type=dev_type, enabled=False)
            if hasattr(torch, "autocast")
            else contextlib.nullcontext()
        )
        with ac:
            loss_hm = heatmap_loss(preds_s["heatmaps"], [h.to(device) for h in hm_pseudo], mask, kind=str(w.hm_type))
            loss_speed = speed_huber(preds_s["speed"], speed_pseudo.to(device), mask)
            loss_vis = vis_ce(preds_s["vis_logits"], vis_state_pseudo.to(device))
            loss_unsup = w.lambda_hm * loss_hm + w.lambda_speed * loss_speed + w.lambda_vis * loss_vis
        self.log_dict(
            {"unsup/hm": loss_hm, "unsup/speed": loss_speed, "unsup/vis": loss_vis},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        return {"loss_unsup": loss_unsup}
