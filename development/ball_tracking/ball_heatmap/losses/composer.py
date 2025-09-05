from __future__ import annotations
from typing import Any, Dict, Tuple

import torch
from contextlib import nullcontext

from .frame import (
    heatmap_loss,
    speed_huber,
    vis_ce,
    offset_huber,
)
from .regularization import (
    tv_loss,
    ranking_margin_loss,
    peak_sharpening_loss,
    peak_lower_bound_loss,
)
from ..trackers.features import soft_argmax_2d


class LossComposer:
    """Orchestrates the calculation of multiple weighted supervised losses."""

    def __init__(self, cfg_losses: Any):
        """Initializes the LossComposer with loss weights and settings."""
        self.w = cfg_losses

    def __call__(
        self, preds: Dict[str, Any], targets: Dict[str, Any], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculates the total supervised loss and returns it along with a log dictionary.
        Loss math is forced to run in float32 to avoid FP16 underflow with mixed precision.
        """
        total_sup_loss = torch.tensor(0.0, device=device)
        log_dict = {}

        mask_hm = targets["vis_mask_hm"].to(device)
        mask_speed = targets["vis_mask_speed"].to(device)

        # Disable autocast for numerically stable loss computation
        dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
        ac = torch.autocast(device_type=dev_type, enabled=False) if hasattr(torch, "autocast") else nullcontext()
        with ac:
            # --- Core Losses ---
            loss_hm = heatmap_loss(
                preds["heatmaps"], [t.to(device) for t in targets["hm"]], mask_hm, kind=str(self.w.hm_type)
            )
            loss_speed = speed_huber(preds["speed"], targets["speed"].to(device), mask_speed)
            loss_vis = vis_ce(preds["vis_logits"], targets["vis_state"].to(device))

            total_sup_loss = total_sup_loss + (
                self.w.lambda_hm * loss_hm + self.w.lambda_speed * loss_speed + self.w.lambda_vis * loss_vis
            )
            log_dict.update({"sup/hm": loss_hm, "sup/speed": loss_speed, "sup/vis": loss_vis})

            # --- Optional Losses ---
            if self.w.lambda_off > 0 and "offset" in preds and "offset" in targets:
                loss_off = offset_huber(preds["offset"], targets["offset"].to(device), mask_hm)
                total_sup_loss = total_sup_loss + self.w.lambda_off * loss_off
                log_dict["sup/offset"] = loss_off

            # For TV losses, calculate coordinates once from the highest-resolution heatmap
            if self.w.lambda_v > 0 or self.w.lambda_a > 0:
                pred_coords = soft_argmax_2d(preds["heatmaps"][0])
                if self.w.lambda_v > 0:
                    loss_v = tv_loss(pred_coords, mask_hm, order=1)
                    total_sup_loss = total_sup_loss + self.w.lambda_v * loss_v
                    log_dict["sup/tv_v"] = loss_v
                if self.w.lambda_a > 0:
                    loss_a = tv_loss(pred_coords, mask_hm, order=2)
                    total_sup_loss = total_sup_loss + self.w.lambda_a * loss_a
                    log_dict["sup/tv_a"] = loss_a

            # Use the highest-resolution heatmap for peak-based losses
            pred_hm_main = preds["heatmaps"][0]
            tgt_hm_main = targets["hm"][0].to(device)

            if self.w.lambda_rank > 0:
                loss_rank = ranking_margin_loss(pred_hm_main, tgt_hm_main, mask_hm)
                total_sup_loss = total_sup_loss + self.w.lambda_rank * loss_rank
                log_dict["sup/rank"] = loss_rank

            if self.w.lambda_sharp > 0:
                loss_sharp = peak_sharpening_loss(pred_hm_main, mask_hm)
                total_sup_loss = total_sup_loss + self.w.lambda_sharp * loss_sharp
                log_dict["sup/sharp"] = loss_sharp

            if self.w.lambda_peak > 0:
                loss_peak = peak_lower_bound_loss(pred_hm_main, mask_hm)
                total_sup_loss = total_sup_loss + self.w.lambda_peak * loss_peak
                log_dict["sup/peak"] = loss_peak

        log_dict["sup/total"] = total_sup_loss.detach()
        return total_sup_loss, log_dict
