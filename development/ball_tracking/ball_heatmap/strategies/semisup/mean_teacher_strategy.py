from __future__ import annotations
from typing import Any, Dict

import torch

from .base import SemisupStrategy
from development.ball_tracking.ball_heatmap.losses.semisup import (
    hm_consistency,
    speed_consistency,
    vislogit_consistency,
)


class MeanTeacherStrategy(SemisupStrategy):
    def __init__(self, cfg):
        self.cfg = cfg

    @torch.no_grad()
    def update_ema(self, student, teacher, momentum=0.999):
        for p_s, p_t in zip(student.parameters(), teacher.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1 - momentum)

    def training_step(self, module, batch_unsup: Dict[str, Any], global_step: int) -> Dict[str, Any]:
        if not batch_unsup:
            return {}
        weak = batch_unsup["weak"]
        strong = batch_unsup["strong"]

        teacher = getattr(module, "teacher", None)
        if teacher is None:
            return {}

        with torch.no_grad():
            preds_w = teacher(weak)
        preds_s = module(strong)

        B, T = preds_s["speed"].shape[:2]
        mask = torch.ones(B, T, dtype=torch.bool, device=strong.device)
        loss_hm = hm_consistency(preds_w["hm"], preds_s["hm"], mask)
        loss_sp = speed_consistency(preds_w["speed"], preds_s["speed"], mask)
        loss_vis = vislogit_consistency(preds_w["vis_state_logits"], preds_s["vis_state_logits"]).mean()
        w = float(self.cfg.semisup.get("consistency_weight", 0.5))
        loss = w * (loss_hm + loss_sp + loss_vis)

        module.log_dict(
            {
                "unsup/cons_hm": loss_hm,
                "unsup/cons_speed": loss_sp,
                "unsup/cons_vis": loss_vis,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        return {"loss_unsup": loss}
