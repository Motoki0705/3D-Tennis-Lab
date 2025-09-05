from __future__ import annotations
from typing import Any, Dict, List

import torch

from .base import SemisupStrategy


def heatmap_peak_ratio(hm: torch.Tensor) -> torch.Tensor:
    """hm: [B,T,1,H,W] -> per-frame peak ratio score in [0,1]."""
    B, T, _, H, W = hm.shape
    flat = hm.view(B, T, -1)
    top2 = torch.topk(flat, k=2, dim=-1).values  # [B,T,2]
    ratio = top2[..., 0] / (top2[..., 1] + 1e-6)
    return torch.clamp(ratio / (ratio.max(dim=1, keepdim=True).values + 1e-6), 0, 1)


def heatmap_entropy(hm: torch.Tensor) -> torch.Tensor:
    B, T, _, H, W = hm.shape
    flat = hm.view(B, T, -1)
    p = flat / (flat.sum(dim=-1, keepdim=True) + 1e-6)
    ent = -(p * (p + 1e-6).log()).sum(dim=-1)
    ent_norm = 1 - ent / (ent.max(dim=1, keepdim=True).values + 1e-6)
    return ent_norm


class SelfTrainStrategy(SemisupStrategy):
    def __init__(self, cfg):
        self.cfg = cfg
        self.weights = cfg.semisup.pseudo.q_weights

        # Teacher is expected to be provided by module (either frozen or EMA)

    def _quality(
        self, hm_list: List[torch.Tensor], physics: torch.Tensor | None = None, disc: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Use the highest-resolution heatmap for Q
        hm = hm_list[0]
        peak = heatmap_peak_ratio(hm)
        ent = heatmap_entropy(hm)
        q = self.weights.peak * peak + self.weights.entropy * ent
        if physics is not None:
            q = q + self.weights.physics * physics
        if disc is not None:
            q = q + self.weights.disc * disc
        return torch.clamp(q, 0.0, 1.0)  # [B,T]

    def _tau(self, current_epoch: int) -> float:
        st = float(self.cfg.semisup.pseudo.tau_start)
        ed = float(self.cfg.semisup.pseudo.tau_end)
        w = int(self.cfg.semisup.pseudo.warmup_epochs)
        if current_epoch >= w:
            return ed
        alpha = current_epoch / max(w, 1)
        return st * (1 - alpha) + ed * alpha

    @torch.no_grad()
    def training_step(self, module, batch_unsup: Dict[str, Any], global_step: int) -> Dict[str, Any]:
        if not batch_unsup:
            return {}
        weak = batch_unsup["weak"]  # [B,T,3,320,640]
        strong = batch_unsup["strong"]
        B, T, C, H, W = weak.shape

        # Expect module.teacher to be set (frozen or EMA)
        teacher = getattr(module, "teacher", None)
        if teacher is None:
            return {}

        weak_btc = weak  # already [B,T,C,H,W]
        preds_t = teacher(weak_btc)
        # Build pseudo-targets
        hm_pseudo: List[torch.Tensor] = [p.detach() for p in preds_t["heatmaps"]]
        speed_pseudo: torch.Tensor = preds_t["speed"].detach()
        vis_state_pseudo = preds_t["vis_logits"].detach().argmax(dim=-1)

        # Quality per-frame
        physics_score = None  # module.physics_score if available
        disc_score = None  # module.disc_score if available
        q = self._quality(hm_pseudo, physics_score, disc_score)  # [B,T]
        tau = self._tau(module.current_epoch)
        mask = q >= tau

        # Student predictions on strong
        preds_s = module.forward(strong)

        # Compute losses analogous to supervised using masks
        losses = module.compute_semisup_losses(preds_s, hm_pseudo, speed_pseudo, vis_state_pseudo, mask)
        # Log acceptance
        module.log("unsup/q_tau", tau, prog_bar=False, on_step=True, on_epoch=False)
        module.log("unsup/q_accept_rate", mask.float().mean(), prog_bar=True, on_step=True, on_epoch=True)
        return losses
