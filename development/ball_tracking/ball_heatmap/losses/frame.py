from __future__ import annotations
from typing import List

import torch
import torch.nn.functional as F


def heatmap_mse(pred_hm: List[torch.Tensor], tgt_hm: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    """pred_hm/tgt_hm: list per scale of [B,T,1,H,W]; mask: [B,T]"""
    total = 0.0
    count = 0
    for p, t in zip(pred_hm, tgt_hm):
        p = p.float()
        t = t.float()
        B, T, _, H, W = p.shape
        p = p.view(B, T, -1)
        t = t.view(B, T, -1)
        loss = F.mse_loss(p, t, reduction="none")
        m = mask.unsqueeze(-1).float()
        total = total + (loss * m).mean()
        count += 1
    return total / max(count, 1)


def heatmap_focal(
    pred_hm: List[torch.Tensor],
    tgt_hm: List[torch.Tensor],
    mask: torch.Tensor,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Binary focal loss over heatmaps with soft targets."""
    total = 0.0
    count = 0
    for p, t in zip(pred_hm, tgt_hm):
        p = p.float()
        t = t.float()
        B, T, _, H, W = p.shape
        p = torch.sigmoid(p)
        p = p.view(B, T, -1)
        t = t.view(B, T, -1).clamp(0.0, 1.0)
        loss_pos = -(t * ((1 - p).clamp_min(eps) ** gamma) * torch.log(p.clamp_min(eps)))
        loss_neg = -((1 - t) * (p.clamp_min(eps) ** gamma) * torch.log((1 - p).clamp_min(eps)))
        loss = (loss_pos + loss_neg).mean(dim=-1)
        total = total + (loss * mask.float()).mean()
        count += 1
    return total / max(count, 1)


def heatmap_kl(
    pred_hm: List[torch.Tensor],
    tgt_hm: List[torch.Tensor],
    mask: torch.Tensor,
    tau: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL divergence between spatial distributions of pred and target heatmaps."""
    total = 0.0
    count = 0
    for p, t in zip(pred_hm, tgt_hm):
        p = p.float()
        t = t.float()
        B, T, _, H, W = p.shape
        p = p.view(B, T, -1) / max(tau, eps)
        t = t.view(B, T, -1) / max(tau, eps)
        logp = F.log_softmax(p, dim=-1)
        q = F.softmax(t, dim=-1)
        kl = F.kl_div(logp, q, reduction="none").sum(dim=-1)
        total = total + (kl * mask.float()).mean()
        count += 1
    return total / max(count, 1)


def heatmap_loss(
    pred_hm: List[torch.Tensor],
    tgt_hm: List[torch.Tensor],
    mask: torch.Tensor,
    kind: str = "mse",
    focal_gamma: float = 2.0,
    kl_tau: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    kind = (kind or "mse").lower()
    if kind == "mse":
        return heatmap_mse(pred_hm, tgt_hm, mask)
    if kind == "focal":
        return heatmap_focal(pred_hm, tgt_hm, mask, gamma=focal_gamma, eps=eps)
    if kind in ("kl", "kld", "kl_div"):
        return heatmap_kl(pred_hm, tgt_hm, mask, tau=kl_tau, eps=eps)
    raise ValueError(f"Unknown heatmap loss kind: {kind}")


def speed_huber(
    pred_speed: torch.Tensor, tgt_speed: torch.Tensor, mask: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """pred_speed/tgt_speed: [B,T,2]; mask: [B,T]"""
    diff = pred_speed.float() - tgt_speed.float()
    loss = F.huber_loss(diff, torch.zeros_like(diff), delta=delta, reduction="none")
    loss = loss.sum(dim=-1)
    loss = (loss * mask.float()).mean()
    return loss


def vis_ce(logits: torch.Tensor, vis_state: torch.Tensor) -> torch.Tensor:
    """logits: [B,T,3]; vis_state: [B,T] Long"""
    B, T, C = logits.shape
    return F.cross_entropy(logits.float().view(B * T, C), vis_state.view(B * T), reduction="mean")
