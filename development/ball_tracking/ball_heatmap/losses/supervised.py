from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn.functional as F
from development.ball_tracking.ball_heatmap.trackers.features import soft_argmax_2d


def heatmap_mse(pred_hm: List[torch.Tensor], tgt_hm: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    """pred_hm/tgt_hm: list per scale of [B,T,1,H,W]; mask: [B,T]"""
    total = 0.0
    count = 0
    for p, t in zip(pred_hm, tgt_hm):
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
    """Binary focal loss over heatmaps with soft targets.
    pred_hm/tgt_hm: list per scale of [B,T,1,H,W]; mask: [B,T]
    Applies sigmoid to predictions and computes focal loss against [0,1] targets.
    """
    total = 0.0
    count = 0
    for p, t in zip(pred_hm, tgt_hm):
        B, T, _, H, W = p.shape
        p = torch.sigmoid(p)
        # flatten per-frame
        p = p.view(B, T, -1)
        t = t.view(B, T, -1).clamp(0.0, 1.0)
        # focal: -[ y*(1-p)^g*log(p) + (1-y)*p^g*log(1-p) ]
        loss_pos = -(t * ((1 - p).clamp_min(eps) ** gamma) * torch.log(p.clamp_min(eps)))
        loss_neg = -((1 - t) * (p.clamp_min(eps) ** gamma) * torch.log((1 - p).clamp_min(eps)))
        loss = (loss_pos + loss_neg).mean(dim=-1)  # [B,T]
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
    """KL divergence between spatial distributions of pred and target heatmaps.
    Use softmax over spatial dims. pred: log_softmax, target: softmax.
    pred_hm/tgt_hm: list per scale of [B,T,1,H,W]; mask: [B,T]
    """
    total = 0.0
    count = 0
    for p, t in zip(pred_hm, tgt_hm):
        B, T, _, H, W = p.shape
        p = p.view(B, T, -1) / max(tau, eps)
        t = t.view(B, T, -1) / max(tau, eps)
        logp = F.log_softmax(p, dim=-1)
        q = F.softmax(t, dim=-1)
        # KL(p || q) with logp input and q target
        kl = F.kl_div(logp, q, reduction="none").sum(dim=-1)  # [B,T]
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


@torch.no_grad()
def keypoint_accuracy(
    pred_hm: List[torch.Tensor],
    tgt_hm: List[torch.Tensor],
    mask: torch.Tensor,
    img_hw: Tuple[int, int],  # (H, W)
    threshold_px: float = 5.0,
) -> torch.Tensor:
    """Compute detection accuracy on highest-resolution heatmap.
    Correct if L2 distance between predicted and GT positions (in pixels) < threshold_px.
    - pred_hm/tgt_hm: list per scale of [B,T,1,Hs,Ws]
    - mask: [B,T] boolean for valid frames
    - img_hw: original image size (after letterbox), to convert hm coords to pixels
    """
    if not pred_hm or not tgt_hm:
        return torch.tensor(0.0)
    p0 = pred_hm[0]
    t0 = tgt_hm[0]
    B, T, _, Hs, Ws = p0.shape
    # coords on heatmap grid
    pred_xy_hm = soft_argmax_2d(p0)  # [B,T,2]
    gt_xy_hm = soft_argmax_2d(t0)
    H, W = img_hw
    # scale to pixel space using anisotropic factors
    sx = float(W) / float(Ws)
    sy = float(H) / float(Hs)
    pred_xy_px = pred_xy_hm.clone()
    pred_xy_px[..., 0] = pred_xy_px[..., 0] * sx
    pred_xy_px[..., 1] = pred_xy_px[..., 1] * sy
    gt_xy_px = gt_xy_hm.clone()
    gt_xy_px[..., 0] = gt_xy_px[..., 0] * sx
    gt_xy_px[..., 1] = gt_xy_px[..., 1] * sy

    dist = torch.linalg.norm(pred_xy_px - gt_xy_px, dim=-1)  # [B,T]
    valid = mask.bool()
    correct = (dist <= float(threshold_px)) & valid
    denom = valid.sum().clamp_min(1)
    acc = correct.sum().float() / denom.float()
    return acc


def speed_huber(
    pred_speed: torch.Tensor, tgt_speed: torch.Tensor, mask: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """pred_speed/tgt_speed: [B,T,2]; mask: [B,T]"""
    diff = pred_speed - tgt_speed
    loss = F.huber_loss(diff, torch.zeros_like(diff), delta=delta, reduction="none")
    loss = loss.sum(dim=-1)  # [B,T]
    loss = (loss * mask.float()).mean()
    return loss


def vis_ce(logits: torch.Tensor, vis_state: torch.Tensor) -> torch.Tensor:
    """logits: [B,T,3]; vis_state: [B,T] Long"""
    B, T, C = logits.shape
    return F.cross_entropy(logits.view(B * T, C), vis_state.view(B * T), reduction="mean")
