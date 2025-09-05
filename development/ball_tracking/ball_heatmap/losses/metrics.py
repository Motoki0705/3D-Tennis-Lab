from __future__ import annotations
from typing import List, Tuple

import torch
from ..trackers.features import soft_argmax_2d


@torch.no_grad()
def keypoint_accuracy(
    pred_hm: List[torch.Tensor],
    tgt_hm: List[torch.Tensor],
    mask: torch.Tensor,
    img_hw: Tuple[int, int],
    threshold_px: float = 5.0,
) -> torch.Tensor:
    """Compute detection accuracy on highest-resolution heatmap."""
    if not pred_hm or not tgt_hm:
        return torch.tensor(0.0)
    p0 = pred_hm[0]
    t0 = tgt_hm[0]
    B, T, _, Hs, Ws = p0.shape
    pred_xy_hm = soft_argmax_2d(p0)
    gt_xy_hm = soft_argmax_2d(t0)
    H, W = img_hw
    sx = float(W) / float(Ws)
    sy = float(H) / float(Hs)
    pred_xy_px = pred_xy_hm.clone()
    pred_xy_px[..., 0] = pred_xy_px[..., 0] * sx
    pred_xy_px[..., 1] = pred_xy_px[..., 1] * sy
    gt_xy_px = gt_xy_hm.clone()
    gt_xy_px[..., 0] = gt_xy_px[..., 0] * sx
    gt_xy_px[..., 1] = gt_xy_px[..., 1] * sy
    dist = torch.linalg.norm(pred_xy_px - gt_xy_px, dim=-1)
    valid = mask.bool()
    correct = (dist <= float(threshold_px)) & valid
    denom = valid.sum().clamp_min(1)
    acc = correct.sum().float() / denom.float()
    return acc
