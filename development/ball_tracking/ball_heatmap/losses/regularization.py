from __future__ import annotations

import torch
import torch.nn.functional as F


def tv_loss(coords: torch.Tensor, mask: torch.Tensor, order: int = 1, p: int = 1) -> torch.Tensor:
    """Total variation loss on coordinates."""
    if order < 1:
        return torch.tensor(0.0, device=coords.device)
    diff = coords.float()
    for _ in range(order):
        diff = torch.diff(diff, dim=1)
    loss = torch.linalg.norm(diff, ord=p, dim=-1)
    valid_mask = mask.float()
    for i in range(order):
        valid_mask = valid_mask * torch.roll(mask, shifts=-i - 1, dims=1).float()
    valid_mask = valid_mask[:, : loss.shape[1]]
    return (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


def ranking_margin_loss(
    pred_hm: torch.Tensor,
    tgt_hm: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.1,
    neighborhood_size: int = 5,
) -> torch.Tensor:
    """Hinge loss on the margin between the value at the GT location and the 2nd peak."""
    pred_hm = pred_hm.float()
    tgt_hm = tgt_hm.float()
    B, T, _, H, W = pred_hm.shape
    hm_flat = pred_hm.view(B, T, -1)
    tgt_flat = tgt_hm.view(B, T, -1)
    gt_val = (hm_flat * tgt_flat).sum(dim=-1)
    tgt_coords = torch.argmax(tgt_flat, dim=-1)
    y_gt, x_gt = tgt_coords // W, tgt_coords % W
    x_grid, y_grid = torch.meshgrid(
        torch.arange(W, device=pred_hm.device), torch.arange(H, device=pred_hm.device), indexing="xy"
    )
    dist_sq = (x_grid.float() - x_gt.unsqueeze(-1).unsqueeze(-1)) ** 2 + (
        y_grid.float() - y_gt.unsqueeze(-1).unsqueeze(-1)
    ) ** 2
    outside_neighborhood = (dist_sq > neighborhood_size**2).view(B, T, H * W)
    second_peak_val = (hm_flat * outside_neighborhood).max(dim=-1).values
    loss = F.relu(margin - (gt_val - second_peak_val))
    return (loss * mask.float()).mean()


def peak_sharpening_loss(pred_hm: torch.Tensor, mask: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Negative entropy loss to encourage peaky distributions."""
    pred_hm = pred_hm.float()
    B, T, _, H, W = pred_hm.shape
    hm_flat = pred_hm.view(B, T, -1)
    p = F.softmax(hm_flat / temperature, dim=-1)
    log_p = F.log_softmax(hm_flat / temperature, dim=-1)
    neg_entropy = (p * log_p).sum(dim=-1)
    return (neg_entropy * mask.float()).mean()


def peak_lower_bound_loss(pred_hm: torch.Tensor, mask: torch.Tensor, min_peak_val: float = 0.1) -> torch.Tensor:
    """Loss to prevent the heatmap from collapsing."""
    pred_hm = pred_hm.float()
    B, T, _, H, W = pred_hm.shape
    max_per_frame = pred_hm.view(B, T, -1).max(dim=-1).values
    loss = F.relu(min_peak_val - max_per_frame)
    return (loss * mask.float()).mean()
