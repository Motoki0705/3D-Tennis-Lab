from __future__ import annotations
import torch


def soft_argmax_2d(hm: torch.Tensor) -> torch.Tensor:
    """hm: [B,T,1,H,W] -> [B,T,2] in pixel coordinates (x,y)."""
    B, T, _, H, W = hm.shape
    flat = hm.view(B, T, -1)
    p = flat / (flat.sum(dim=-1, keepdim=True) + 1e-6)
    ys = torch.arange(H, device=hm.device, dtype=torch.float32).view(1, 1, H, 1)
    xs = torch.arange(W, device=hm.device, dtype=torch.float32).view(1, 1, 1, W)
    hm2 = hm.squeeze(2)
    px = (hm2 * xs).sum(dim=(-1, -2)) / (hm2.sum(dim=(-1, -2)) + 1e-6)
    py = (hm2 * ys).sum(dim=(-1, -2)) / (hm2.sum(dim=(-1, -2)) + 1e-6)
    return torch.stack([px, py], dim=-1)


def velocities_from_positions(pos: torch.Tensor) -> torch.Tensor:
    """pos: [B,T,2] -> vel [B,T,2] finite differences, last=prev."""
    vel = pos[:, 1:, :] - pos[:, :-1, :]
    vel = torch.cat([vel, vel[:, -1:, :]], dim=1)
    return vel
