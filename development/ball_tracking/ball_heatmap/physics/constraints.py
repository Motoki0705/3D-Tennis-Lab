from __future__ import annotations
import torch


def physics_score(pos: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """Return per-frame normalized score in [0,1].
    pos: [B,T,2] pixel positions (could be soft-argmax of HM)
    Simple bounds check as a stub: inside frame -> 1, else 0.
    """
    x, y = pos[..., 0], pos[..., 1]
    inside = (x >= 0) & (x < img_w) & (y >= 0) & (y < img_h)
    return inside.float()
