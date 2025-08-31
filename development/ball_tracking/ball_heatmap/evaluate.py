from __future__ import annotations

import torch


def pck(pred: torch.Tensor, tgt: torch.Tensor, threshold_ratio: float = 0.05) -> torch.Tensor:
    # pred/tgt: [B,1,H,W]
    b, _, h, w = pred.shape
    flat_p = pred.view(b, -1)
    flat_t = tgt.view(b, -1)
    ip = flat_p.argmax(dim=-1)
    it = flat_t.argmax(dim=-1)
    yp = (ip // w).float()
    xp = (ip % w).float()
    yt = (it // w).float()
    xt = (it % w).float()
    d = torch.sqrt((xp - xt) ** 2 + (yp - yt) ** 2)
    thr = threshold_ratio * ((h**2 + w**2) ** 0.5)
    return (d < thr).float().mean()
