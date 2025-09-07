from __future__ import annotations
from typing import List

import torch
import torch.nn.functional as F


def hm_consistency(pred_w: List[torch.Tensor], pred_s: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    total = 0.0
    count = 0
    for pw, ps in zip(pred_w, pred_s):
        diff = (pw - ps) ** 2  # [B,T,1,H,W]
        m = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        total = total + (diff * m).mean()
        count += 1
    return total / max(count, 1)


def speed_consistency(
    pred_w: torch.Tensor, pred_s: torch.Tensor, mask: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    diff = pred_w - pred_s
    loss = F.huber_loss(diff, torch.zeros_like(diff), delta=delta, reduction="none").sum(dim=-1)
    return (loss * mask.float()).mean()


def vislogit_consistency(logits_w: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
    pw = F.log_softmax(logits_w, dim=-1)
    ps = F.softmax(logits_s, dim=-1)
    return F.kl_div(pw, ps, reduction="batchmean")
