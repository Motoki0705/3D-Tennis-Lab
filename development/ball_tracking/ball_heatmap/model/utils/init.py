from __future__ import annotations

import torch.nn as nn


def trunc_normal_(module: nn.Module, std: float = 0.02):
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.trunc_normal_(p, std=std)
