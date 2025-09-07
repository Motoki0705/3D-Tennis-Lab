from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """Base temporal block expecting x: [B,T,C,H,W] -> [B,T,C,H,W]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
