"""Loss wrapper for sequential heatmap regression."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapBCELoss(nn.Module):
    def __init__(self, *, reduction: str = "mean", target_key: str = "heatmaps") -> None:
        super().__init__()
        self.reduction = reduction
        self.target_key = target_key

    def forward(self, preds: torch.Tensor, targets: Any) -> torch.Tensor:
        target_tensor = self._resolve_targets(targets)
        return F.binary_cross_entropy_with_logits(preds, target_tensor, reduction=self.reduction)

    def _resolve_targets(self, targets: Any) -> torch.Tensor:
        if torch.is_tensor(targets):
            return targets
        if isinstance(targets, Mapping):
            value = targets.get(self.target_key)
            if torch.is_tensor(value):
                return value
        raise TypeError("HeatmapBCELoss expects targets to be a tensor or mapping containing " f"'{self.target_key}'.")


def build_loss(**kwargs: Any) -> HeatmapBCELoss:
    return HeatmapBCELoss(**kwargs)


__all__ = ["HeatmapBCELoss", "build_loss"]
