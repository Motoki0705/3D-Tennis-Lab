from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import torch


@dataclass
class FeatureStats:
    mean: torch.Tensor  # (D,)
    std: torch.Tensor  # (D,)

    def to(self, device):
        return FeatureStats(self.mean.to(device), self.std.to(device))


def compute_feature_stats(samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> FeatureStats:
    # samples: iterable of (seq[T, D], target[D])
    total = 0
    sum_vec = None
    sum_sq = None
    for seq, tgt in samples:
        # concatenate along time: include target as one more timestep
        seq2 = torch.cat([seq, tgt.unsqueeze(0)], dim=0)  # (T+1, D)
        if sum_vec is None:
            D = seq2.shape[-1]
            sum_vec = torch.zeros(D, dtype=torch.float64)
            sum_sq = torch.zeros(D, dtype=torch.float64)
        sum_vec += seq2.double().sum(dim=0)
        sum_sq += (seq2.double().pow(2)).sum(dim=0)
        total += seq2.size(0)
    mean = (sum_vec / max(total, 1)).float()
    var = (sum_sq / max(total, 1)).float() - mean.pow(2)
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    return FeatureStats(mean=mean, std=std)


class Standardizer(torch.nn.Module):
    def __init__(self, stats: Optional[FeatureStats]):
        super().__init__()
        self.has_stats = stats is not None
        if self.has_stats:
            self.register_buffer("mean", stats.mean)
            self.register_buffer("std", stats.std)
        else:
            self.register_buffer("mean", torch.tensor([]))
            self.register_buffer("std", torch.tensor([]))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_stats:
            return x
        return (x - self.mean) / (self.std + 1e-6)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_stats:
            return x
        return x * (self.std + 1e-6) + self.mean
