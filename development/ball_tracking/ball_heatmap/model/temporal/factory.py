from __future__ import annotations

from typing import Dict

from .base import TemporalBlock
from .conv1d import TemporalConv1D
from .convgru import TemporalConvGRU
from .tsm import TemporalShift


def build_temporal_block(cfg: Dict, channels: int) -> TemporalBlock:
    t = str(cfg.get("type", "conv1d")).lower()
    if t == "conv1d":
        return TemporalConv1D(
            channels=channels,
            kernel_t=int(cfg.get("kernel_t", 3)),
            causal=bool(cfg.get("causal", False)),
            expand=int(cfg.get("expand", 1)),
        )
    if t == "convgru":
        return TemporalConvGRU(
            channels=channels,
            kernel_t=int(cfg.get("kernel_t", 3)),
        )
    if t == "tsm":
        return TemporalShift(
            channels=channels,
            shift_ratio=float(cfg.get("shift_ratio", 1 / 8)),
        )
    raise ValueError(f"Unknown temporal type: {t}")
