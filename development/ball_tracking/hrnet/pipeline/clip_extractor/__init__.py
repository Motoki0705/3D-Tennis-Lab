from __future__ import annotations

from omegaconf import DictConfig

from .base import Clip, ClipExtractor
from .clustering import ClusteringClipExtractor, ClusteringConfig
from .contiguous import ContiguousClipExtractor, ContiguousConfig

__all__ = ["build_clip_extractor", "Clip", "ClipExtractor"]


def build_clip_extractor(cfg: DictConfig) -> ClipExtractor:
    strategy = cfg.strategy
    if strategy == "contiguous":
        params = cfg.contiguous
        return ContiguousClipExtractor(
            ContiguousConfig(
                max_gap_frames=int(params.max_gap_frames),
                min_clip_len=int(params.min_clip_len),
            )
        )
    if strategy == "clustering":
        params = cfg.clustering
        return ClusteringClipExtractor(
            ClusteringConfig(
                algo=str(params.algo),
                features=list(params.features),
                eps=float(params.get("eps", 0.8)) if params.get("eps") is not None else None,
                min_samples=int(params.get("min_samples", 5)) if params.get("min_samples") is not None else None,
                min_clip_len=int(params.min_clip_len),
            )
        )
    raise ValueError(f"Unknown clip_extractor strategy: {strategy}")
