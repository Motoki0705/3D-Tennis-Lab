"""Factory helpers for the DINO sequential model stack."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .dinov_3_vit_gru_fpn import NetConfig, SequenceHeatmapNet
from .lit_module import SequenceHeatmapLitModule

try:  # Prefer Hydra's path resolution when available.
    from hydra.utils import to_absolute_path as _to_absolute_path
except Exception:  # pragma: no cover - hydra not installed

    def _to_absolute_path(path: str) -> str:
        base = Path(os.environ.get("HYDRA_ORIG_CWD", Path.cwd()))
        return str((base / path).resolve())


def _resolve_path(path: str | os.PathLike[str]) -> str:
    return _to_absolute_path(str(path))


def create_model(**kwargs: Any) -> SequenceHeatmapNet:
    """Instantiate the sequential heatmap network with resolved paths."""

    params = dict(kwargs)
    params.setdefault("repo_dir", NetConfig.repo_dir)
    params.setdefault("weights", NetConfig.weights)

    params["repo_dir"] = _resolve_path(params["repo_dir"])
    params["weights"] = _resolve_path(params["weights"])

    cfg = NetConfig(**params)
    return SequenceHeatmapNet(cfg)


def create_lit_module(
    *,
    config: Mapping[str, Any],
    model: SequenceHeatmapNet,
    loss_fn,
    metric_fns: Mapping[str, Any] | None = None,
    target_key: str = "heatmaps",
) -> SequenceHeatmapLitModule:
    """Factory matching the signature expected by ``development.core.run``."""

    return SequenceHeatmapLitModule(
        config=config,
        model=model,
        loss_fn=loss_fn,
        metric_fns=dict(metric_fns or {}),
        target_key=target_key,
    )


__all__ = ["create_model", "create_lit_module"]
