from __future__ import annotations

from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from .architecture import DinoFpnHeatmapNet, DinoFpnModelConfig
from .backbone import DinoBackboneConfig
from .decoder import HeatmapDecoderConfig
from .lit_module import CourtPoseLitModule


def _to_dict(data: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(data, DictConfig):
        container = OmegaConf.to_container(data, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("Expected DictConfig to convert into dict.")
        return container
    return dict(data)


def create_model(*, backbone: Mapping[str, Any] | DictConfig, decoder: Mapping[str, Any] | DictConfig, **_: Any):
    """Instantiate the DINOv3+FPN heatmap network."""

    backbone_cfg = DinoBackboneConfig(**_to_dict(backbone))
    decoder_cfg = HeatmapDecoderConfig(**_to_dict(decoder))
    model_cfg = DinoFpnModelConfig(backbone=backbone_cfg, decoder=decoder_cfg)
    return DinoFpnHeatmapNet(model_cfg)


def create_lit_module(
    *,
    cfg,
    model,
    loss_fn,
    metric_fns: Mapping[str, Any] | None = None,
    target_key: str = "heatmaps",
    include_inputs: bool = True,
    **_: Any,
):
    """Instantiate LightningModule with logging-aware validation/test steps."""

    return CourtPoseLitModule(
        cfg,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        target_key=target_key,
        include_inputs=include_inputs,
    )


__all__ = ["create_model", "create_lit_module"]
