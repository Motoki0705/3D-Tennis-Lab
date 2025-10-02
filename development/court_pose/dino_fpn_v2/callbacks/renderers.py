from __future__ import annotations

from typing import Callable, Dict

import torch

from development.core.callbacks.plaggable_logger import (
    RenderDict,
    render_heatmaps_dict,
    render_overlay_max_dict,
)


def build_court_heatmap_renderer(
    *,
    upsample_to: int = 448,
    overlay_alpha: float = 0.5,
    include_inputs: bool = True,
    input_key: str = "images",
    pred_key: str = "pred_heatmaps",
    gt_key: str = "target_heatmaps",
) -> Callable[[Dict[str, torch.Tensor], int, str], RenderDict]:
    """Renderer that logs per-keypoint heatmaps plus overlay visuals."""

    heatmaps_renderer = render_heatmaps_dict(
        upsample_to=upsample_to,
        include_inputs=include_inputs,
        input_key=input_key,
        pred_key=pred_key,
        gt_key=gt_key,
        pred_category="Pred",
        gt_category="GT",
        input_category="Input",
    )
    pred_overlay_renderer = render_overlay_max_dict(
        upsample_to=upsample_to,
        alpha=overlay_alpha,
        input_key=input_key,
        pred_key=pred_key,
        overlay_category="PredOverlay",
    )
    gt_overlay_renderer = render_overlay_max_dict(
        upsample_to=upsample_to,
        alpha=overlay_alpha,
        input_key=input_key,
        pred_key=gt_key,
        overlay_category="GTOverlay",
    )

    def _renderer(buf: Dict[str, torch.Tensor], step: int, stage: str) -> RenderDict:
        out: RenderDict = {}
        heatmap_dict = heatmaps_renderer(buf, step, stage)
        if isinstance(heatmap_dict, dict):
            out.update(heatmap_dict)

        pred_overlay = pred_overlay_renderer(buf, step, stage)
        if isinstance(pred_overlay, dict):
            out.update(pred_overlay)

        gt_overlay = gt_overlay_renderer(buf, step, stage)
        if isinstance(gt_overlay, dict):
            out.update(gt_overlay)

        return out

    return _renderer


__all__ = ["build_court_heatmap_renderer"]
