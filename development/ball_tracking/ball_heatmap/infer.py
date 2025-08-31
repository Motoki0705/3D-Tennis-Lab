from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def argmax_with_offset(heatmap: torch.Tensor, offset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute subpixel coordinates from heatmap argmax plus offset.
    heatmap: [B,1,H,W]
    offset: [B,2,H,W], where offsets are fractional [dx,dy] relative to the argmax cell.
    Returns: coords_hm (B,2) in heatmap coordinates, scores (B,)
    """
    b, _, _h, w = heatmap.shape
    flat = heatmap.view(b, -1)
    idx = flat.argmax(dim=-1)  # [B]
    ys = (idx // w).float()
    xs = (idx % w).float()
    # Gather offsets at argmax locations
    dx = offset[:, 0].view(b, -1).gather(1, idx.view(-1, 1)).squeeze(1)
    dy = offset[:, 1].view(b, -1).gather(1, idx.view(-1, 1)).squeeze(1)
    coords = torch.stack([xs + dx, ys + dy], dim=-1)  # [B,2]
    scores = flat.gather(1, idx.view(-1, 1)).squeeze(1)
    return coords, scores


def upscale_coords(coords_hm: torch.Tensor, stride: int) -> torch.Tensor:
    return coords_hm * float(stride)


def predict_coords_multiscale(
    outputs: Dict[str, List[torch.Tensor]], strides: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Choose the highest-resolution output for coordinate prediction.
    outputs: dict with 'heatmaps' and 'offsets' lists ordered low->high resolution.
    strides: list of strides matching outputs order.
    Returns: coords in input pixel space [B,2] and scores [B]
    """
    heatmaps: List[torch.Tensor] = outputs["heatmaps"]
    offsets: List[torch.Tensor] = outputs["offsets"]
    # Use highest resolution (last)
    hmap = heatmaps[-1]
    offs = offsets[-1]
    stride = int(strides[-1])
    coords_hm, scores = argmax_with_offset(hmap, offs)
    coords_img = upscale_coords(coords_hm, stride)
    return coords_img, scores
