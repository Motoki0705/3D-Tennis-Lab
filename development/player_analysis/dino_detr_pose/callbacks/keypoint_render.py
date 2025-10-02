"""TensorBoard renderer for predicted vs. ground-truth keypoints (compatible with older torchvision)."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple
import inspect

import torch
from torchvision.utils import draw_keypoints

# -----------------------------------------------------------------------------
# Runtime feature check: older torchvision doesn't have `keypoints_mask`.
# -----------------------------------------------------------------------------
_SIG = inspect.signature(draw_keypoints)
_HAS_KEYPOINTS_MASK = "keypoints_mask" in _SIG.parameters

COCO_SKELETON: Sequence[Tuple[int, int]] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

# -----------------------------------------------------------------------------


def _to_uint8(image: torch.Tensor) -> torch.Tensor:
    """Normalize-to-[0,1] CHW tensor -> uint8 CHW."""
    img = image.detach().cpu()
    if img.dim() == 3 and img.size(0) in (1, 3):
        pass
    elif img.dim() == 3 and img.size(-1) in (1, 3):
        img = img.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")
    img = img.clamp(0, 1) * 255.0
    return img.to(torch.uint8)


def _filtered_connectivity(vis: torch.Tensor, skeleton: Iterable[Tuple[int, int]], k: int) -> list[Tuple[int, int]]:
    """Keep only edges whose endpoints are both visible and within range."""
    vis = vis.to(torch.bool)
    kept = []
    for i, j in skeleton:
        if 0 <= i < k and 0 <= j < k and vis[i] and vis[j]:
            kept.append((i, j))
    return kept


def _draw_instances(
    canvas: torch.Tensor,
    keypoints: torch.Tensor,  # (K, 2) in normalized xy [0..1]
    visibility: torch.Tensor,  # (K,) boolean/float>0
    *,
    color: str,
    skeleton: Iterable[Tuple[int, int]],
    radius: int = 3,
) -> torch.Tensor:
    if keypoints.numel() == 0:
        return canvas

    c, h, w = canvas.shape
    xy = keypoints.clone()
    xy[..., 0] *= float(w)
    xy[..., 1] *= float(h)

    vis = visibility > 0
    k = xy.shape[-2]

    if _HAS_KEYPOINTS_MASK:
        # Newer torchvision: use keypoints_mask directly
        conn = _filtered_connectivity(vis, skeleton, k)
        return draw_keypoints(
            canvas,
            xy.unsqueeze(0),  # (N=1, K, 2)
            connectivity=conn,
            colors=color,
            radius=radius,
            keypoints_mask=vis.unsqueeze(0),  # (N=1, K)
        )
    else:
        # Older torchvision: emulate masking
        conn = _filtered_connectivity(vis, skeleton, k)

        out = canvas
        # 1) draw only the lines between visible joints (suppress point markers with radius=0)
        if len(conn) > 0:
            out = draw_keypoints(
                out,
                xy.unsqueeze(0),
                connectivity=conn,
                colors=color,
                radius=0,  # attempt to hide point markers on this pass
            )

        # 2) draw point markers only for visible joints (no connectivity)
        if vis.any():
            xy_vis = xy[vis].unsqueeze(0)  # (1, K_vis, 2)
            out = draw_keypoints(
                out,
                xy_vis,
                colors=color,
                radius=radius,
            )
        return out


# -----------------------------------------------------------------------------


def render_pose_overlays(
    *,
    tag_overlay: str = "PoseOverlay",
    tag_pred: str = "PosePred",
    tag_gt: str = "PoseGT",
    image_key: str = "images",
    pred_keypoints_key: str = "pred_keypoints",
    pred_scores_key: str = "pred_scores",
    target_keypoints_key: str = "target_keypoints",
    visibility_key: str = "target_visibility",
    score_threshold: float = 0.3,
    visibility_threshold: float = 0.05,
    max_instances: int = 3,
    skeleton: Sequence[Tuple[int, int]] = COCO_SKELETON,
) -> Dict[str, torch.Tensor]:
    """Factory returning a pluggable logger renderer."""

    def _renderer(buffer: Dict[str, torch.Tensor], step: int, stage: str):
        images = buffer.get(image_key)
        pred_kp = buffer.get(pred_keypoints_key)
        pred_scores = buffer.get(pred_scores_key)
        gt_kp = buffer.get(target_keypoints_key)
        gt_vis = buffer.get(visibility_key)
        if images is None or pred_kp is None or gt_kp is None:
            return {}

        N = images.size(0)
        pred_imgs = []
        gt_imgs = []
        overlay_imgs = []

        for idx in range(N):
            base = _to_uint8(images[idx])
            canvas_pred = base.clone()
            canvas_gt = base.clone()
            canvas_overlay = base.clone()

            # Predictions: (M, K, 3) with (x, y, v)
            preds = pred_kp[idx]
            scores = pred_scores[idx] if pred_scores is not None else None
            num_pred = min(preds.size(0), max_instances)
            for det_idx in range(num_pred):
                if scores is not None and scores[det_idx].item() < score_threshold:
                    continue
                kp = preds[det_idx]  # (K, 3)
                vis = kp[:, 2] > visibility_threshold  # (K,)
                canvas_pred = _draw_instances(canvas_pred, kp[:, :2], vis, color="red", skeleton=skeleton)
                canvas_overlay = _draw_instances(canvas_overlay, kp[:, :2], vis, color="red", skeleton=skeleton)

            # Ground truth: (G, K, 3) with (x, y, v)
            gts = gt_kp[idx]
            vis_scores = gt_vis[idx] if gt_vis is not None else None
            num_gt = min(gts.size(0), max_instances)
            for person_idx in range(num_gt):
                if vis_scores is not None and vis_scores[person_idx].item() <= 0:
                    continue
                kp = gts[person_idx]  # (K, 3)
                vis = kp[:, 2] > visibility_threshold  # (K,)
                canvas_gt = _draw_instances(canvas_gt, kp[:, :2], vis, color="green", skeleton=skeleton)
                canvas_overlay = _draw_instances(canvas_overlay, kp[:, :2], vis, color="green", skeleton=skeleton)

            pred_imgs.append(canvas_pred.float() / 255.0)
            gt_imgs.append(canvas_gt.float() / 255.0)
            overlay_imgs.append(canvas_overlay.float() / 255.0)

        return {
            tag_pred: torch.stack(pred_imgs, dim=0),
            tag_gt: torch.stack(gt_imgs, dim=0),
            tag_overlay: torch.stack(overlay_imgs, dim=0),
        }

    return _renderer


__all__ = ["render_pose_overlays", "COCO_SKELETON"]
