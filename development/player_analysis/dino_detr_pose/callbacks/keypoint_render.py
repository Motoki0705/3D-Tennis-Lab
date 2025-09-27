"""TensorBoard renderer for predicted vs. ground-truth keypoints."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
from torchvision.utils import draw_keypoints


COCO_SKELETON: Sequence[tuple[int, int]] = (
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


def _to_uint8(image: torch.Tensor) -> torch.Tensor:
    img = image.detach().cpu()
    if img.dim() == 3 and img.size(0) in (1, 3):
        pass
    elif img.dim() == 3 and img.size(-1) in (1, 3):
        img = img.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")
    img = img.clamp(0, 1) * 255.0
    return img.to(torch.uint8)


def _draw_instances(
    canvas: torch.Tensor,
    keypoints: torch.Tensor,
    visibility: torch.Tensor,
    *,
    color: str,
    skeleton: Iterable[tuple[int, int]],
    radius: int = 3,
) -> torch.Tensor:
    if keypoints.numel() == 0:
        return canvas
    c, h, w = canvas.shape
    keypoints = keypoints.clone()
    keypoints[..., 0] *= float(w)
    keypoints[..., 1] *= float(h)
    mask = visibility > 0
    keypoints = keypoints.unsqueeze(0)
    mask = mask.unsqueeze(0)
    return draw_keypoints(
        canvas,
        keypoints,
        connectivity=list(skeleton),
        colors=color,
        radius=radius,
        keypoints_mask=mask,
    )


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
    skeleton: Sequence[tuple[int, int]] = COCO_SKELETON,
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

            preds = pred_kp[idx]
            scores = pred_scores[idx] if pred_scores is not None else None
            count = min(preds.size(0), max_instances)
            for det_idx in range(count):
                if scores is not None and scores[det_idx].item() < score_threshold:
                    continue
                kp = preds[det_idx]
                vis = kp[:, 2] > visibility_threshold
                canvas_pred = _draw_instances(canvas_pred, kp[:, :2], vis, color="red", skeleton=skeleton)
                canvas_overlay = _draw_instances(canvas_overlay, kp[:, :2], vis, color="red", skeleton=skeleton)

            gts = gt_kp[idx]
            vis_scores = gt_vis[idx] if gt_vis is not None else None
            count_gt = min(gts.size(0), max_instances)
            for person_idx in range(count_gt):
                kp = gts[person_idx]
                vis = kp[:, 2] > visibility_threshold
                if vis_scores is not None and vis_scores[person_idx].item() <= 0:
                    continue
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
