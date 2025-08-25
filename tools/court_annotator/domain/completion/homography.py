# Step 4.2: Homography-based completion strategy
import numpy as np
import logging
from ...core.geometry import estimate_homography, project_H, rmse_inliers
from ...core.types import FitResult, Keypoint
from .base import CompletionStrategy


def _collect_user_points(kps: list[Keypoint]):
    """Collects points manually placed by the user."""
    src_idx = []
    dst_xy = []
    for i, kp in enumerate(kps):
        if kp.skip:
            continue
        if kp.source == "user" and kp.v > 0 and kp.x is not None and kp.y is not None:
            src_idx.append(i)
            dst_xy.append((kp.x, kp.y))
    return src_idx, np.asarray(dst_xy, np.float32)


class HomographyCompletion(CompletionStrategy):
    """Implements the completion logic using homography."""

    def __init__(self, template_xy):
        self.template_xy = np.asarray(template_xy, np.float32)

    def fit_and_complete(self, frame_state, court_spec, params) -> FitResult:
        idx, dst = _collect_user_points(frame_state.kps)
        if len(idx) < params.get("min_used_points", 4):
            # Downgrade to debug to avoid flooding logs during interactive use
            logging.debug("Homography: insufficient user points for estimation")
            return FitResult(H=None, used=0, rmse=None, inlier_idx=[])

        src = self.template_xy[idx]
        H, mask = estimate_homography(src, dst, params.get("ransac_reproj_thresh", 2.0))

        if H is None or mask is None or not mask.any():
            logging.warning("Homography estimation failed or no inliers")
            return FitResult(H=None, used=0, rmse=None, inlier_idx=[])

        used_idx = [i for i, m in zip(idx, mask) if m]
        rmse = rmse_inliers(H, src, dst, mask)

        # Autocomplete logic (preserve user-entered points)
        proj = project_H(H, self.template_xy)
        for i in range(len(frame_state.kps)):
            kp = frame_state.kps[i]
            if kp.skip or kp.locked:
                continue
            if kp.v == 0 or kp.source == "auto":
                px, py = float(proj[i, 0]), float(proj[i, 1])
                kp.x, kp.y = px, py
                if kp.v == 0:
                    kp.v = 2  # Mark as visible
                kp.source = "auto"

        return FitResult(H=H.tolist(), used=int(mask.sum()), rmse=rmse, inlier_idx=used_idx)
