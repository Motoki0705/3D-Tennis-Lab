from __future__ import annotations

from typing import Dict, Tuple, List

import logging

import numpy as np
import cv2

from ..base import Bundle, Stage
from ..base import register


def _gather_matches(court3d: Dict[str, np.ndarray], keypoints: Dict[str, Tuple[float, float, int, float]]):
    X_plane: List[List[float]] = []
    U: List[List[float]] = []
    for name, xyz in court3d.items():
        if name not in keypoints:
            continue

        u, v, vis, w = keypoints[name]
        if vis and w > 0:
            X_plane.append([float(xyz[0]), float(xyz[1])])
            U.append([float(u), float(v)])
    if len(U) < 4:
        logging.getLogger(__name__).warning(
            "InitHomography: need >=4 visible weighted points, got %d; skipping homography.",
            len(U),
        )

    X_plane_np = np.asarray(X_plane, dtype=np.float64)
    U_np = np.asarray(U, dtype=np.float64)
    return X_plane_np, U_np


@register("s10_init_homography")
class InitHomography(Stage):
    required_inputs = ["frames", "court3d", "min_visible_4", "ref_integrity"]
    produces = ["H", "report.homography"]
    STAGE_VERSION = "1.0.0"

    def run(self, B: Bundle) -> Bundle:
        if not B.frames:
            raise ValueError("InitHomography: Bundle.frames is empty")
        kp = B.frames[0].keypoints
        X_plane, U = _gather_matches(B.court3d, kp)

        ransac = self.cfg.get("ransac", {})
        thr = float(ransac.get("reproj_px", 2.0))
        conf = float(ransac.get("conf", 0.999))
        iters = int(ransac.get("iters", 2000))

        # If insufficient correspondences, log/report and skip without raising
        if U.shape[0] < 4:
            B.H = None
            total = int(U.shape[0])
            B.report.setdefault("homography", {})
            B.report["homography"].update({
                "inliers": 0,
                "total": total,
                "ratio": 0.0,
                "reproj_px": thr,
                "success": False,
                "reason": f"need >=4 visible weighted points, got {total}",
            })
            # Also record a user-facing warning entry
            B.report.setdefault("warnings", {})
            B.report["warnings"].setdefault("homography", []).append(
                f"InitHomography: insufficient correspondences (<4). Got {total}; rejected and skipped."
            )
            return B

        H, mask = cv2.findHomography(
            X_plane,
            U,
            method=cv2.RANSAC,
            ransacReprojThreshold=thr,
            confidence=conf,
            maxIters=iters,
        )
        if H is not None:
            H = np.asarray(H, dtype=np.float64)
        B.H = H
        inliers = int(mask.sum()) if mask is not None else 0
        total = int(len(U))
        ratio = float(inliers / total) if total > 0 else 0.0
        B.report.setdefault("homography", {})
        B.report["homography"].update({
            "inliers": inliers,
            "total": total,
            "ratio": ratio,
            "reproj_px": thr,
            "success": bool(H is not None),
        })
        return B
