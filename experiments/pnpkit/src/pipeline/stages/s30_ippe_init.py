from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import cv2

from ..base import Bundle, Stage
from ..base import register
from ...core.camera import Intrinsics, PoseCW
from ...core.geometry import project_points_W


def _gather_matches3d2d(court3d: Dict[str, np.ndarray], keypoints: Dict[str, Tuple[float, float, int, float]]):
    Xw: List[List[float]] = []
    U: List[List[float]] = []
    for name, xyz in court3d.items():
        if name not in keypoints:
            continue
        u, v, vis, w = keypoints[name]
        if vis and w > 0:
            Xw.append([float(xyz[0]), float(xyz[1]), 0.0])
            U.append([float(u), float(v)])
    if len(U) < 4:
        raise ValueError(f"IPPEInit: need >=4 visible weighted points, got {len(U)}")
    Xw = np.asarray(Xw, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    return Xw, U


@register("s30_ippe_init")
class IPPEInit(Stage):
    required_inputs = ["frames", "court3d", "K", "min_visible_4", "ref_integrity"]
    produces = ["pose_cw", "poses", "report.ippe"]
    STAGE_VERSION = "1.0.0"

    def run(self, B: Bundle) -> Bundle:
        if not B.frames:
            raise ValueError("IPPEInit: Bundle.frames is empty")
        if not B.K:
            raise ValueError("IPPEInit: Bundle.K is missing (need fx, fy, cx, cy, optional dist)")
        Kd = B.K
        fx, fy, cx, cy = Kd.get("fx"), Kd.get("fy"), Kd.get("cx"), Kd.get("cy")
        if None in (fx, fy, cx, cy):
            raise ValueError("IPPEInit: K must include fx, fy, cx, cy")
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = None
        if "dist" in Kd and Kd["dist"] is not None:
            d = Kd["dist"]
            dist = np.array(
                [
                    float(d.get("k1", 0.0)),
                    float(d.get("k2", 0.0)),
                    float(d.get("p1", 0.0)),
                    float(d.get("p2", 0.0)),
                    float(d.get("k3", 0.0)),
                ],
                dtype=np.float64,
            )

        intr = Intrinsics(
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), skew=float(Kd.get("skew", 0.0) or 0.0)
        )
        if "dist" in Kd and Kd["dist"] is not None:
            d = Kd["dist"]
            intr.dist.k1 = float(d.get("k1", 0.0))
            intr.dist.k2 = float(d.get("k2", 0.0))
            intr.dist.p1 = float(d.get("p1", 0.0))
            intr.dist.p2 = float(d.get("p2", 0.0))
            intr.dist.k3 = float(d.get("k3", 0.0))

        poses = []
        best_rmse_first = None
        num_candidates_first = 0
        for idx, fr in enumerate(B.frames):
            kp = fr.keypoints
            Xw, uv = _gather_matches3d2d(B.court3d, kp)
            ok, rvecs, tvecs, _reprojErrs = cv2.solvePnPGeneric(
                Xw,
                uv,
                K,
                dist,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if not ok or rvecs is None or tvecs is None or len(rvecs) == 0:
                raise ValueError(f"IPPEInit: solvePnPGeneric failed for frame {fr.frame_idx}")
            best_rmse = float("inf")
            best_R = None
            best_t = None
            for rvec, tvec in zip(rvecs, tvecs):
                R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
                t = np.asarray(tvec, dtype=np.float64).reshape(3)
                pose = PoseCW(R=R, t=t)
                uv_hat = project_points_W(Xw, intr, pose)
                rmse = float(np.sqrt(np.mean(np.sum((uv_hat - uv) ** 2, axis=1))))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_R = R
                    best_t = t
            poses.append({"frame_idx": fr.frame_idx, "R": best_R, "t": best_t})
            if idx == 0:
                best_rmse_first = best_rmse
                num_candidates_first = len(rvecs)

        # Set first pose_cw for compatibility and store all poses
        B.pose_cw = {"R": poses[0]["R"], "t": poses[0]["t"], "method": "IPPE"}
        B.poses = poses
        B.report.setdefault("ippe", {})
        B.report["ippe"].update({"num_candidates": int(num_candidates_first), "rmse_px": float(best_rmse_first)})
        return B
