from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np

from ..base import Bundle, Stage
from ..base import register
from ...core.camera import Intrinsics, PoseCW
from ...core.geometry import project_points_W, invert_pose


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
    return np.asarray(Xw, dtype=np.float64), np.asarray(U, dtype=np.float64)


@register("s80_evaluate_bundle")
class EvaluateBundle(Stage):
    required_inputs = ["frames", "court3d", "K"]
    produces = ["report.eval_bundle"]
    STAGE_VERSION = "1.0.0"

    def run(self, B: Bundle) -> Bundle:
        if not B.frames:
            raise ValueError("EvaluateBundle: no frames")
        if not B.K:
            raise ValueError("EvaluateBundle: K missing")

        Kd = B.K
        fx, fy, cx, cy = Kd.get("fx"), Kd.get("fy"), Kd.get("cx"), Kd.get("cy")
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

        # Map frame_idx -> pose
        pose_map: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        if getattr(B, "poses", None):
            for p in B.poses:
                fid = int(p.get("frame_idx", -1))
                if fid >= 0:
                    pose_map[fid] = (
                        np.asarray(p.get("R"), dtype=np.float64),
                        np.asarray(p.get("t"), dtype=np.float64).reshape(3),
                    )
        # fallback to single pose
        single_R = None
        single_t = None
        if not pose_map and B.pose_cw is not None:
            single_R = np.asarray(B.pose_cw.get("R"), dtype=np.float64)
            single_t = np.asarray(B.pose_cw.get("t"), dtype=np.float64).reshape(3)

        thr = float(self.cfg.get("threshold_px", 3.0))

        total_sq = 0.0
        total_pts = 0
        per_frame = []
        heights = []
        for fr in B.frames:
            Xw, uv = _gather_matches3d2d(B.court3d, fr.keypoints)
            if Xw.size == 0:
                continue
            if fr.frame_idx in pose_map:
                R, t = pose_map[fr.frame_idx]
            else:
                if single_R is None or single_t is None:
                    # skip frame if no pose
                    continue
                R, t = single_R, single_t
            uv_hat = project_points_W(Xw, intr, PoseCW(R=R, t=t))
            res = uv_hat - uv
            sq = float(np.sum(res**2))
            total_sq += sq
            total_pts += int(len(uv))
            inliers = int(np.sum(np.sum(res**2, axis=1) <= thr * thr))
            rmse = float(np.sqrt(np.mean(np.sum(res**2, axis=1))))
            per_frame.append({
                "frame_idx": int(fr.frame_idx),
                "rmse_px": rmse,
                "inliers": inliers,
                "total": int(len(uv)),
            })
            # camera height

            _Rwc, twc = invert_pose(PoseCW(R=R, t=t))
            heights.append(float(twc[2]))

        overall_rmse = float(np.sqrt(total_sq / max(1, total_pts)))
        height_mean = float(np.mean(heights)) if heights else None

        B.report.setdefault("eval_bundle", {})
        B.report["eval_bundle"].update({
            "overall_rmse_px": overall_rmse,
            "total_points": int(total_pts),
            "n_frames": int(len(per_frame)),
            "threshold_px": thr,
            "per_frame": per_frame,
            "camera_height_mean_m": height_mean,
        })
        return B
