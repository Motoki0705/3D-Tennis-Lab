from __future__ import annotations

import numpy as np

from ..base import Bundle, Stage
from ..base import register
from ...core.camera import Intrinsics, PoseCW
from ...core.geometry import project_points_W, invert_pose


@register("s50_evaluate")
class Evaluate(Stage):
    required_inputs = ["frames", "court3d", "K", "pose_cw", "min_visible_4", "ref_integrity"]
    produces = ["report.eval"]
    STAGE_VERSION = "1.0.0"

    def run(self, B: Bundle) -> Bundle:
        if not B.frames:
            raise ValueError("Evaluate: Bundle.frames is empty")
        if not B.K:
            raise ValueError("Evaluate: Bundle.K is missing")
        if not B.pose_cw:
            raise ValueError("Evaluate: pose_cw is missing")

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

        kp = B.frames[0].keypoints
        Xw = []
        uv = []
        for name, xyz in B.court3d.items():
            if name not in kp:
                continue
            u, v, vis, w = kp[name]
            if vis and w > 0:
                Xw.append([float(xyz[0]), float(xyz[1]), 0.0])
                uv.append([float(u), float(v)])
        Xw = np.asarray(Xw, dtype=np.float64)
        uv = np.asarray(uv, dtype=np.float64)

        R = np.asarray(B.pose_cw.get("R"), dtype=np.float64)
        t = np.asarray(B.pose_cw.get("t"), dtype=np.float64).reshape(3)
        uv_hat = project_points_W(Xw, intr, PoseCW(R=R, t=t))
        res = uv_hat - uv
        rmse = float(np.sqrt(np.mean(np.sum(res**2, axis=1))))

        thr = float(self.cfg.get("threshold_px", 3.0))
        inliers = int(np.sum(np.sum(res**2, axis=1) <= thr * thr))
        total = int(len(uv))
        ratio = float(inliers / total) if total > 0 else 0.0

        _Rwc, twc = invert_pose(PoseCW(R=R, t=t))
        camera_height_m = float(twc[2])

        B.report.setdefault("eval", {})
        B.report["eval"].update({
            "rmse_px": rmse,
            "inlier_ratio": ratio,
            "inliers": inliers,
            "total": total,
            "threshold_px": thr,
            "camera_height_m": camera_height_m,
        })
        return B
