from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from ..base import Bundle, Stage
from ..base import register
from ...core.camera import Intrinsics, PoseCW
from ...core.geometry import project_points_W


@register("s55_viz_overlay")
class VizOverlay(Stage):
    required_inputs = ["frames", "court3d", "K", "pose_cw", "min_visible_4"]
    produces = ["report.viz"]
    STAGE_VERSION = "1.0.0"

    def __init__(
        self,
        enabled: bool = True,
        radius: int = 3,
        thickness: int = 1,
        font_scale: float = 0.4,
        out_dir: str | None = None,
        num_samples: int = 3,
    ):
        self.enabled = enabled
        self.radius = int(radius)
        self.thickness = int(thickness)
        self.font_scale = float(font_scale)
        self.out_dir = out_dir
        self.num_samples = int(num_samples)

    def run(self, B: Bundle) -> Bundle:
        if not self.enabled:
            return B
        if not B.frames:
            return B
        Kd = B.K or {}
        if not B.pose_cw:
            return B

        fx, fy, cx, cy = Kd.get("fx"), Kd.get("fy"), Kd.get("cx"), Kd.get("cy")
        intr = Intrinsics(
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), skew=float(Kd.get("skew", 0.0) or 0.0)
        )
        if Kd.get("dist"):
            d = Kd["dist"]
            intr.dist.k1 = float(d.get("k1", 0.0))
            intr.dist.k2 = float(d.get("k2", 0.0))
            intr.dist.p1 = float(d.get("p1", 0.0))
            intr.dist.p2 = float(d.get("p2", 0.0))
            intr.dist.k3 = float(d.get("k3", 0.0))

        # Helper to fetch pose for a given frame
        def _pose_for_frame(fid: int):
            if getattr(B, "poses", None):
                for p in B.poses:
                    if int(p.get("frame_idx", -1)) == int(fid):
                        return np.asarray(p.get("R"), dtype=np.float64), np.asarray(
                            p.get("t"), dtype=np.float64
                        ).reshape(3)
            return (
                np.asarray(B.pose_cw.get("R"), dtype=np.float64),
                np.asarray(B.pose_cw.get("t"), dtype=np.float64).reshape(3),
            )

        # Default to Hydra run dir via env or current directory; save under bundle_id if present
        import os

        base_dir = Path(self.out_dir or os.environ.get("RUN_DIR", "."))
        subdir = getattr(B, "bundle_id", None)
        out_dir = base_dir / (str(subdir) if subdir else "")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Determine sample frames (up to num_samples, spread across bundle)
        n = len(B.frames)
        idxs = list(range(n))
        if n > self.num_samples and self.num_samples > 0:
            import numpy as _np

            # Select approximately evenly spaced indices
            idxs = [int(round(i)) for i in _np.linspace(0, n - 1, self.num_samples)]

        saved_paths = []
        for i in idxs:
            fr_i = B.frames[i]
            # recompute overlays per selected frame
            R, t = _pose_for_frame(fr_i.frame_idx)
            kp_i = fr_i.keypoints
            Xw_i = []
            uv_obs_i = []
            for name, xyz in B.court3d.items():
                if name not in kp_i:
                    continue
                u, v, vis, w = kp_i[name]
                if vis and w > 0:
                    Xw_i.append([float(xyz[0]), float(xyz[1]), 0.0])
                    uv_obs_i.append([float(u), float(v)])
            Xw_i = np.asarray(Xw_i, dtype=np.float64)
            uv_obs_i = np.asarray(uv_obs_i, dtype=np.float64)
            uv_hat_i = project_points_W(Xw_i, intr, PoseCW(R=R, t=t))
            res_i = uv_hat_i - uv_obs_i
            mask_i = None
            if B.report.get("refine_lm", {}).get("inlier_mask") is not None and len(
                B.report["refine_lm"]["inlier_mask"]
            ) == len(res_i):
                mask_i = np.array(B.report["refine_lm"]["inlier_mask"], dtype=int)
            else:
                thr = float(B.report.get("eval", {}).get("threshold_px", 3.0))
                mask_i = (np.sum(res_i**2, axis=1) <= thr * thr).astype(int)

            img_i = cv2.imread(fr_i.image_path)
            if img_i is None:
                Hh = int(max(2 * cy, 720))
                Ww = int(max(2 * cx, 1280))
                img_i = np.full((Hh, Ww, 3), 255, dtype=np.uint8)

            for j, (u, v) in enumerate(uv_obs_i):
                color = (0, 200, 0) if mask_i[j] == 1 else (0, 0, 200)
                cv2.circle(img_i, (int(round(u)), int(round(v))), self.radius, color, -1, lineType=cv2.LINE_AA)
            for j, (u, v) in enumerate(uv_hat_i):
                color = (0, 150, 0) if mask_i[j] == 1 else (0, 0, 150)
                p = (int(round(u)), int(round(v)))
                cv2.line(
                    img_i,
                    (p[0] - self.radius, p[1]),
                    (p[0] + self.radius, p[1]),
                    color,
                    self.thickness,
                    lineType=cv2.LINE_AA,
                )
                cv2.line(
                    img_i,
                    (p[0], p[1] - self.radius),
                    (p[0], p[1] + self.radius),
                    color,
                    self.thickness,
                    lineType=cv2.LINE_AA,
                )

            out_path_i = out_dir / f"overlay_{fr_i.frame_idx:06d}.png"
            cv2.imwrite(str(out_path_i), img_i)
            saved_paths.append(str(out_path_i).replace("\\", "/"))

        B.report.setdefault("viz", {})
        B.report["viz"]["overlay_paths"] = saved_paths
        return B
