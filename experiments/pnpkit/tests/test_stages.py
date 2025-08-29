from __future__ import annotations

import numpy as np

from experiments.pnpkit.src.pipeline.base import Bundle, FrameObs
from experiments.pnpkit.src.pipeline.stages.s10_init_homography import InitHomography
from experiments.pnpkit.src.pipeline.stages.s30_ippe_init import IPPEInit
from experiments.pnpkit.src.core.camera import Intrinsics, PoseCW
from experiments.pnpkit.src.core.geometry import project_points_W


def _make_court_rect():
    # Simple rectangle 10x20 m centered at origin, Z=0
    pts = {
        "A": np.array([-5.0, -10.0, 0.0], dtype=float),
        "B": np.array([5.0, -10.0, 0.0], dtype=float),
        "C": np.array([5.0, 10.0, 0.0], dtype=float),
        "D": np.array([-5.0, 10.0, 0.0], dtype=float),
    }
    return pts


def test_init_homography_and_ippe_smoke():
    court3d = _make_court_rect()

    intr = Intrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
    R = np.eye(3, dtype=float)
    t = np.array([0.0, 0.0, 10.0], dtype=float)
    pose = PoseCW(R=R, t=t)

    # Build observations with slight noise
    Xw = np.stack(list(court3d.values()), axis=0)
    uv = project_points_W(Xw, intr, pose)
    rng = np.random.default_rng(42)
    uv_noisy = uv + rng.normal(scale=0.5, size=uv.shape)

    keypoints = {}
    for (name, _), (u, v) in zip(court3d.items(), uv_noisy):
        keypoints[name] = (float(u), float(v), 1, 1.0)

    B = Bundle(court3d=court3d, frames=[FrameObs(frame_idx=0, image_path="", keypoints=keypoints)])
    B.K = {
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.cx,
        "cy": intr.cy,
        "skew": 0.0,
        "dist": {"k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0},
    }

    # Run homography
    sH = InitHomography(ransac={"reproj_px": 3.0, "iters": 2000, "conf": 0.999})
    B = sH(B)
    assert B.H is not None
    assert "homography" in B.report

    # Run IPPE init
    sI = IPPEInit()
    B = sI(B)
    assert B.pose_cw is not None
    assert "ippe" in B.report
    assert B.report["ippe"]["rmse_px"] < 3.0
