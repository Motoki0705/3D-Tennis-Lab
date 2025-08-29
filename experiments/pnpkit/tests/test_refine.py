from __future__ import annotations

import numpy as np

from experiments.pnpkit.src.pipeline.base import Bundle, FrameObs
from experiments.pnpkit.src.pipeline.stages.s30_ippe_init import IPPEInit
from experiments.pnpkit.src.pipeline.stages.s40_refine_lm import RefineLM
from experiments.pnpkit.src.pipeline.stages.s50_evaluate import Evaluate
from experiments.pnpkit.src.core.camera import Intrinsics, PoseCW
from experiments.pnpkit.src.core.geometry import project_points_W


def _make_court_points():
    # 6 points rectangle + midpoints
    pts = {
        "A": np.array([-5.0, -10.0, 0.0], dtype=float),
        "B": np.array([5.0, -10.0, 0.0], dtype=float),
        "C": np.array([5.0, 10.0, 0.0], dtype=float),
        "D": np.array([-5.0, 10.0, 0.0], dtype=float),
        "E": np.array([0.0, -10.0, 0.0], dtype=float),
        "F": np.array([0.0, 10.0, 0.0], dtype=float),
    }
    return pts


def test_refine_pipeline_smoke():
    court3d = _make_court_points()

    intr = Intrinsics(fx=1200.0, fy=1200.0, cx=640.0, cy=360.0)
    R = np.eye(3, dtype=float)
    t = np.array([0.0, 0.0, 12.0], dtype=float)
    pose = PoseCW(R=R, t=t)

    Xw = np.stack(list(court3d.values()), axis=0)
    uv = project_points_W(Xw, intr, pose)
    rng = np.random.default_rng(0)
    uv_noisy = uv + rng.normal(scale=1.0, size=uv.shape)

    kps = {}
    for (name, _), (u, v) in zip(court3d.items(), uv_noisy):
        kps[name] = (float(u), float(v), 1, 1.0)

    B = Bundle(court3d=court3d, frames=[FrameObs(frame_idx=0, image_path="", keypoints=kps)])
    B.K = {
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.cx,
        "cy": intr.cy,
        "skew": 0.0,
        "dist": {"k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0},
    }

    # IPPE init
    B = IPPEInit()(B)
    assert B.pose_cw is not None

    # Refine LM
    B = RefineLM(inlier_threshold_px=3.0)(B)
    assert B.report["refine_lm"]["rmse_px_after"] < B.report["refine_lm"]["rmse_px_before"]

    # Evaluate
    B = Evaluate(threshold_px=3.0)(B)
    assert B.report["eval"]["rmse_px"] < 1.8
    assert 5.0 <= B.report["eval"]["camera_height_m"] <= 20.0
