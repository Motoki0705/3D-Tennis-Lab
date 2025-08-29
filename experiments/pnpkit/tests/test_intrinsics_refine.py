from __future__ import annotations

import math
import numpy as np
import pytest

from experiments.pnpkit.src.pipeline.base import Bundle, FrameObs
from experiments.pnpkit.src.pipeline.stages.s30_ippe_init import IPPEInit
from experiments.pnpkit.src.pipeline.stages.s40_refine_lm import RefineLM
from experiments.pnpkit.src.pipeline.stages.s50_evaluate import Evaluate
from experiments.pnpkit.src.core.camera import Intrinsics, PoseCW
from experiments.pnpkit.src.core.geometry import project_points_W


def _spread_points(n: int = 12):
    xs = np.linspace(-6.0, 6.0, num=4)
    ys = np.linspace(-12.0, 12.0, num=3)
    pts = {}
    idx = 0
    for y in ys:
        for x in xs:
            pts[f"P{idx:02d}"] = np.array([x, y, 0.0], dtype=float)
            idx += 1
    return pts


def test_intrinsics_refine_scipy_backend():
    try:
        import scipy  # noqa: F401
    except Exception:
        pytest.skip("SciPy not installed; skipping intrinsics-refine test")

    court3d = _spread_points(12)
    # Ground-truth intrinsics
    intr_true = Intrinsics(fx=1300.0, fy=1300.0, cx=640.0, cy=360.0)
    # Pose
    rng = np.random.default_rng(123)
    # small random rotation around x,y
    rx, ry = np.deg2rad(2.0) * (rng.random(2) - 0.5)
    R = np.array(
        [
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)],
        ],
        dtype=float,
    ) @ np.array(
        [
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)],
        ],
        dtype=float,
    )
    t = np.array([0.0, 0.0, 12.0], dtype=float)
    pose_true = PoseCW(R=R, t=t)

    Xw = np.stack(list(court3d.values()), axis=0)
    uv = project_points_W(Xw, intr_true, pose_true)
    uv_noisy = uv + rng.normal(scale=0.7, size=uv.shape)

    # Start with biased intrinsics
    intr0 = Intrinsics(fx=1000.0, fy=1000.0, cx=620.0, cy=380.0)
    kps = {}
    for (name, _), (u, v) in zip(court3d.items(), uv_noisy):
        kps[name] = (float(u), float(v), 1, 1.0)

    B = Bundle(court3d=court3d, frames=[FrameObs(frame_idx=0, image_path="", keypoints=kps)])
    B.K = {
        "fx": intr0.fx,
        "fy": intr0.fy,
        "cx": intr0.cx,
        "cy": intr0.cy,
        "skew": 0.0,
        "dist": {"k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0},
    }

    # IPPE init
    B = IPPEInit()(B)

    # Refine with scipy backend; schedule to unlock fx,fy then cx,cy
    B = RefineLM(
        refine={
            "backend": "scipy",
            "R": True,
            "t": True,
            "fx": True,
            "fy": True,
            "cx": True,
            "cy": True,
            "dist": False,
        },
        robust={"loss": "huber", "f_scale": 1.0},
        schedule=[
            {"unlock": ["R", "t"], "max_nfev": 60},
            {"unlock": ["fx", "fy"], "max_nfev": 80},
            {"unlock": ["cx", "cy"], "max_nfev": 60},
        ],
        bounds={
            "fx": [200.0, 10000.0],
            "fy": [200.0, 10000.0],
            "cx": [0.0, 4096.0],
            "cy": [0.0, 4096.0],
            "k": [-0.5, 0.5],
            "p": [-0.1, 0.1],
        },
        priors={"use": False},
        inlier_threshold_px=3.0,
    )(B)

    assert B.report["refine_lm"]["rmse_px_after"] <= B.report["refine_lm"]["rmse_px_before"] + 1e-6

    # Evaluate
    B = Evaluate(threshold_px=3.0)(B)
    assert B.report["eval"]["rmse_px"] < 1.2

    fx_est, fy_est, cx_est, cy_est = B.K["fx"], B.K["fy"], B.K["cx"], B.K["cy"]
    assert abs(fx_est - 1300.0) < 30.0
    assert abs(fy_est - 1300.0) < 30.0
    assert abs(cx_est - 640.0) < 8.0
    assert abs(cy_est - 360.0) < 8.0
