from __future__ import annotations

import numpy as np
import math
import pytest

from experiments.pnpkit.src.pipeline.base import Bundle, FrameObs
from experiments.pnpkit.src.pipeline.stages.s30_ippe_init import IPPEInit
from experiments.pnpkit.src.pipeline.stages.s70_bundle_adjust import BundleAdjust
from experiments.pnpkit.src.core.camera import Intrinsics, PoseCW
from experiments.pnpkit.src.core.geometry import project_points_W


def _spread_points(n: int = 16):
    xs = np.linspace(-6.0, 6.0, num=4)
    ys = np.linspace(-12.0, 12.0, num=4)
    pts = {}
    idx = 0
    for y in ys:
        for x in xs:
            pts[f"P{idx:02d}"] = np.array([x, y, 0.0], dtype=float)
            idx += 1
    return pts


def test_bundle_adjust_multi_frame():
    try:
        import scipy  # noqa: F401
    except Exception:
        pytest.skip("SciPy not installed; skipping bundle adjust test")

    court3d = _spread_points()
    fx_gt, fy_gt, cx_gt, cy_gt = 1200.0, 1200.0, 640.0, 360.0
    intr = Intrinsics(fx=fx_gt, fy=fy_gt, cx=cx_gt, cy=cy_gt)

    # Create multiple frames with slight camera motion
    frames = []
    poses_true = []
    rng = np.random.default_rng(10)
    for i in range(8):
        rx = math.radians(rng.normal(0.0, 1.0))
        ry = math.radians(rng.normal(0.0, 1.0))
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=float)
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]], dtype=float)
        R = Rx @ Ry
        t = np.array([0.0, 0.0, 12.0 + 0.3 * i], dtype=float)
        poses_true.append(PoseCW(R=R, t=t))

        Xw = np.stack(list(court3d.values()), axis=0)
        uv = project_points_W(Xw, intr, PoseCW(R=R, t=t))
        uv_noisy = uv + rng.normal(scale=0.8, size=uv.shape)
        kps = {}
        for (name, _), (u, v) in zip(court3d.items(), uv_noisy):
            kps[name] = (float(u), float(v), 1, 1.0)
        frames.append(FrameObs(frame_idx=i, image_path="", keypoints=kps))

    B = Bundle(court3d=court3d, frames=frames)
    B.K = {
        "fx": 1000.0,
        "fy": 1000.0,
        "cx": 620.0,
        "cy": 380.0,
        "skew": 0.0,
        "dist": {"k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0},
    }

    # Per-frame init
    B = IPPEInit()(B)
    assert B.poses and len(B.poses) == len(frames)

    # Bundle adjust unlocking Rt then fx,fy
    B = BundleAdjust(
        refine={"R": True, "t": True, "fx": True, "fy": True, "cx": False, "cy": False, "dist": False},
        robust={"loss": "huber", "f_scale": 1.0},
        schedule=[{"unlock": ["R", "t"], "max_nfev": 60}, {"unlock": ["fx", "fy"], "max_nfev": 80}],
        bounds={
            "fx": [200.0, 10000.0],
            "fy": [200.0, 10000.0],
            "cx": [0.0, 4096.0],
            "cy": [0.0, 4096.0],
            "k": [-0.5, 0.5],
            "p": [-0.1, 0.1],
        },
    )(B)

    assert B.report["bundle_adjust"]["rmse_px_after"] <= B.report["bundle_adjust"]["rmse_px_before"] + 1e-6
    assert abs(B.K["fx"] - fx_gt) < 30.0
