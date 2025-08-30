from __future__ import annotations

import numpy as np
import math
import pytest

from experiments.pnpkit.src.pipeline.base import Bundle, FrameObs
from experiments.pnpkit.src.pipeline.stages.s20_estimate_k import EstimateK


def _synthesize_H(fx, fy, cx, cy, rx_deg=5.0, ry_deg=3.0, tz=10.0):
    # R from small rotations around x and y
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=float)
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]], dtype=float)
    R = Rx @ Ry
    t = np.array([0.0, 0.0, tz], dtype=float)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return H


def test_estimate_k_square_center():
    H = _synthesize_H(1200.0, 1300.0, 640.0, 360.0)
    B = Bundle(court3d={}, frames=[FrameObs(frame_idx=0, image_path="", keypoints={})])
    B.H = H
    B.K = {"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0}
    stage = EstimateK(principal_center=True, assume_square_pixels=True, enabled=True)
    B = stage(B)
    assert B.report["estimate_k"]["success"] is True
    f = B.K["fx"]
    assert 200.0 <= f <= 10000.0


def test_estimate_k_free_if_scipy():
    try:
        import scipy  # noqa: F401
    except Exception:
        pytest.skip("SciPy not installed; skipping free K estimation test")
    H = _synthesize_H(1200.0, 1300.0, 640.0, 360.0)
    B = Bundle(court3d={}, frames=[FrameObs(frame_idx=0, image_path="", keypoints={})])
    B.H = H
    B.K = {"fx": 900.0, "fy": 900.0, "cx": 700.0, "cy": 300.0}
    stage = EstimateK(principal_center=False, assume_square_pixels=False, enabled=True)
    B = stage(B)
    assert B.report["estimate_k"]["success"] is True
    assert abs(B.K["cx"] - 640.0) < 40.0
