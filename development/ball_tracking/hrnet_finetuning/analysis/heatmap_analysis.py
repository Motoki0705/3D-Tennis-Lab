from __future__ import annotations

from typing import Optional, Tuple, Literal, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


class HeatmapAnalyzer:
    """
    Utilities to convert a ball heatmap to pixel coordinates on the output frame.

    API:
      - to_point(hm, frame_size_hw, strategy='peak+centroid', channel='auto')
        -> ((x, y) | None, confidence [0..1])
    """

    @staticmethod
    def _to_numpy(hm: ArrayLike) -> np.ndarray:
        if isinstance(hm, torch.Tensor):
            return hm.detach().cpu().float().numpy()
        return np.asarray(hm, dtype=np.float32)

    @staticmethod
    def _select_channel(hm3: np.ndarray, channel: Literal["auto", "last", "max"]) -> np.ndarray:
        """
        hm3: (C, H, W)
        - auto : last if C>1 else first
        - last : use hm3[-1]
        - max  : reduce over channels by max
        """
        if hm3.ndim != 3:
            raise ValueError("Expected (C, H, W) heatmap for channel selection.")
        C = hm3.shape[0]
        if channel == "last":
            return hm3[-1]
        if channel == "max":
            return hm3.max(axis=0)
        # auto
        return hm3[-1] if C > 1 else hm3[0]

    @staticmethod
    def _normalize01(hm: np.ndarray) -> np.ndarray:
        hm = hm - float(hm.min())
        mx = float(hm.max())
        if mx > 0.0:
            hm = hm / mx
        return hm

    @staticmethod
    def _peak_with_centroid(hm01: np.ndarray) -> Tuple[Tuple[int, int], float, Tuple[float, float]]:
        """
        Returns:
          - discrete peak (py, px) int
          - conf at peak (0..1)
          - refined (py_ref, px_ref) float using 3x3 weighted centroid
        """
        idx = int(np.argmax(hm01))
        py, px = np.unravel_index(idx, hm01.shape)
        conf = float(hm01[py, px])

        y0, y1 = max(py - 1, 0), min(py + 2, hm01.shape[0])
        x0, x1 = max(px - 1, 0), min(px + 2, hm01.shape[1])
        patch = hm01[y0:y1, x0:x1]
        if patch.size > 0 and float(patch.sum()) > 0.0:
            ys, xs = np.mgrid[y0:y1, x0:x1]
            py_ref = float((ys * patch).sum() / patch.sum())
            px_ref = float((xs * patch).sum() / patch.sum())
        else:
            py_ref, px_ref = float(py), float(px)
        return (py, px), conf, (py_ref, px_ref)

    @staticmethod
    def to_point(
        hm: ArrayLike,
        frame_size_hw: Tuple[int, int],
        *,
        strategy: Literal["peak+centroid"] = "peak+centroid",
        channel: Literal["auto", "last", "max"] = "auto",
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Convert heatmap to (x, y) pixel coordinates in the *resized* frame space.

        Args:
          hm: (H, W) or (C, H, W)
          frame_size_hw: (H_frame, W_frame) of final rendering frame
          strategy: selection/refinement strategy
          channel: if hm is (C, H, W), how to select channel

        Returns:
          (point | None, confidence)
        """
        hm_np = HeatmapAnalyzer._to_numpy(hm)
        # Support (C,H,W) or (H,W)
        if hm_np.ndim == 3:
            hm_np = HeatmapAnalyzer._select_channel(hm_np, channel)

        hm01 = HeatmapAnalyzer._normalize01(hm_np)
        if float(hm01.max()) <= 0.0:
            return None, 0.0

        if strategy == "peak+centroid":
            (_, _), conf, (py_ref, px_ref) = HeatmapAnalyzer._peak_with_centroid(hm01)
        else:
            raise NotImplementedError(f"Unknown strategy: {strategy}")

        Hf, Wf = frame_size_hw
        Hy, Wx = hm01.shape
        sx, sy = float(Wf) / float(Wx), float(Hf) / float(Hy)
        px = px_ref * sx
        py = py_ref * sy
        return (px, py), conf
