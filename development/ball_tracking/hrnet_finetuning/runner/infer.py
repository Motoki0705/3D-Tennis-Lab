from __future__ import annotations

import os
from typing import Tuple, Optional

import numpy as np
import torch

from .base import BaseRunner
from hydra.utils import to_absolute_path as abspath


class InferRunner(BaseRunner):
    """
    Inference runner for HRNet3DStem-based ball heatmap → point estimation.

    Tracker can be toggled via config:
      - Preferred: cfg.infer.tracker.enable: bool
      - Legacy fallback: cfg.infer.use_tracker: bool

    Other tracker params (prefer nested):
      - cfg.infer.tracker.tail          (fallback: cfg.infer.track_tail)
      - cfg.infer.tracker.hm_threshold  (fallback: cfg.infer.hm_threshold)
      - cfg.infer.tracker.max_missed    (fallback: cfg.infer.track_max_missed)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.frames_in = cfg.model.frames_in
        self.input_h, self.input_w = cfg.infer.input_size
        self.mean = cfg.infer.normalize.mean
        self.std = cfg.infer.normalize.std
        self.dtype = str(cfg.infer.dtype)

        # Build model
        from ..model.base_hrnet_3dstem import HRNet3DStem

        self.model = HRNet3DStem(cfg.model).to(self.device).eval()

        # Load checkpoint (.pth/.pth.tar/.ckpt supported)
        from ..utils import load_model_weights

        missing, unexpected = load_model_weights(self.model, abspath(cfg.infer.checkpoint), strict=False)
        if missing:
            print(f"[warn] missing keys: {missing}")
        if unexpected:
            print(f"[warn] unexpected keys: {unexpected}")

        # Align model dtype with requested inference dtype (after loading weights)
        dt = self.dtype.lower()
        if dt in ("float16", "fp16", "half"):
            self.model = self.model.half()
        elif dt in ("float64", "fp64", "double"):
            self.model = self.model.double()
        else:
            self.model = self.model.float()

        # --- Tracker config (nested preferred; top-level kept for backward compat) ---
        tr_cfg = getattr(cfg.infer, "tracker", None)
        # enable flag: prefer tracker.enable, fallback to infer.use_tracker, default True
        self.use_tracker: bool = bool(
            getattr(tr_cfg, "enable", getattr(cfg.infer, "use_tracker", True))
            if tr_cfg is not None
            else getattr(cfg.infer, "use_tracker", True)
        )
        # parameters (prefer nested)
        self.track_tail: int = int(
            getattr(tr_cfg, "tail", getattr(cfg.infer, "track_tail", 30))
            if tr_cfg
            else getattr(cfg.infer, "track_tail", 30)
        )
        self.hm_threshold: float = float(
            getattr(tr_cfg, "hm_threshold", getattr(cfg.infer, "hm_threshold", 0.2))
            if tr_cfg
            else getattr(cfg.infer, "hm_threshold", 0.2)
        )
        self.track_max_missed: int = int(
            getattr(tr_cfg, "max_missed", getattr(cfg.infer, "track_max_missed", 30))
            if tr_cfg
            else getattr(cfg.infer, "track_max_missed", 30)
        )

        # Drawing / overlay
        self.overlay_heatmap: bool = bool(getattr(cfg.infer, "overlay_heatmap", False))
        self.draw_radius: int = int(getattr(cfg.infer, "draw_radius", 5))
        self.draw_thickness: int = int(getattr(cfg.infer, "draw_thickness", 2))

    @staticmethod
    def _to_tensor(frames: np.ndarray, mean, std, dtype="float32") -> torch.Tensor:
        # frames: (T, H, W, 3) -> (3T, H, W)
        T, H, W, C = frames.shape
        frames_f = frames.astype(np.float32) / 255.0
        frames_f = (frames_f - np.array(mean).reshape(1, 1, 1, 3)) / np.array(std).reshape(1, 1, 1, 3)
        frames_chw = np.transpose(frames_f, (0, 3, 1, 2)).reshape(T * 3, H, W)
        t = torch.from_numpy(frames_chw)
        dt = str(dtype).lower()
        if dt in ("float16", "fp16", "half"):
            t = t.half()
        elif dt in ("float64", "fp64", "double"):
            t = t.double()
        else:
            t = t.float()
        return t

    @staticmethod
    def _resize_frame(frame: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
        H, W = size_hw
        try:
            import cv2
        except Exception as e:
            raise SystemExit("OpenCV (cv2) が必要です。'pip install opencv-python' でインストールしてください。") from e
        return cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _open_video(path: str):
        try:
            import cv2
        except Exception as e:
            raise SystemExit("OpenCV (cv2) が必要です。'pip install opencv-python' でインストールしてください。") from e
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video: {path}")
        return cap, cv2

    @staticmethod
    def _init_writer(cv2, out_path: str, fps: float, size_hw: Tuple[int, int]):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        H, W = size_hw
        return cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    @staticmethod
    def _overlay_heatmap(cv2, frame_bgr: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        hm = heatmap
        if isinstance(hm, torch.Tensor):
            hm = hm.detach().cpu().float().numpy()
        hm = hm - hm.min()
        if hm.max() > 0:
            hm = hm / hm.max()
        hm_u8 = (hm * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame_bgr, 0.6, color, 0.4, 0)
        return blended

    @staticmethod
    def _heatmap_to_point(
        hm: np.ndarray, frame_size_hw: Tuple[int, int]
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Convert heatmap to pixel coordinates (x, y) in the resized frame.

        Returns (point or None, confidence[0..1]). Confidence is derived from
        normalized heatmap peak value.
        """
        if isinstance(hm, torch.Tensor):
            hm_np = hm.detach().cpu().float().numpy()
        else:
            hm_np = np.asarray(hm, dtype=np.float32)
        if hm_np.ndim == 3:
            # Heuristic: use last channel or max over channels
            hm_np = hm_np[-1] if hm_np.shape[0] > 1 else hm_np[0]

        # Normalize 0..1 for a relative confidence
        mmin = float(hm_np.min())
        hm_norm = hm_np - mmin
        mmax = float(hm_norm.max())
        if mmax > 0:
            hm_norm = hm_norm / mmax
        else:
            return None, 0.0

        # Peak
        idx = np.argmax(hm_norm)
        py_hm, px_hm = np.unravel_index(int(idx), hm_norm.shape)  # row, col (y, x)
        conf = float(hm_norm[py_hm, px_hm])

        # Subpixel refinement using local weighted centroid (3x3 window)
        y0, y1 = max(py_hm - 1, 0), min(py_hm + 2, hm_norm.shape[0])
        x0, x1 = max(px_hm - 1, 0), min(px_hm + 2, hm_norm.shape[1])
        patch = hm_norm[y0:y1, x0:x1]
        if patch.size > 0 and float(patch.sum()) > 0.0:
            ys, xs = np.mgrid[y0:y1, x0:x1]
            py_hm_ref = float((ys * patch).sum() / patch.sum())
            px_hm_ref = float((xs * patch).sum() / patch.sum())
        else:
            py_hm_ref, px_hm_ref = float(py_hm), float(px_hm)

        Hf, Wf = frame_size_hw
        Hy, Wx = hm_norm.shape[0], hm_norm.shape[1]
        sx, sy = float(Wf) / float(Wx), float(Hf) / float(Hy)
        px = px_hm_ref * sx
        py = py_hm_ref * sy
        return (px, py), conf

    @staticmethod
    def _draw_tracking(
        cv2,
        frame_bgr: np.ndarray,
        trail_xy: list[Tuple[float, float]],
        cur_xy: Tuple[float, float],
        radius: int = 5,
        thickness: int = 2,
    ) -> np.ndarray:
        img = frame_bgr.copy()
        # Draw trail as fading line segments
        n = len(trail_xy)
        if n >= 2:
            for i in range(1, n):
                a = trail_xy[i - 1]
                b = trail_xy[i]
                # Fade color older->newer (light blue -> yellow)
                t = i / (n - 1 + 1e-6)
                color = (
                    int(255 * (1.0 - t)),  # B
                    int(200 * t + 50 * (1.0 - t)),  # G
                    int(50 + 205 * t),  # R
                )
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, thickness)

        # Draw current point
        cv2.circle(img, (int(cur_xy[0]), int(cur_xy[1])), radius, (0, 0, 255), -1)
        cv2.circle(img, (int(cur_xy[0]), int(cur_xy[1])), max(1, radius + 2), (255, 255, 255), 1)
        return img

    def run(self):
        cfg = self.cfg
        cap, cv2 = self._open_video(abspath(cfg.infer.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out_writer = None
        if cfg.infer.output_video:
            out_writer = self._init_writer(cv2, abspath(cfg.infer.output_video), fps, (self.input_h, self.input_w))

        # Build tracker (optional)
        tracker = None
        simple_trail: list[Tuple[float, float]] = []
        missed_simple = 0
        if self.use_tracker:
            from ..tracker import BallTracker

            tracker = BallTracker(
                image_size_hw=(self.input_h, self.input_w),
                history_len=self.track_tail,
                min_conf=self.hm_threshold,
                max_missed=self.track_max_missed,
            )

        if cfg.infer.save_heatmaps_dir:
            os.makedirs(abspath(cfg.infer.save_heatmaps_dir), exist_ok=True)

        window: list[np.ndarray] = []
        stride = int(cfg.infer.stride)
        out_idx = 0

        # Progress bar over frames
        try:
            from tqdm import tqdm  # type: ignore

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Inferring frames", unit="frm")
        except Exception:

            class _NoTQDM:
                def update(self, n=1):
                    pass

                def close(self):
                    pass

                def set_postfix(self, **kwargs):
                    pass

            pbar = _NoTQDM()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = self._resize_frame(frame, (self.input_h, self.input_w))
            window.append(frame)
            if len(window) < self.frames_in:
                pbar.update(1)
                continue

            batch = self._to_tensor(np.stack(window[-self.frames_in :], axis=0), self.mean, self.std, self.dtype)
            x = batch.unsqueeze(0).to(self.device)
            with torch.no_grad():
                y_out = self.model(x)

            # Assume dict of scale -> tensor list; take the first output scale
            scale = cfg.model.out_scales[0]
            y = y_out[scale][0]
            hm = y[-1] if (y.ndim == 3 and y.shape[0] > 1) else (y[0] if y.ndim == 3 else y.squeeze(0))

            # Measurement from heatmap
            meas_xy, conf = self._heatmap_to_point(hm, (self.input_h, self.input_w))

            if tracker is not None:
                est_xy = tracker.update(meas_xy, conf)
                trail = tracker.get_trail(self.track_tail)
                missed_to_show = tracker.missed
            else:
                # No tracker: pass-through measurement and keep a simple visualization trail
                if (meas_xy is not None) and (conf >= self.hm_threshold):
                    simple_trail.append(meas_xy)
                    if len(simple_trail) > self.track_tail:
                        simple_trail = simple_trail[-self.track_tail :]
                    est_xy = meas_xy
                    missed_simple = 0
                else:
                    # If no confident detection, keep previous estimate but count as missed
                    missed_simple += 1
                    est_xy = simple_trail[-1] if simple_trail else (self.input_w * 0.5, self.input_h * 0.5)
                trail = simple_trail
                missed_to_show = missed_simple

            # Compose output frame
            if self.overlay_heatmap:
                disp = self._overlay_heatmap(cv2, frame, hm)
            else:
                disp = frame

            disp = self._draw_tracking(
                cv2,
                disp,
                trail,
                est_xy,
                radius=self.draw_radius,
                thickness=self.draw_thickness,
            )

            # Optional: draw debug text
            cv2.putText(
                disp,
                f"conf={conf:.2f} missed={missed_to_show}{' (trk)' if self.use_tracker else ' (no-trk)'}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if out_writer is not None:
                out_writer.write(disp)

            if cfg.infer.save_heatmaps_dir:
                np.save(
                    os.path.join(abspath(cfg.infer.save_heatmaps_dir), f"{out_idx:06d}.npy"),
                    hm.detach().cpu().numpy(),
                )
            out_idx += 1
            try:
                pbar.set_postfix(outputs=out_idx)
            except Exception:
                pass

            # slide window
            if stride >= self.frames_in:
                window = []
            else:
                window = window[stride:]
            pbar.update(1)

        cap.release()
        if out_writer is not None:
            out_writer.release()
        try:
            pbar.close()
        except Exception:
            pass
        print("Inference done.")


__all__ = ["InferRunner"]
