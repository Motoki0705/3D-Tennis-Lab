from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch

from .base import BaseRunner
from hydra.utils import to_absolute_path as abspath


class InferRunner(BaseRunner):
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

        # Load checkpoint
        ckpt = torch.load(abspath(cfg.infer.checkpoint), map_location="cpu")
        state_dict = self._load_state_dict(ckpt)
        cleaned = {}
        for k, v in state_dict.items():
            nk = k[len("model.") :] if k.startswith("model.") else k
            cleaned[nk] = v
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[warn] missing keys: {missing}")
        if unexpected:
            print(f"[warn] unexpected keys: {unexpected}")

    @staticmethod
    def _to_tensor(frames: np.ndarray, mean, std, dtype="float32") -> torch.Tensor:
        # frames: (T, H, W, 3) -> (3T, H, W)
        T, H, W, C = frames.shape
        frames_f = frames.astype(np.float32) / 255.0
        frames_f = (frames_f - np.array(mean).reshape(1, 1, 1, 3)) / np.array(std).reshape(1, 1, 1, 3)
        frames_chw = np.transpose(frames_f, (0, 3, 1, 2)).reshape(T * 3, H, W)
        t = torch.from_numpy(frames_chw)
        if dtype == "float16":
            t = t.half()
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

    def run(self):
        cfg = self.cfg
        cap, cv2 = self._open_video(abspath(cfg.infer.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out_writer = None
        if cfg.infer.output_video:
            out_writer = self._init_writer(cv2, abspath(cfg.infer.output_video), fps, (self.input_h, self.input_w))

        if cfg.infer.save_heatmaps_dir:
            os.makedirs(abspath(cfg.infer.save_heatmaps_dir), exist_ok=True)

        window: list[np.ndarray] = []
        stride = int(cfg.infer.stride)
        out_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = self._resize_frame(frame, (self.input_h, self.input_w))
            window.append(frame)
            if len(window) < self.frames_in:
                continue

            batch = self._to_tensor(np.stack(window[-self.frames_in :], axis=0), self.mean, self.std, self.dtype)
            x = batch.unsqueeze(0).to(self.device)
            with torch.no_grad():
                y_out = self.model(x)
            scale = cfg.model.out_scales[0]
            y = y_out[scale][0]
            hm = y[-1] if (y.ndim == 3 and y.shape[0] > 1) else (y[0] if y.ndim == 3 else y.squeeze(0))

            if out_writer is not None:
                over = self._overlay_heatmap(cv2, frame, hm)
                out_writer.write(over)

            if cfg.infer.save_heatmaps_dir:
                np.save(
                    os.path.join(abspath(cfg.infer.save_heatmaps_dir), f"{out_idx:06d}.npy"), hm.detach().cpu().numpy()
                )
            out_idx += 1

            # slide window
            if stride >= self.frames_in:
                window = []
            else:
                window = window[stride:]

        cap.release()
        if out_writer is not None:
            out_writer.release()
        print("Inference done.")


__all__ = ["InferRunner"]
