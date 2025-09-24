from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as T

# Add WASB-SBDT to path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
WASB_SRC = SCRIPT_DIR.parent.parent.parent.parent / "third_party" / "WASB-SBDT" / "src"
sys.path.append(str(WASB_SRC))

from detectors import build_detector  # type: ignore  # noqa: E402
from trackers import build_tracker  # type: ignore  # noqa: E402
from utils.image import get_affine_transform  # type: ignore  # noqa: E402

from omegaconf import DictConfig

from .parquet_io import append_records
from .state import StateRepository

LOG = logging.getLogger(__name__)


def make_transform(img_shape: tuple[int, int], input_wh: tuple[int, int], inv: int = 0) -> np.ndarray:
    h, w = img_shape
    c = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    s = max(h, w) * 1.0
    input_w, input_h = input_wh
    return get_affine_transform(c, s, 0, [input_w, input_h], inv=inv)


class InferenceEngine:
    def __init__(self, detect_cfg: DictConfig):
        self.cfg = detect_cfg
        requested_device = detect_cfg.runner.device
        if requested_device == "cuda" and not torch.cuda.is_available():
            LOG.warning("CUDA requested but not available. Falling back to CPU.")
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.detector = build_detector(detect_cfg)
        self.tracker = build_tracker(detect_cfg)

        self.frames_in = int(detect_cfg.model.frames_in)
        self.input_size = (int(detect_cfg.model.inp_width), int(detect_cfg.model.inp_height))
        self.output_size = (int(detect_cfg.model.out_width), int(detect_cfg.model.out_height))
        self.output_scale = detect_cfg.model.out_scales[0]

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def run(
        self,
        video_path: Path,
        parquet_path: Path,
        repo: StateRepository,
        video_id: str,
        *,
        start_frame: int = 0,
        save_every_n_frames: int = 100,
    ) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        repo.update_progress(video_id, total_frames=total_frames, status="running")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        trans_input = make_transform((height, width), self.input_size)
        trans_output_inv = make_transform((height, width), self.output_size, inv=1)
        trans_output_tensor = torch.tensor(trans_output_inv, dtype=torch.float32, device=self.device).unsqueeze(0)

        frame_buffer: deque[torch.Tensor] = deque(maxlen=self.frames_in)
        records: List[dict] = []

        self.tracker.refresh()

        frame_idx = start_frame
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            warped = cv2.warpAffine(frame, trans_input, self.input_size, flags=cv2.INTER_LINEAR)
            img_tensor = self.transforms(warped).to(self.device)
            frame_buffer.append(img_tensor)

            record = {
                "frame_idx": frame_idx,
                "t": timestamp,
                "has_ball": False,
                "xc": 0.0,
                "yc": 0.0,
                "conf": 0.0,
            }

            if len(frame_buffer) == self.frames_in:
                input_tensor = torch.cat(list(frame_buffer), dim=0).unsqueeze(0)
                batch_results, _ = self.detector.run_tensor(
                    input_tensor,
                    {self.output_scale: trans_output_tensor},
                )
                preds = batch_results[0][self.frames_in - 1]
                tracked = self.tracker.update(preds)
                if tracked and tracked.get("visi"):
                    record.update({
                        "has_ball": True,
                        "xc": float(tracked.get("x", 0.0)),
                        "yc": float(tracked.get("y", 0.0)),
                        "conf": float(tracked.get("score", 1.0)),
                    })

            records.append(record)
            processed_frames += 1
            if processed_frames % save_every_n_frames == 0:
                append_records(parquet_path, records)
                repo.update_progress(video_id, last_processed_frame=frame_idx, status="running")
                records = []

            frame_idx += 1

        cap.release()

        if records:
            append_records(parquet_path, records)

        last_frame = frame_idx - 1 if frame_idx > start_frame else start_frame - 1
        repo.update_progress(video_id, last_processed_frame=last_frame, status="done")
        LOG.info("Finished inference for %s (frames: %s)", video_path, frame_idx - start_frame)
