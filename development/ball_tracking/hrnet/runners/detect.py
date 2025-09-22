import logging
from collections import deque
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from tqdm import tqdm

import os
import sys

# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the WASB-SBDT/src directory
wasb_src_path = os.path.join(script_dir, "..", "..", "..", "..", "third_party", "WASB-SBDT", "src")
# Add it to the Python path
sys.path.append(os.path.normpath(wasb_src_path))

from detectors import build_detector
from trackers import build_tracker
from utils.image import get_affine_transform
from .base import BaseRunner

log = logging.getLogger(__name__)


def get_transform(img_shape, input_wh, inv=0):
    h, w = img_shape
    c = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    s = max(h, w) * 1.0
    input_w, input_h = input_wh
    trans = get_affine_transform(c, s, 0, [input_w, input_h], inv=inv)
    return trans


class DetectRunner(BaseRunner):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        requested_device = cfg.runner.device
        if requested_device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but no GPU is available. Falling back to CPU execution.")
            requested_device = "cpu"

        self.device = torch.device(requested_device)
        self.detector = build_detector(cfg)
        self.tracker = build_tracker(cfg)

        self.img_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def run(self, video_path: str, output_path: str):
        log.info(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, video_fps, (video_width, video_height))

        input_w = self._cfg.model.inp_width
        input_h = self._cfg.model.inp_height
        frames_in = self._cfg.model.frames_in

        # Get transformation matrices
        trans_input = get_transform((video_height, video_width), (input_w, input_h))
        trans_input_inv = get_transform((video_height, video_width), (input_w, input_h), inv=1)

        # Heatmap transform (assuming output size is 1/4 of input)
        output_w, output_h = self._cfg.model.out_width, self._cfg.model.out_height
        trans_output_inv = get_transform((video_height, video_width), (output_w, output_h), inv=1)

        frame_buffer = deque(maxlen=frames_in)

        self.tracker.refresh()

        for frame_idx in tqdm(range(total_frames), desc="[INFERENCE]"):
            ret, frame = cap.read()
            if not ret:
                break

            # Pre-process frame
            warped_frame = cv2.warpAffine(frame, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
            img_tensor = self.img_transforms(warped_frame).to(self.device)
            frame_buffer.append(img_tensor)

            if len(frame_buffer) < frames_in:
                out_writer.write(frame)  # Write original frame until buffer is full
                continue

            # Inference
            # the detector expects frames concatenated along the channel axis
            input_tensor = torch.cat(list(frame_buffer), dim=0).unsqueeze(0)

            # The second argument to run_tensor is a dictionary of output transforms
            # For now, we only have one scale.
            trans_outputs = {
                self._cfg.model.out_scales[0]: torch.tensor(
                    trans_output_inv, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            }

            batch_results, _ = self.detector.run_tensor(input_tensor, trans_outputs)

            # Post-process and track
            # batch_results is a dict {batch_idx: {frame_idx_in_clip: preds}}
            # We have batch_size=1, so batch_idx is 0.
            # We care about the prediction for the last frame in the buffer.
            preds_for_last_frame = batch_results[0][frames_in - 1]

            tracked_results = self.tracker.update(preds_for_last_frame)

            # Draw results
            if tracked_results and tracked_results["visi"]:
                x, y = tracked_results["x"], tracked_results["y"]

                px, py = int(round(x)), int(round(y))

                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({px}, {py})", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            out_writer.write(frame)

        cap.release()
        out_writer.release()
        log.info(f"Output video saved to: {output_path}")
