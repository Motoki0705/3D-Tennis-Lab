from __future__ import annotations

import os
from typing import Any, List

import torch
from PIL import Image
import numpy as np

from .base import BaseRunner
from hydra.utils import to_absolute_path as abspath
from ..model import build_detection_model


def _load_image(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        arr = np.array(rgb)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


class InferRunner(BaseRunner):
    def __init__(self, cfg: Any):
        super().__init__(cfg)
        self.score_thresh = float(getattr(cfg.infer, "score_thresh", 0.3))
        self.max_dets = int(getattr(cfg.infer, "max_dets", 100))

        self.model = build_detection_model(cfg.model).to(self.device).eval()

        ckpt = getattr(cfg.infer, "checkpoint", None)
        if ckpt:
            ckpt_path = abspath(ckpt)
            obj = torch.load(ckpt_path, map_location=self.device)
            sd = obj.get("state_dict", obj)
            try:
                self.model.load_state_dict(sd, strict=False)
            except Exception:
                # If Lightning checkpoint, strip module prefixes
                cleaned = {k.split("model.", 1)[-1]: v for k, v in sd.items()}
                self.model.load_state_dict(cleaned, strict=False)

    @torch.no_grad()
    def run(self):
        images: List[torch.Tensor] = []
        img_path = getattr(self.cfg.infer, "image", None)
        img_dir = getattr(self.cfg.infer, "images_dir", None)
        if img_path:
            images.append(_load_image(abspath(img_path)))
        if img_dir:
            for name in sorted(os.listdir(abspath(img_dir))):
                p = os.path.join(abspath(img_dir), name)
                if os.path.isfile(p) and any(p.lower().endswith(ext) for ext in [".jpg", ".png", ".jpeg"]):
                    images.append(_load_image(p))

        if not images:
            print("No input images found for inference.")
            return

        images = [t.to(self.device) for t in images]
        outputs = self.model(images)

        for i, out in enumerate(outputs):
            boxes = out.get("boxes", torch.empty(0, 4))
            scores = out.get("scores", torch.empty(0))
            labels = out.get("labels", torch.empty(0, dtype=torch.long))
            keep = scores >= self.score_thresh
            boxes = boxes[keep][: self.max_dets]
            scores = scores[keep][: self.max_dets]
            labels = labels[keep][: self.max_dets]
            print(f"Image {i}: {boxes.shape[0]} detections (score >= {self.score_thresh})")
            if boxes.shape[0] > 0:
                print("  Top-5 scores:", [f"{float(s):.2f}" for s in scores[:5].tolist()])


__all__ = ["InferRunner"]
