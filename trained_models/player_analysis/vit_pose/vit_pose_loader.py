# pose_loader.py
"""
Pose loader (HF only, no training checkpoint).

What this does
--------------
- Loads a pose-estimation model and its matching processor directly from Hugging Face.
- Returns (model, processor, device) ready for inference.
- Plays nicely with RT-DETR person detection by accepting COCO-format boxes (x, y, w, h).

Quickstart
----------
from pose_loader import PoseLoadConfig, load_pose_from_hub

cfg = PoseLoadConfig.from_yaml("pose_config.yaml")
pose_model, pose_processor, device = load_pose_from_hub(cfg)

# Example inference with RT-DETR person boxes
# -------------------------------------------
# Assume `image` is a PIL.Image and `person_boxes` is a NumPy array of shape [N, 4]
# in COCO format (x, y, w, h), as produced by your RT-DETR step.

import torch

inputs = pose_processor(
    image,
    boxes=[person_boxes],          # list length = batch size (here 1)
    return_tensors="pt"
)

# Move to the same device as the model
inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

pose_model.eval()
with torch.inference_mode():
    outputs = pose_model(**inputs)

# Convert model outputs to keypoints/scores in image coordinates
pose_results = pose_processor.post_process_pose_estimation(
    outputs, boxes=[person_boxes]
)
# pose_results[0] is a list (len=N persons) of dicts with 'keypoints' (Kx2) and 'scores' (K)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
import yaml
from transformers import AutoProcessor, VitPoseForPoseEstimation


# -----------------------
# Config
# -----------------------
@dataclass
class PoseLoadConfig:
    # Model to pull from the Hub (VitPose base is a solid default)
    model_id: str = "usyd-community/vitpose-base-simple"

    # Runtime
    device: Literal["cuda", "cpu", "mps", "auto"] = "auto"
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    use_device_map: bool = False  # set True if you want HF/accelerate to shard automatically

    # Optional: local cache directory
    cache_dir: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> PoseLoadConfig:
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False, allow_unicode=True)


# -----------------------
# Loader
# -----------------------
def _select_device(preferred: str) -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype | None:
    if dtype == "auto":
        # default to fp16 on CUDA, bf16 on CPU if available, else fp32
        if device.type == "cuda":
            return torch.float16
        if torch.cuda.is_bf16_supported() or getattr(torch, "supports_bf16", False):
            return torch.bfloat16
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def load_pose_from_hub(
    cfg: PoseLoadConfig,
) -> tuple[VitPoseForPoseEstimation, AutoProcessor, torch.device]:
    """
    Load a pose model + processor from Hugging Face and move the model to device.

    Returns
    -------
    model : VitPoseForPoseEstimation
    processor : transformers.AutoProcessor
    device : torch.device

    Notes
    -----
    - Boxes passed to the processor must be COCO format (x, y, w, h) in absolute pixels.
    - For a batch of size B, pass `boxes=[boxes_img0, boxes_img1, ...]` (list length B).
    - If you use RT-DETR for person detection, pass its per-image person boxes directly.
    """
    device = _select_device(cfg.device)
    torch_dtype = _resolve_dtype(cfg.dtype, device)

    processor = AutoProcessor.from_pretrained(cfg.model_id, cache_dir=cfg.cache_dir)

    common_kwargs = {"torch_dtype": torch_dtype, "cache_dir": cfg.cache_dir}
    if cfg.use_device_map:
        # Let accelerate place layers (requires accelerate installed)
        model = VitPoseForPoseEstimation.from_pretrained(cfg.model_id, device_map="auto", **common_kwargs)
        # When device_map is used, we don't .to(device)
    else:
        model = VitPoseForPoseEstimation.from_pretrained(cfg.model_id, **common_kwargs).to(device)

    model.eval()
    return model, processor, device
