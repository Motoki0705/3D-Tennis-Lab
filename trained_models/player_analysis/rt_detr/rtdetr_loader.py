# rtdetr_loader.py
"""
Loader that:
  1) Instantiates the pure Transformers RT-DETR model (e.g., 'PekingU/rtdetr_v2_r18vd')
  2) Overwrites its weights using a Lightning checkpoint (after removing 'model.' prefix)
  3) Returns the model, processor (transforms), and device

Quickstart
----------
from rtdetr_loader import RTDetrLoadConfig, load_hf_rtdetr_with_ckpt

cfg = RTDetrLoadConfig.from_yaml("rtdetr_config.yaml")
model, processor, device = load_hf_rtdetr_with_ckpt(cfg)

# Inference example
# -----------------
# 1) Preprocess
from PIL import Image
image = Image.open("example.jpg").convert("RGB")
batch = processor(images=image, return_tensors="pt")
batch = {k: v.to(device) for k, v in batch.items()}

# 2) Forward
import torch
model.eval()
with torch.inference_mode():
    outputs = model(**batch)

# 3) Post-process to get boxes/scores/labels in image coords
w, h = image.size
results = processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=[(h, w)]
)[0]

print({
    "boxes": results["boxes"].tolist(),
    "scores": [float(s) for s in results["scores"]],
    "labels": [int(l) for l in results["labels"]],
})
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch
import yaml
from rtdetr_util import (
    align_and_load,
    load_lightning_state_dict,
    strip_prefix,
)
from transformers import AutoImageProcessor, RTDetrForObjectDetection


@dataclass
class RTDetrLoadConfig:
    # --- Paths ---
    checkpoint_path: str

    # --- Base model ---
    pretrained_model_name_or_path: str = "PekingU/rtdetr_v2_r18vd"
    num_labels: int = 1  # must match your training

    # --- Runtime ---
    device: str = "cuda"  # "cuda" | "cpu" | "mps"
    strict: bool = False  # strong consistency check for load_state_dict
    remove_prefix: str = "model."  # LightningModule wrapper usually prefixes "model."
    allow_partial: bool = True  # keep intersecting keys only

    # --- Optional performance toggles ---
    torch_compile: bool = False  # requires PyTorch 2.x

    @classmethod
    def from_yaml(cls, path: str) -> RTDetrLoadConfig:
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False, allow_unicode=True)


def _select_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_hf_rtdetr_with_ckpt(
    cfg: RTDetrLoadConfig,
) -> tuple[RTDetrForObjectDetection, AutoImageProcessor, torch.device]:
    """
    Build a pure HF RT-DETR model and overwrite weights from a Lightning checkpoint.

    Returns
    -------
    model : RTDetrForObjectDetection
        HF model with weights loaded from the checkpoint (as far as keys matched).
    processor : AutoImageProcessor
        Proper preprocessing / post-processing for RT-DETR.
    device : torch.device
        Device where the model has been moved.

    Notes
    -----
    - If your classifier head (num_labels) differs from the base model, some head
      weights won't match (expected). Use `strict=False` to allow partial load.
    - The loader prints a concise load report (matched/missing/unexpected keys).
    """
    if not os.path.isfile(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

    device = _select_device(cfg.device)

    # 1) Instantiate *pure* Transformers model
    model = RTDetrForObjectDetection.from_pretrained(
        cfg.pretrained_model_name_or_path,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,  # helpful when head size differs
    ).to(device)

    # 2) Create processor (transforms)
    processor = AutoImageProcessor.from_pretrained(cfg.pretrained_model_name_or_path)

    # 3) Read Lightning checkpoint -> strip "model." -> align & load
    raw_sd = load_lightning_state_dict(cfg.checkpoint_path)
    if cfg.remove_prefix:
        raw_sd = strip_prefix(raw_sd, prefix=cfg.remove_prefix)

    if cfg.allow_partial:
        align_and_load(model, raw_sd, strict=cfg.strict)
    else:
        # Force exact key match (typically not recommended unless you know they match)
        model.load_state_dict(raw_sd, strict=cfg.strict)

    # Optional compile for inference speed (PyTorch 2.x)
    if cfg.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            print("[rtdetr_loader] Model compiled with torch.compile")
        except Exception as e:
            print(f"[rtdetr_loader] torch.compile failed (continuing without it): {e}")

    model.eval()
    return model, processor, device
