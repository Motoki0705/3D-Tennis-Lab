"""Utilities to load and run inference with the DINO-DETR pose checkpoint.

The loader mirrors the structure of the other trained model helpers under
``trained_models/player_analysis``.  It exposes a dataclass driven config, a
``load_dino_detr_pose`` factory, and lightweight preprocessing / post-processing
helpers so you can stand up inference in just a few lines.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as F

from development.player_analysis.dino_detr_pose.model import (
    create_model,
    create_postprocessors,
)
from trained_models.player_analysis.rt_detr.rtdetr_utils import (
    align_and_load,
    load_lightning_state_dict,
    strip_prefix,
)

Tensor = torch.Tensor


@dataclass
class DinoDetrPoseLoadConfig:
    """Configuration for ``load_dino_detr_pose``.

    Parameters mirror the RT-DETR loader so the workflow stays familiar.
    """

    checkpoint_path: str
    model_config_path: str = "development/player_analysis/dino_detr_pose/configs/model_cfg/dino_detr_pose.yaml"
    device: str = "cuda"  # "cuda" | "cpu" | "mps"
    strict: bool = False
    remove_prefix: str = "model."
    allow_partial: bool = True
    image_height: int = 320
    image_width: int = 640
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    torch_compile: bool = False

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> DinoDetrPoseLoadConfig:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls(**data)

    def to_yaml(self, path: str | os.PathLike[str]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(asdict(self), handle, sort_keys=False, allow_unicode=True)


def _select_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model_cfg(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def preprocess_image_factory(
    *,
    image_height: int,
    image_width: int,
    normalize_mean: Sequence[float],
    normalize_std: Sequence[float],
) -> Callable[[Image.Image | Tensor], Tensor]:
    """Create a preprocessing callable that mirrors the training pipeline."""

    size = (int(image_height), int(image_width))
    mean = torch.tensor(normalize_mean, dtype=torch.float32)
    std = torch.tensor(normalize_std, dtype=torch.float32)

    def _to_chw_tensor(image: Image.Image | Tensor) -> Tensor:
        if isinstance(image, Image.Image):
            tensor = F.pil_to_tensor(image)
        elif torch.is_tensor(image):
            tensor = image.clone()
        else:
            raise TypeError(f"Unsupported image type: {type(image)!r}")

        if tensor.ndim != 3:
            raise ValueError("Expected image tensor with shape [C,H,W] or [H,W,C].")

        if tensor.dtype != torch.float32:
            tensor = tensor.float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0

        if tensor.shape[0] in (1, 3):
            return tensor
        if tensor.shape[-1] in (1, 3):
            return tensor.permute(2, 0, 1).contiguous()
        raise ValueError("Could not infer channel dimension for the input image.")

    def _preprocess(image: Image.Image | Tensor) -> Tensor:
        tensor = _to_chw_tensor(image)
        tensor = F.resize(tensor, size, antialias=True)
        tensor = F.normalize(tensor, mean=mean, std=std)
        return tensor

    return _preprocess


def load_dino_detr_pose(
    cfg: DinoDetrPoseLoadConfig,
) -> tuple[torch.nn.Module, Callable[[Image.Image | Tensor], Tensor], torch.nn.Module, torch.device]:
    """
    Instantiate the DETRPose model, load weights from a Lightning checkpoint, and
    return the model together with preprocessing and post-processing helpers.
    """

    ckpt_path = Path(cfg.checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_cfg = _load_model_cfg(cfg.model_config_path)
    device = _select_device(cfg.device)

    model = create_model(model_cfg).to(device)
    postprocessors = create_postprocessors(model_cfg)
    pose_post = postprocessors.get("pose")
    if pose_post is None:
        raise RuntimeError("Pose postprocessor not available in factory output.")

    raw_state = load_lightning_state_dict(str(ckpt_path))
    if cfg.remove_prefix:
        raw_state = strip_prefix(raw_state, cfg.remove_prefix)

    if cfg.allow_partial:
        align_and_load(model, raw_state, strict=cfg.strict)
    else:
        model.load_state_dict(raw_state, strict=cfg.strict)

    if cfg.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - optional optimisation
            print(f"[dino_detr_pose_loader] torch.compile failed: {exc}")

    model.eval()

    preprocess = preprocess_image_factory(
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        normalize_mean=cfg.normalize_mean,
        normalize_std=cfg.normalize_std,
    )

    return model, preprocess, pose_post.to(device), device


def rescale_keypoints_to_original(
    detections: Iterable[Mapping[str, Tensor]],
    processed_hw: Sequence[float] | Tensor,
    original_hw: Sequence[float] | Tensor,
) -> list[dict[str, Tensor]]:
    """Rescale keypoints from processed resolution back to the original image size."""

    proc_h, proc_w = _as_hw(processed_hw)
    orig_h, orig_w = _as_hw(original_hw)

    scale_x = float(orig_w) / float(proc_w)
    scale_y = float(orig_h) / float(proc_h)

    scaled: list[dict[str, Tensor]] = []
    for det in detections:
        keypoints = det["keypoints"]
        if not torch.is_tensor(keypoints):
            keypoints = torch.tensor(keypoints)
        kp = keypoints.view(-1, 3).clone()
        kp[:, 0] *= scale_x
        kp[:, 1] *= scale_y

        scaled.append({
            "scores": det["scores"].detach().clone() if torch.is_tensor(det["scores"]) else torch.tensor(det["scores"]),
            "labels": det["labels"].detach().clone() if torch.is_tensor(det["labels"]) else torch.tensor(det["labels"]),
            "keypoints": kp,
        })
    return scaled


def _as_hw(value: Sequence[float] | Tensor) -> tuple[float, float]:
    if torch.is_tensor(value):
        if value.numel() != 2:
            raise ValueError("Expected tensor with 2 elements for (H, W).")
        return float(value[0]), float(value[1])
    if len(value) != 2:
        raise ValueError("Expected sequence with 2 elements for (H, W).")
    return float(value[0]), float(value[1])


__all__ = [
    "DinoDetrPoseLoadConfig",
    "load_dino_detr_pose",
    "preprocess_image_factory",
    "rescale_keypoints_to_original",
]
