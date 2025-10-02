# Trained Model Loaders

Utilities in this directory package trained checkpoints so they can be reused without diving back into
`development/` pipelines. Each subdirectory bundles lightweight loaders, configs, and documentation that make a
single model (or model pair) ready for inference.

## Directory map

- `ball_tracking/hrnet/` – HRNet + TrackNetV2 tennis ball tracker copied from `development/ball_tracking`.
- `court_pose/dino_fpn/` – COAT (DINO backbone + FPN head) court keypoint estimator (v1).
- `court_pose/dino_fpn_v2/` – DINOv3 + FPN upgrade with Lightning-compatible loader.
- `player_analysis/rt_detr/` – RT-DETR Lightning checkpoint adapter for player detection.
- `player_analysis/vit_pose/` – ViT-Pose loader that pulls weights from the Hugging Face Hub.

Activate the project environment (`source .venv-wsl/bin/activate`) before running any of these loaders. PyTorch 2.3
with CUDA 12.x is the expected baseline; individual sections call out extra dependencies.

## Ball tracking (HRNet + TrackNetV2)

Helper: `trained_models/ball_tracking/hrnet/hrnet_loader.py`

Prerequisites:

- Python 3.10+
- CUDA-capable PyTorch install (detector requires GPU)
- OpenCV, NumPy, Pillow, Hydra
- TrackNetV2 checkpoint (`.pth.tar`)

```python
from trained_models.ball_tracking.hrnet.hrnet_loader import HRNetLoadConfig, load_hrnet_with_ckpt

cfg = HRNetLoadConfig(
    checkpoint_path="/abs/path/to/best_model.pth.tar",
)
detector, tracker, transform, device, resolved_cfg = load_hrnet_with_ckpt(cfg)
```

The loader composes the bundled Hydra configs, validates CUDA availability, rebuilds the WASB TrackNetV2 modules, and
returns `(detector, tracker, transform, device, resolved_cfg)`. See `trained_models/ball_tracking/README.md` for a
complete streaming inference example.

## Court pose estimation (COAT)

Helpers:

- `trained_models/court_pose/dino_fpn/dino_fpn_loader.py`
- `trained_models/court_pose/dino_fpn_v2/dino_fpn_v2_loader.py`

Prerequisites:

- Python 3.10+
- PyTorch (CUDA optional but recommended)
- `torchvision`, Pillow
- Lightning checkpoint compatible with the copied model definition

```python
from pathlib import Path

from trained_models.court_pose.dino_fpn.dino_fpn_loader import CoatLoadConfig, load_coat_with_ckpt
from trained_models.court_pose.dino_fpn_v2.dino_fpn_v2_loader import (
    DinoFpnV2LoadConfig,
    load_dino_fpn_v2_with_ckpt,
)

v1_cfg = CoatLoadConfig.from_yaml(Path("trained_models/court_pose/dino_fpn/config.yaml"))
v1_cfg.checkpoint_path = "/abs/path/to/v1_lightning.ckpt"
model_v1, transform_v1, device_v1 = load_coat_with_ckpt(v1_cfg)

v2_cfg = DinoFpnV2LoadConfig.from_yaml("trained_models/court_pose/dino_fpn_v2/config.yaml")
v2_cfg.checkpoint_path = "/abs/path/to/v2_lightning.ckpt"
model_v2, transform_v2, device_v2 = load_dino_fpn_v2_with_ckpt(v2_cfg)
```

Both loaders normalise inputs with ImageNet statistics and align Lightning checkpoints using their local utility
modules. The v2 variant expects access to the `third_party/dinov3` repo because it initialises the DINOv3 backbone via
`torch.hub.load`. Replace the argmax post-processing in the per-directory README with soft-argmax or Gaussian peak
fitting for higher accuracy.

## Player analysis (RT-DETR + ViT-Pose)

Helpers:

- `trained_models/player_analysis/rt_detr/rtdetr_loader.py`
- `trained_models/player_analysis/vit_pose/vit_pose_loader.py`

Prerequisites:

- Python 3.10+
- PyTorch with CUDA for best performance (both fall back to CPU)
- `transformers >= 4.39`
- RT-DETR Lightning checkpoint (for detection stage)

```python
from trained_models.player_analysis.rt_detr.rtdetr_loader import RTDetrLoadConfig, load_hf_rtdetr_with_ckpt
from trained_models.player_analysis.vit_pose.vit_pose_loader import PoseLoadConfig, load_pose_from_hub

rt_cfg = RTDetrLoadConfig.from_yaml("trained_models/player_analysis/rt_detr/config.yaml")
rt_cfg.checkpoint_path = "/abs/path/to/epoch=9-step=500.ckpt"
rt_model, rt_processor, rt_device = load_hf_rtdetr_with_ckpt(rt_cfg)

pose_cfg = PoseLoadConfig.from_yaml("trained_models/player_analysis/vit_pose/config.yaml")
pose_model, pose_processor, pose_device = load_pose_from_hub(pose_cfg)
```

Run RT-DETR first to produce person boxes, convert them to COCO `(x, y, w, h)` format, and feed them into ViT-Pose.
Both loaders print concise reports when aligning checkpoints so you can quickly verify which parameters were loaded.
See `trained_models/player_analysis/README.md` for a complete end-to-end example.
