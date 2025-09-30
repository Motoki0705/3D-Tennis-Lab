# Player Analysis Inference

This directory packages three complementary loaders:

1. `rt_detr/rtdetr_loader.py` – person detection using an RT-DETR Lightning
   checkpoint.
2. `vit_pose/vit_pose_loader.py` – downstream pose estimation with a
   ViT-Pose model pulled from the Hugging Face Hub.
3. `dino_detr_pose/dino_detr_pose_loader.py` – direct end-to-end player pose
   estimation with the DINOv3 + DETRPose model fine-tuned in this repository.

Together they reproduce the detection + pose pipeline used for player
analytics, and now also expose the single-stage DINO-DETR pose baseline.

## Prerequisites

- Python 3.10+
- PyTorch with CUDA for best performance (both loaders fall back to CPU when
  necessary)
- `transformers` >= 4.39 for ViT-Pose
- Access to the RT-DETR Lightning checkpoint trained on your dataset

## Person detection with RT-DETR

```python
from pathlib import Path

import torch
from PIL import Image

from trained_models.player_analysis.rt_detr.rtdetr_loader import (
    RTDetrLoadConfig,
    load_hf_rtdetr_with_ckpt,
)

cfg = RTDetrLoadConfig.from_yaml(
    Path("trained_models/player_analysis/rt_detr/configs/rtdetr_config.yaml")
)
cfg.checkpoint_path = "/path/to/lightning.ckpt"

model, processor, device = load_hf_rtdetr_with_ckpt(cfg)

image = Image.open("frame.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)

model.eval()
with torch.inference_mode():
    outputs = model(**inputs)

target_size = torch.tensor([[image.height, image.width]], device=device)
results = processor.post_process_object_detection(
    outputs,
    threshold=0.5,
    target_sizes=target_size,
)[0]

boxes_xyxy = []
scores = []
for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
    if label.item() == 0:  # 0 = person for COCO-trained RT-DETR
        boxes_xyxy.append(box.cpu())
        scores.append(float(score))

if boxes_xyxy:
    boxes_xyxy = torch.stack(boxes_xyxy)
    boxes_xywh = torch.stack(
        [
            boxes_xyxy[:, 0],
            boxes_xyxy[:, 1],
            boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
        ],
        dim=1,
    )
else:
    boxes_xywh = torch.empty((0, 4))

boxes_np = boxes_xywh.numpy()
print("Detected players (COCO x, y, w, h):", boxes_np)
```

## Pose estimation with ViT-Pose

```python
from trained_models.player_analysis.vit_pose.vit_pose_loader import (
    PoseLoadConfig,
    load_pose_from_hub,
)

pose_cfg = PoseLoadConfig.from_yaml(
    "trained_models/player_analysis/vit_pose/configs/vit_pose_config.yaml"
)
pose_model, pose_processor, pose_device = load_pose_from_hub(pose_cfg)

pose_model.eval()
with torch.inference_mode():
    pose_inputs = pose_processor(
        image,
        boxes=[boxes_np],
        return_tensors="pt",
    )
    pose_inputs = {k: v.to(pose_device) if hasattr(v, "to") else v for k, v in pose_inputs.items()}
    pose_outputs = pose_model(**pose_inputs)

pred_keypoints = pose_processor.post_process_pose_estimation(
    pose_outputs,
    boxes=[boxes_np],
)[0]

print("Pose results for first player:", pred_keypoints[0])
```

### Combining both stages

- Run RT-DETR first to obtain person detections per frame.
- Convert the `results["boxes"]` tensors from `(x_min, y_min, x_max, y_max)`
  to COCO `(x, y, w, h)` via `boxes[:, 2:] - boxes[:, :2]` before passing
  them to `pose_processor`. The snippet above stores both forms in
  `boxes_xyxy` and `boxes_np`.
- Ensure both models use the same device if you intend to fuse results; moving
  RT-DETR outputs to the pose model's device avoids unnecessary copies.
- For videos, wrap the snippets in a loop and batch frames where possible to
  keep GPU utilisation high.

## Direct pose estimation with DINO-DETR Pose

If you prefer to skip the two-stage pipeline, the
`trained_models/player_analysis/dino_detr_pose` folder contains a loader and
README that walk through rebuilding the DINOv3 + DETRPose network from a
Lightning checkpoint. The helper returns the pure model, preprocessing, and the
official post-processor so you can obtain keypoints in a handful of lines:

```python
from trained_models.player_analysis.dino_detr_pose.dino_detr_pose_loader import (
    DinoDetrPoseLoadConfig,
    load_dino_detr_pose,
)

cfg = DinoDetrPoseLoadConfig.from_yaml(
    "trained_models/player_analysis/dino_detr_pose/config.yaml"
)
model, preprocess, postprocessor, device = load_dino_detr_pose(cfg)
```

Refer to the dedicated README for a detailed inference walkthrough, including
rescaling utilities to map predictions back to the original image resolution.
