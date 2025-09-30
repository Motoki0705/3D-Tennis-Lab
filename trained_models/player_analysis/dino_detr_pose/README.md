# DINO-DETR Pose Inference

This folder packages everything needed to rebuild the DINOv3 + DETRPose model
used in the `player_analysis/dino_detr_pose` experiment and run it for
inference. The loader mirrors the other utilities under
`trained_models/player_analysis` so you can mix-and-match detection and pose
components.

Contents:

- `config.yaml` – dataclass-friendly config used by the loader. Point the
  `checkpoint_path` field at your Lightning `.ckpt` file.
- `dino_detr_pose_loader.py` – functions to instantiate the pure DETRPose model
  and load weights, plus preprocessing and rescaling helpers.

## 1. Prerequisites

- Python 3.10+
- PyTorch (CUDA strongly recommended, CPU works for debugging)
- `torchvision` for basic tensor image transforms
- The repo-level dependencies required by
  `development.player_analysis.dino_detr_pose` (notably DINOv3 weights under
  `third_party/dinov3/weights/`)
- A Lightning checkpoint produced by the DINO-DETR pose training recipe

## 2. Configure the loader

```yaml
# trained_models/player_analysis/dino_detr_pose/config.yaml
checkpoint_path: "/path/to/your/dino_detr_pose.ckpt"
model_config_path: "development/player_analysis/dino_detr_pose/configs/model_cfg/dino_detr_pose.yaml"
device: "cuda" # or "cpu"/"mps"
strict: false # set true if you expect an exact key match
remove_prefix: "model." # LightningModule prefix
allow_partial: true # load intersecting keys only
image_height: 320 # must match training resolution
image_width: 640
normalize_mean: [0.485, 0.456, 0.406]
normalize_std: [0.229, 0.224, 0.225]
torch_compile: false
```

Update at least `checkpoint_path` (and optionally `device`). Leave the other
values unless you intentionally changed the training resolution or
normalisation.

## 3. Load the model

```python
from pathlib import Path
import torch
from PIL import Image

from trained_models.player_analysis.dino_detr_pose.dino_detr_pose_loader import (
    DinoDetrPoseLoadConfig,
    load_dino_detr_pose,
    rescale_keypoints_to_original,
)

# Read and tweak the YAML config
cfg = DinoDetrPoseLoadConfig.from_yaml(
    Path("trained_models/player_analysis/dino_detr_pose/config.yaml")
)
# cfg.checkpoint_path = "/absolute/path/to/epoch=9-step=500.ckpt"

model, preprocess, postprocessor, device = load_dino_detr_pose(cfg)
```

The loader returns:

1. The pure DETRPose network (`model`) on the requested device.
2. A preprocessing callable (`preprocess`) that mirrors the training pipeline.
3. The DETRPose postprocessor (`postprocessor`) for decoding logits into
   keypoints.
4. The selected `torch.device` (`device`).

## 4. Single image inference (step-by-step)

```python
# 1) Load your RGB frame and remember the original size (H, W)
image = Image.open("sample_frame.jpg").convert("RGB")
orig_hw = image.height, image.width

# 2) Preprocess and create a batch
processed = preprocess(image)
tensor = processed.unsqueeze(0).to(device)

# 3) Forward pass
model.eval()
with torch.inference_mode():
    outputs = model(tensor)

# 4) Decode detections at the processed resolution
proc_hw = torch.tensor([[tensor.shape[-2], tensor.shape[-1]]], device=device)
detections = postprocessor(outputs, proc_hw)

# 5) Optionally map keypoints back to the original resolution
scaled_detections = rescale_keypoints_to_original(
    detections,
    processed_hw=proc_hw[0],
    original_hw=orig_hw,
)

# 6) Access the first detection
if scaled_detections:
    det0 = scaled_detections[0]
    keypoints = det0["keypoints"].cpu().numpy()   # shape: [K, 3] -> (x, y, visibility)
    score = det0["scores"][0].item()
    print("Score:", score)
    print("First keypoint (x, y, v):", keypoints[0])
else:
    print("No players detected.")
```

### Notes on the outputs

- `postprocessor` selects the top-`num_select` queries (default 60) and returns a
  list of dictionaries. Each dictionary corresponds to one player proposal with
  keys `scores`, `labels`, and `keypoints` (flattened `num_keypoints * 3`).
- The helper `rescale_keypoints_to_original` rescales the `(x, y)` coordinates
  from the processed lattice (`image_height` × `image_width`) back to your
  original frame size while preserving the score/visibility values.
- Keep the processed resolution and normalisation consistent with training; the
  DINO backbone was fine-tuned on 320×640 RGB inputs with ImageNet statistics.

## 5. Batching and downstream usage

- For batched inference, stack multiple preprocessed tensors into the same batch
  before calling `model`. Keep track of the original `(H, W)` for each sample so
  you can rescale keypoints individually.
- Downstream consumers (e.g. trajectory filters) generally expect COCO-style
  keypoints. The loader already outputs `(x, y, visibility)` triples per joint,
  so you can serialise them directly or feed them into evaluation scripts.
- When mixing with the RT-DETR detector, run detection first, crop/associate the
  players, then feed the crops or track IDs into whichever pipeline you prefer.

## 6. Troubleshooting

- **`FileNotFoundError: Checkpoint not found`** – ensure the path in
  `config.yaml` points to a real `.ckpt` file. Relative paths are resolved from
  the repo root.
- **`RuntimeError: Pose postprocessor not available`** – verify your
  `model_config_path` still describes the standard DETRPose architecture.
- **Key mismatch warnings** – keep `allow_partial=true` unless your checkpoint
  exactly matches the freshly instantiated model.

With this setup you can reproduce the full DINO-DETR pose inference pipeline in
Python without relying on Hydra or the Lightning training scripts.
