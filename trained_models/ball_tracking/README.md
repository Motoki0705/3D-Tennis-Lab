# Ball Tracking Inference

This directory packages everything required to reuse the HRNet-based tennis ball
tracker that lives under `development/ball_tracking`. The
`hrnet/hrnet_loader.py` helper mirrors the other trained-model loaders: it
composes the copied Hydra configs, instantiates the TrackNetV2 detector and
tracker from WASB-SBDT, and returns ready-to-use objects.

## Prerequisites

- Python 3.10+
- PyTorch with CUDA support (the detector only runs on GPU)
- OpenCV, NumPy, and PIL (already used throughout the repository)
- A TrackNetV2 checkpoint (`.pth.tar`) exported during training

## Loading the detector stack

```python
from collections import deque

import cv2
import torch
from PIL import Image

from trained_models.ball_tracking.hrnet.hrnet_loader import (
    HRNetLoadConfig,
    load_hrnet_with_ckpt,
)

from dataloaders.dataset_loader import get_transform as build_affine_transform

cfg = HRNetLoadConfig(
    checkpoint_path="/path/to/best_model.pth.tar",
)
detector, tracker, transform, device, resolved_cfg = load_hrnet_with_ckpt(cfg)

tracker.refresh()

frames_in = resolved_cfg.model.frames_in
input_wh = (resolved_cfg.model.inp_width, resolved_cfg.model.inp_height)
output_wh = (resolved_cfg.model.out_width, resolved_cfg.model.out_height)
out_scales = resolved_cfg.model.out_scales

frame_buffer = deque(maxlen=frames_in)
cap = cv2.VideoCapture("match_clip.mp4")

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    trans_input = build_affine_transform(frame_bgr, input_wh)
    warped = cv2.warpAffine(frame_bgr, trans_input, input_wh, flags=cv2.INTER_LINEAR)

    # detector expects RGB tensors normalised like training data
    pil_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_img)
    frame_buffer.append(tensor)

    if len(frame_buffer) < frames_in:
        continue

    clip = torch.cat(list(frame_buffer), dim=0).unsqueeze(0).to(device)

    # Prepare inverse transforms for post-processing
    trans_outputs = {}
    out_w, out_h = output_wh
    for scale in out_scales:
        trans_output_inv = build_affine_transform(frame_bgr, (out_w, out_h), inv=1)
        trans_outputs[int(scale)] = torch.tensor(
            trans_output_inv,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        out_w = max(out_w // 2, 1)
        out_h = max(out_h // 2, 1)

    batch_results, _ = detector.run_tensor(clip, trans_outputs)
    preds = batch_results[0][frames_in - 1]

    tracked = tracker.update(preds)
    if tracked and tracked["visi"]:
        print(f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}: (x, y) = ({tracked['x']:.1f}, {tracked['y']:.1f})")
```

### Notes

- `build_affine_transform` is the same helper used in the original runner.
  It warps frames to the detector's expected 512x288 input while providing the
  inverse matrices needed to map predictions back to the source resolution.
- `tracker.update` converts the detector heatmaps into a single ball
  trajectory; when `tracked["visi"]` is `True` the coordinates are in the
  original image space.
- For writing annotated videos replicate the overlay logic from
  `development/ball_tracking/hrnet/runners/detect.py`.
