from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union, List

import torch
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover
    pl = None  # type: ignore


# ========== Types ==========
# A single image tensor: [3,H,W] (CHW) or [N,3,H,W] (NCHW). Values in 0..1.
ImageLike = torch.Tensor

# Renderer returns a dict:
#   {
#     "Input": Tensor([N,3,H,W]) or Tensor([3,H,W]) or { "sample_00": Tensor([3,H,W]), ... },
#     "Pred":  {"sample00_kp000": Tensor([3,H,W]), ...},
#     "GT":    {"sample00_kp000": Tensor([3,H,W]), ...},
#     "Overlay": Tensor([N,3,H,W]),
#     ...
#   }
RenderDict = Dict[str, Union[ImageLike, Dict[str, ImageLike]]]

# (buffer, step, stage) -> RenderDict
RenderFn = Callable[[Dict[str, torch.Tensor], int, str], RenderDict]


# ========== Helpers ==========
def _is_chw(x: torch.Tensor) -> bool:
    return x.dim() == 3 and x.size(0) in (1, 3)


def _is_nchw(x: torch.Tensor) -> bool:
    return x.dim() == 4 and x.size(1) in (1, 3)


def _to_chw3(x: torch.Tensor) -> torch.Tensor:
    """Coerce [H,W] or [1,H,W]/[3,H,W] to [3,H,W], clamp to 0..1."""
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [1,H,W]
    if x.dim() != 3:
        raise ValueError(f"Expected [H,W] or [C,H,W]; got {tuple(x.shape)}")
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    return x.clamp(0, 1)


# ========== Pluggable logger (renderer returns a dict of images) ==========
@dataclass
class HeatmapLoggerPluggable(pl.callbacks.Callback if pl else object):
    """Lightning callback that delegates *all drawing* to an injected renderer.

    The renderer returns a dict; this callback logs each entry under:
      "{stage}/{category}" or "{stage}/{category}/{subkey}"
    where stage is "Val" or "Test".

    Expected (conventional) tensors in step outputs (not enforced):
      images:          [N, C, H, W]
      pred_heatmaps:   [N, K, H, W]
      target_heatmaps: [N, K, H, W]
      bboxes:          varies
      class_labels:    varies
    """

    render_fn: RenderFn
    max_samples: int = 2  # keep first N samples per tensor
    take_first_batch_only: bool = True  # if False, accumulate all batches (first buffer is used by default)
    every_n_epochs: int = 1  # log every n epochs
    move_to_cpu: bool = True  # detach+cpu before storing
    clamp_outputs: bool = True  # clamp logged images to [0,1]

    # internal state
    _ready: bool = field(default=True, init=False)
    _buffers: List[Dict[str, torch.Tensor]] = field(default_factory=list, init=False)

    # ----- hooks -----
    def on_validation_epoch_start(self, trainer, pl_module):
        self._on_epoch_start(pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        self._on_epoch_start(pl_module)

    def _on_epoch_start(self, pl_module):
        epoch = getattr(pl_module, "current_epoch", 0)
        if (epoch % self.every_n_epochs) != 0:
            self._ready = False
            self._buffers.clear()
            return
        self._ready = True
        self._buffers.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._collect(outputs, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._collect(outputs, batch_idx)

    # ----- data collection -----
    def _collect(self, outputs, batch_idx: int):
        if not self._ready:
            return
        if not isinstance(outputs, dict):
            return

        buf: Dict[str, torch.Tensor] = {}
        for k, v in outputs.items():
            if not torch.is_tensor(v):
                continue
            vv = v
            if self.move_to_cpu:
                vv = v.detach().cpu()
            if vv.dim() >= 1 and vv.size(0) > self.max_samples:
                vv = vv[: self.max_samples]  # slice batch dimension
            buf[k] = vv

        if not buf:
            return

        self._buffers.append(buf)
        if self.take_first_batch_only and batch_idx == 0:
            self._ready = False  # stop collecting further batches

    # ----- epoch end -----
    def on_validation_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module, stage="Val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module, stage="Test")

    def _on_epoch_end(self, trainer, pl_module, stage: str):
        if not self._buffers:
            return
        logger = getattr(trainer, "logger", None)
        if logger is None or not hasattr(logger, "experiment"):
            return
        writer = logger.experiment  # TensorBoard SummaryWriter
        step = getattr(pl_module, "current_epoch", 0)

        # Strategy: use the first collected buffer (could be extended to merge)
        buf = self._buffers[0]

        # Delegate to renderer
        render_dict = self.render_fn(buf, step, stage)
        if not isinstance(render_dict, dict):
            raise TypeError("render_fn must return a dict[str, Tensor|dict[str, Tensor]].")

        for category, value in render_dict.items():
            base_tag = f"{stage}/{category}"
            if torch.is_tensor(value):  # CHW or NCHW
                self._log_tensor(writer, base_tag, value, step)
                continue
            if isinstance(value, dict):  # nested dict of images
                for subkey, img in value.items():
                    if not torch.is_tensor(img):
                        raise TypeError(f"Renderer returned non-tensor at {category}/{subkey}.")
                    self._log_tensor(writer, f"{base_tag}/{subkey}", img, step)
                continue

            raise TypeError(f"Renderer value for key '{category}' must be Tensor or dict[str, Tensor].")

        self._buffers.clear()

    # ----- logging helpers -----
    def _log_tensor(self, writer, tag: str, img: torch.Tensor, step: int):
        x = img.clamp(0, 1) if self.clamp_outputs else img
        if _is_chw(x):
            writer.add_image(tag, x, step, dataformats="CHW")
        elif _is_nchw(x):
            writer.add_images(tag, x, step)
        elif x.dim() == 2:  # [H,W] -> [3,H,W]
            writer.add_image(tag, _to_chw3(x), step, dataformats="CHW")
        else:
            raise ValueError(f"Unsupported image shape for tag={tag}: {tuple(x.shape)}")


# ========== Example renderers (dict-based) ==========
def render_heatmaps_dict(
    upsample_to: int = 512,
    normalize_each: bool = True,
    include_inputs: bool = True,
    input_key: str = "images",
    pred_key: str = "pred_heatmaps",
    gt_key: str = "target_heatmaps",
    pred_category: str = "Pred",
    gt_category: str = "GT",
    input_category: str = "Input",
) -> RenderFn:
    """Renderer: save ALL K heatmaps (GT/Pred) as big single images under dict categories."""

    def _norm(x: torch.Tensor) -> torch.Tensor:
        if not normalize_each:
            return x
        mn, mx = x.min(), x.max()
        if (mx - mn) > 1e-8:
            return (x - mn) / (mx - mn)
        return torch.zeros_like(x)

    def _upsample_hw(x: torch.Tensor, size: int) -> torch.Tensor:
        x2 = x.unsqueeze(0).unsqueeze(0)
        x2 = F.interpolate(x2, size=(size, size), mode="bilinear", align_corners=False)
        return x2.squeeze(0).squeeze(0)

    def _renderer(buf: Dict[str, torch.Tensor], step: int, stage: str) -> RenderDict:
        out: RenderDict = {}
        pred = buf.get(pred_key)
        gt = buf.get(gt_key)
        imgs = buf.get(input_key)

        if pred is None or gt is None:
            return out

        N, K, H, W = pred.shape
        pred_dict: Dict[str, ImageLike] = {}
        gt_dict: Dict[str, ImageLike] = {}

        for i in range(N):
            for k in range(K):
                hp = _norm(pred[i, k])
                ht = _norm(gt[i, k])
                hp = _upsample_hw(hp, upsample_to)
                ht = _upsample_hw(ht, upsample_to)
                pred_dict[f"sample_{i:02d}/kp_{k:03d}"] = _to_chw3(hp)
                gt_dict[f"sample_{i:02d}/kp_{k:03d}"] = _to_chw3(ht)

        out[pred_category] = pred_dict
        out[gt_category] = gt_dict

        if include_inputs and imgs is not None:
            ii = imgs
            if ii.dim() == 4:
                ii = F.interpolate(ii, size=(upsample_to, upsample_to), mode="bilinear", align_corners=False)
            out[input_category] = ii.clamp(0, 1)
        return out

    return _renderer


def render_overlay_max_dict(
    upsample_to: int = 512,
    alpha: float = 0.5,
    input_key: str = "images",
    pred_key: str = "pred_heatmaps",
    overlay_category: str = "Overlay",
) -> RenderFn:
    """Renderer: overlay max-pooled pred heatmap over input images and return as a category."""
    import matplotlib.cm as cm

    def _renderer(buf: Dict[str, torch.Tensor], step: int, stage: str) -> RenderDict:
        out: RenderDict = {}
        imgs = buf.get(input_key)
        pred = buf.get(pred_key)
        if imgs is None or pred is None:
            return out
        N, K, H, W = pred.shape

        vis = imgs.clamp(0, 1)
        vis = F.interpolate(vis, size=(upsample_to, upsample_to), mode="bilinear", align_corners=False)

        hp = pred.float().amax(dim=1)  # [N,H,W]
        hp = F.interpolate(
            hp.unsqueeze(1), size=(upsample_to, upsample_to), mode="bilinear", align_corners=False
        ).squeeze(1)

        overlays: List[torch.Tensor] = []
        for i in range(N):
            h = hp[i].clamp(0, 1).detach().cpu().numpy()
            color = cm.jet(h)[..., :3]  # [H,W,3] in 0..1
            color_t = torch.from_numpy(color).permute(2, 0, 1).to(vis.device, vis.dtype)
            ov = (1 - alpha) * vis[i] + alpha * color_t
            overlays.append(ov.clamp(0, 1))
        out[overlay_category] = torch.stack(overlays, dim=0)  # [N,3,H,W]
        return out

    return _renderer


def render_bboxes_on_images_dict(
    to_uint8: bool = True,
    tag: str = "BBoxes",
    image_key: str = "images",
    bboxes_key: str = "bboxes",
    labels_key: Optional[str] = "class_labels",
    assume_xyxy: bool = True,
) -> RenderFn:
    """Renderer: draw bboxes (+labels) on images; returns {tag: [N,3,H,W]} as float 0..1."""
    from torchvision.utils import draw_bounding_boxes

    def _renderer(buf: Dict[str, torch.Tensor], step: int, stage: str) -> RenderDict:
        out: RenderDict = {}
        imgs = buf.get(image_key)
        bboxes = buf.get(bboxes_key)
        labels = buf.get(labels_key) if (labels_key and labels_key in buf) else None
        if imgs is None or bboxes is None:
            return out

        N = imgs.size(0)
        drawn: List[torch.Tensor] = []
        for i in range(N):
            img = imgs[i]
            if img.size(0) == 1:
                img = img.repeat(3, 1, 1)
            img_u8 = (img.clamp(0, 1) * 255).to(torch.uint8) if to_uint8 else img
            boxes = bboxes[i]
            if not assume_xyxy and boxes.numel() > 0:
                # convert xywh -> xyxy
                x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                boxes = torch.stack([x, y, x + w, y + h], dim=1)
            lbls = None
            if labels is not None:
                lbls = [str(int(x)) for x in labels[i].tolist()]
            vis = draw_bounding_boxes(img_u8, boxes, labels=lbls, colors="red", width=2)
            drawn.append((vis.float() / 255.0) if to_uint8 else vis.clamp(0, 1))
        out[tag] = torch.stack(drawn, dim=0)
        return out

    return _renderer


__all__ = [
    "HeatmapLoggerPluggable",
    "render_heatmaps_dict",
    "render_overlay_max_dict",
    "render_bboxes_on_images_dict",
]
