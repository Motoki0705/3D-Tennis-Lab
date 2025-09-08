# rtdetr_hf_adapter.py
from __future__ import annotations
from typing import Any, Dict, List

import torch
import torch.nn as nn
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    return torch.stack([x1, y1, w, h], dim=-1)


def _cxcywh_norm_to_xywh_abs(cxcywh: torch.Tensor, size_hw: torch.Tensor) -> torch.Tensor:
    """cx,cy,w,h are in [0,1]; size_hw is (H,W). Return xywh in absolute pixels."""
    H, W = size_hw.to(cxcywh.device, cxcywh.dtype)
    cx, cy, w, h = cxcywh.unbind(-1)
    x = (cx - w * 0.5) * W
    y = (cy - h * 0.5) * H
    return torch.stack([x, y, w * W, h * H], dim=-1)


class HFRTDetrWrapper(nn.Module):
    """
    Adapter that matches your training signature:
      forward(images: List[Tensor[C,H,W]], targets: List[dict]) -> Dict[str, Tensor]

    Supported target box formats via cfg.data.boxes_format:
      - "xyxy"           (absolute pixels)
      - "xywh" | "coco"  (absolute pixels, COCO TLWH)
      - "cxcywh_norm"    (normalized to [0,1] using post-aug image size)
    """

    def __init__(self, num_classes: int, cfg: Any):
        super().__init__()
        ckpt = str(getattr(cfg.model, "hf_checkpoint", "PekingU/rtdetr_r50vd"))
        self.model = RTDetrForObjectDetection.from_pretrained(ckpt)
        self.processor = RTDetrImageProcessor.from_pretrained(ckpt)

        self.model.config.num_labels = int(num_classes)

        id2label = getattr(cfg.model, "id2label", None)
        if id2label:
            self.model.config.id2label = {int(k): v for k, v in dict(id2label).items()}
            self.model.config.label2id = {v: k for k, v in self.model.config.id2label.items()}

        data_cfg = getattr(cfg, "data", {})
        self.boxes_format = str(getattr(data_cfg, "boxes_format", "xyxy")).lower()
        self.images_are_0_1 = bool(getattr(data_cfg, "images_are_0_1", True))
        self.labels_are_one_based = bool(getattr(data_cfg, "labels_are_one_based", False))

    @torch.no_grad()
    def _maybe_to_uint8(self, img: torch.Tensor) -> torch.Tensor:
        # If tensors are [0,1] float, we pass do_rescale=False to the processor.
        return img

    def _to_hf_ann(self, target: Dict) -> Dict:
        if "boxes" not in target or "labels" not in target:
            raise ValueError("targets must contain 'boxes' and 'labels'")

        # image_id may be int or a 1-element tensor in your dataset
        image_id = target.get("image_id", 0)
        if isinstance(image_id, torch.Tensor):
            image_id = int(image_id.view(-1)[0].item())
        else:
            image_id = int(image_id)

        boxes = target["boxes"]
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = target["labels"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels, dtype=torch.long)
        if self.labels_are_one_based:
            labels = labels - 1  # map 1→0

        # Convert to absolute COCO xywh
        if self.boxes_format in {"xywh", "coco"}:
            boxes_xywh = boxes
        elif self.boxes_format == "xyxy":
            boxes_xywh = _xyxy_to_xywh(boxes)
        elif self.boxes_format == "cxcywh_norm":
            # Need image (post-aug) size from target["size"] (H,W). Fallback to orig_size if needed.
            size = target.get("size", target.get("orig_size", None))
            if size is None:
                raise ValueError("For cxcywh_norm, target must include 'size' or 'orig_size' (H,W)")
            size_hw = size.to(boxes.device, boxes.dtype)
            if size_hw.numel() == 2:
                size_hw = size_hw.view(1, 2).expand(boxes.shape[0], 2)
            boxes_xywh = _cxcywh_norm_to_xywh_abs(boxes, size_hw[0])
        else:
            raise ValueError(f"Unsupported boxes_format: {self.boxes_format}")

        ann_list = []
        for b, l in zip(boxes_xywh, labels):
            x, y, w, h = [float(v) for v in b.tolist()]
            area = max(w, 0.0) * max(h, 0.0)
            ann_list.append({
                "bbox": [x, y, w, h],
                "category_id": int(l),
                "area": area,
                "iscrowd": 0,
            })
        return {"image_id": image_id, "annotations": ann_list}

    def forward(self, images: List[torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        anns = [self._to_hf_ann(t) for t in targets]
        imgs = [self._maybe_to_uint8(im) for im in images]

        inputs = self.processor(
            images=imgs,
            annotations=anns,
            return_tensors="pt",
            do_rescale=not self.images_are_0_1,  # False for [0,1] floats, True for [0,255]
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = self.model(**inputs)  # computes internal loss if `labels` present

        losses = {"loss": out.loss}
        if getattr(out, "loss_dict", None):
            for k, v in out.loss_dict.items():
                losses[f"loss_{k}"] = v if isinstance(v, torch.Tensor) else torch.tensor(v, device=out.loss.device)
        return losses
