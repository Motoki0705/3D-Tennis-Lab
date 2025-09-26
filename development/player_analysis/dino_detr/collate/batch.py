# collate_seq_t1.py
from __future__ import annotations
from typing import List, Dict, Any
import torch


def _to_tensor_xyxy(x):
    """
    x: list[list[4]] or Tensor[M,4] (xyxy)
    return: Tensor[M,4] float32
    """
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x, dtype=torch.float32) if len(x) > 0 else torch.zeros((0, 4), dtype=torch.float32)
    if t.numel() == 0:
        return t.to(dtype=torch.float32)
    # 最低限の形/型
    t = t.to(dtype=torch.float32)
    assert t.ndim == 2 and t.shape[1] == 4, f"boxes must be [M,4], got {tuple(t.shape)}"
    return t


def _to_tensor_labels(x):
    """
    x: list[int] or Tensor[M]
    return: Tensor[M] int64
    """
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor([int(v) for v in x], dtype=torch.int64) if len(x) > 0 else torch.zeros((0,), dtype=torch.int64)
    return t.to(dtype=torch.int64)


def collate_seq_t1(batch: List[Dict[str, Any]]):
    """
    batch: list of samples from PlayerSequenceDataset
      each sample:
        {
          "inputs":  Tensor[T,C,H,W]  (T==1を想定)
          "targets": {
              # ここは _finalize_sample で xyxy/Tensor まで整っている想定
              "boxes":  list[T] of Tensor[Mi,4]  (xyxy)
              "labels": list[T] of Tensor[Mi]
          }
          "metadata": {...}  # 任意
        }

    returns:
      images:  Tensor[B,C,H,W]
      targets: list[{"boxes": Tensor[Mi,4], "labels": Tensor[Mi]}]  (len B)
    """
    B = len(batch)
    # ---- 画像を[T,C,H,W] -> [C,H,W]へ（T=1想定）
    imgs = []
    tgts = []
    for sample in batch:
        inp = sample["inputs"]
        assert inp.ndim == 4, f"inputs must be [T,C,H,W], got {tuple(inp.shape)}"
        assert inp.shape[0] == 1, f"T must be 1 for this collate, got T={inp.shape[0]}"
        img = inp[0]  # [C,H,W]
        imgs.append(img)

        # targets: list[T] -> pick t=0
        t_boxes_seq = sample["targets"].get("boxes", [])
        t_labels_seq = sample["targets"].get("labels", [])
        boxes_t0 = t_boxes_seq[0] if len(t_boxes_seq) > 0 else []
        labels_t0 = t_labels_seq[0] if len(t_labels_seq) > 0 else []

        boxes = _to_tensor_xyxy(boxes_t0)  # [Mi,4] float32
        labels = _to_tensor_labels(labels_t0)  # [Mi] int64
        tgts.append({"boxes": boxes, "labels": labels})

    images = torch.stack(imgs, dim=0).contiguous()  # [B,C,H,W]
    return images, tgts


__all__ = ["collate_seq_t1"]
