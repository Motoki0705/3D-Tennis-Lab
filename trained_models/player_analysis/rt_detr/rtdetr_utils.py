# rtdetr_util.py
"""
Utilities for adapting a PyTorch Lightning checkpoint to a pure Transformers RT-DETR model.

What this gives you
-------------------
- load_lightning_state_dict(ckpt_path): read a .ckpt and return the raw 'state_dict' (OrderedDict).
- strip_prefix(state_dict, prefix="model."): remove "model." from keys saved by a LightningModule wrapper.
- align_and_load(hf_model, state_dict, strict=False): load only intersecting keys into the HF model and
  print a concise report of loaded / missing / unexpected params.

Typical usage
-------------
from transformers import RTDetrForObjectDetection
from rtdetr_util import load_lightning_state_dict, strip_prefix, align_and_load

hf_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd", num_labels=1)
raw_sd = load_lightning_state_dict("path/to/epoch=9-step=500.ckpt")
adapted_sd = strip_prefix(raw_sd, prefix="model.")
align_and_load(hf_model, adapted_sd, strict=False)
"""

from __future__ import annotations

from collections import OrderedDict

import torch


def load_lightning_state_dict(ckpt_path: str) -> OrderedDict[str, torch.Tensor]:
    """Load the 'state_dict' from a PyTorch Lightning .ckpt file."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in ckpt:
        # Some checkpoints (non-Lightning) may store weights directly
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return OrderedDict(ckpt)
        raise KeyError(f"'state_dict' not found in checkpoint: {ckpt_path}")
    sd = ckpt["state_dict"]
    # Ensure OrderedDict for stable ordering
    return OrderedDict(sd)


def strip_prefix(state_dict: OrderedDict[str, torch.Tensor], prefix: str = "model.") -> OrderedDict[str, torch.Tensor]:
    """
    Remove a leading prefix (e.g., 'model.') from all keys. Keys not starting with the prefix are kept.
    """
    out = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _partition_keys(
    target_sd: dict[str, torch.Tensor],
    src_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], set, set]:
    """
    Split keys into: matching, missing_in_src (need by target but absent in src), unexpected_in_src.
    """
    target_keys = set(target_sd.keys())
    src_keys = set(src_sd.keys())
    intersect = target_keys & src_keys
    missing = target_keys - src_keys
    unexpected = src_keys - target_keys
    filtered = {k: src_sd[k] for k in intersect}
    return filtered, missing, unexpected


def align_and_load(
    hf_model,
    src_state_dict: dict[str, torch.Tensor],
    strict: bool = False,
) -> None:
    """
    Load `src_state_dict` into `hf_model` with a helpful report.

    - Filters to the intersection of keys.
    - If `strict=False` (recommended), mismatched / extra keys are ignored.
    - If `strict=True`, raises if some keys are missing/unexpected.
    """
    tgt_sd = hf_model.state_dict()
    filtered, missing, unexpected = _partition_keys(tgt_sd, src_state_dict)

    msg = [
        "[rtdetr_util] Loading weights into HF model:",
        f"  total target params: {len(tgt_sd)}",
        f"  provided params:     {len(src_state_dict)}",
        f"  will load (match):   {len(filtered)}",
        f"  missing in src:      {len(missing)}",
        f"  unexpected in src:   {len(unexpected)}",
    ]
    print("\n".join(msg))

    # Optionally, you may want to warn about obvious head-size mismatches here.

    load_result = hf_model.load_state_dict(filtered, strict=strict)

    # Summarize torch's own report (for completeness)
    if hasattr(load_result, "missing_keys") and hasattr(load_result, "unexpected_keys"):
        print("[rtdetr_util] torch.load_state_dict report:")
        print(f"  missing_keys:   {len(load_result.missing_keys)}")
        print(f"  unexpected_keys:{len(load_result.unexpected_keys)}")
