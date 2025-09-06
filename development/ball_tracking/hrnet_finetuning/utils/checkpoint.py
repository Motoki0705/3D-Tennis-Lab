from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Dict, Iterable, Tuple

import torch


def load_checkpoint_object(path: str) -> Any:
    """
    Load a checkpoint file (.pth, .pth.tar, .ckpt) with torch.load.

    Returns the raw object stored in the file.
    """
    abs_path = path
    if not os.path.isabs(abs_path):
        # Keep relative path behavior consistent with callers if needed
        abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Checkpoint not found: {abs_path}")
    return torch.load(abs_path, map_location="cpu")


def extract_state_dict(ckpt_obj: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Extract a PyTorch state_dict from various checkpoint formats.

    Supports common layouts:
      - { 'state_dict': {...} }
      - { 'model_state_dict': {...} }
      - { 'model': {...} }
      - { 'net': {...} }
      - { 'weights': {...} }
      - plain state_dict (mapping of tensor name -> tensor)

    Returns (state_dict, meta).
    """
    meta: Dict[str, Any] = {
        "root_type": type(ckpt_obj).__name__,
        "root_keys": [],
        "selected_key": None,
    }

    if isinstance(ckpt_obj, (dict, OrderedDict)):
        meta["root_keys"] = list(ckpt_obj.keys())
        for key in [
            "state_dict",
            "model_state_dict",
            "model_state",
            "model",
            "net",
            "weights",
        ]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], (dict, OrderedDict)):
                inner = ckpt_obj[key]
                if any(isinstance(v, torch.Tensor) for v in inner.values()):
                    meta["selected_key"] = key
                    return dict(inner), meta
        # Plain mapping of tensors
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return dict(ckpt_obj), meta

    raise ValueError("Unsupported checkpoint format: could not locate a state_dict-like mapping")


def strip_prefix_from_state_dict(
    state_dict: Dict[str, torch.Tensor], prefixes: Iterable[str] = ("model.", "module.")
) -> Dict[str, torch.Tensor]:
    """
    Remove known prefixes from parameter names in a state_dict.
    """
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
                break
        cleaned[nk] = v
    return cleaned


def load_model_weights(
    model: torch.nn.Module, ckpt_path: str, strict: bool = False
) -> Tuple[Iterable[str], Iterable[str]]:
    """
    Load weights from a checkpoint file into a model.

    - Supports both .pth/.pth.tar and PyTorch Lightning .ckpt files.
    - Strips common wrapper prefixes like 'model.' and 'module.'.

    Returns (missing_keys, unexpected_keys) from `load_state_dict`.
    """
    obj = load_checkpoint_object(ckpt_path)
    state_dict, _ = extract_state_dict(obj)
    state_dict = strip_prefix_from_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return missing, unexpected


__all__ = [
    "load_checkpoint_object",
    "extract_state_dict",
    "strip_prefix_from_state_dict",
    "load_model_weights",
]
