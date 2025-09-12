from __future__ import annotations

from typing import Dict, Any, Optional

import os
import torch
from omegaconf import DictConfig

from ..model.dino_faster_rcnn import DinoFasterRCNN


def build_model_from_cfg(cfg: DictConfig) -> torch.nn.Module:
    """Constructs the model from cfg.model, and if cfg.ckpt_path is set, loads weights.

    Returns a torch.nn.Module ready for eval/train as the caller decides.
    """
    model = DinoFasterRCNN(**cfg.model)
    ckpt_path = _resolve_ckpt_path(cfg)
    if ckpt_path:
        loaded = load_checkpoint_into_model(model, ckpt_path)
        msg = f"Loaded checkpoint into model: {ckpt_path}" if loaded else f"Failed to load checkpoint: {ckpt_path}"
        print(msg)
    return model


def _resolve_ckpt_path(cfg: DictConfig) -> Optional[str]:
    # Prefer top-level ckpt_path
    ckpt = cfg.get("ckpt_path", None)
    if ckpt:
        return ckpt
    # Fallbacks
    if hasattr(cfg, "infer"):
        ckpt = cfg.infer.get("ckpt_path", None)
        if ckpt:
            return ckpt
    return None


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str) -> bool:
    """Loads weights from a checkpoint into a plain torch.nn.Module.

    Supports PyTorch Lightning checkpoints (expects `state_dict`) and raw state dicts.
    Tries common prefixes like `model.`, `module.` and strips them as needed.
    Returns True on a best-effort successful load, False otherwise.
    """
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[model_io] Checkpoint not found: {ckpt_path}")
        return False

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state: Dict[str, Any]
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # Might already be a state_dict
        state = ckpt  # type: ignore[assignment]
    else:
        print("[model_io] Unsupported checkpoint format.")
        return False

    # Try stripping common prefixes
    prefixes = [
        "model.model.",
        "model.",
        "module.",
        "net.",
        "",  # as-is
    ]
    model_keys = set(model.state_dict().keys())
    for p in prefixes:
        remapped = {k[len(p) :] if p and k.startswith(p) else k: v for k, v in state.items()}
        overlap = model_keys.intersection(remapped.keys())
        if not overlap:
            continue
        try:
            missing, unexpected = _load_state_dict_forgiving(model, remapped)
            if len(overlap) > 0:
                if missing:
                    print(f"[model_io] Loaded with missing keys (count={len(missing)}).")
                if unexpected:
                    print(f"[model_io] Ignored unexpected keys (count={len(unexpected)}).")
                return True
        except Exception as e:
            print(f"[model_io] Load attempt with prefix '{p}' failed: {e}")
            continue

    print("[model_io] No matching keys between checkpoint and model.")
    return False


def _load_state_dict_forgiving(model: torch.nn.Module, state_dict: Dict[str, Any]):
    """Load state dict with strict=False and return (missing, unexpected) if available."""
    result = model.load_state_dict(state_dict, strict=False)
    # PyTorch 2 returns a NamedTuple; earlier returns None
    if result is None:
        return [], []
    return getattr(result, "missing_keys", []), getattr(result, "unexpected_keys", [])
