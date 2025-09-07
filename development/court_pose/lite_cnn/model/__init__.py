from __future__ import annotations

from .deep_context_fpn import build_irdw_attn_fpn
from .lite_unet_context import build_preset_a

model_registry = {
    "lite_unet_context": build_preset_a,
    "deep_context_fpn": build_irdw_attn_fpn,
}

__all__ = ["model_registry"]
