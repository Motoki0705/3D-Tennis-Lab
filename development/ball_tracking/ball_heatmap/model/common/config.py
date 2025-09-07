from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class TemporalCfg:
    enabled: bool = False
    stages: Optional[List[int]] = None  # strides to apply temporal at (e.g., [16, 32])
    heads: int = 4
    window_T: int = 5
    causal: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0


@dataclass
class MoECfg:
    enabled: bool = False
    stages: Optional[List[int]] = None  # strides to apply MoE-FFN at
    blocks: Union[str, List[int], Dict[str, int]] = "all"
    num_experts: int = 4
    hidden_mult: float = 4.0
    dropout: float = 0.0
    router_z_loss_coef: float = 1e-2
    load_balance_coef: float = 1e-2


@dataclass
class EncoderCfg:
    embed_dim: int = 96
    depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    window_size: int = 7
    drop_path_rate: float = 0.1
    out_strides: Optional[List[int]] = None  # which strides to return; None = all
