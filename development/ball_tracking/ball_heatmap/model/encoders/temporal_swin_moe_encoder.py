from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..common.config import EncoderCfg, TemporalCfg, MoECfg
from ..common.indexing import _resolve_block_indices
from ..common.switch_ffn import SwitchFFN
from ..common.temporal_msa import TemporalMSA
from ..swin_core.patch_embed import PatchEmbed
from ..swin_core.stage import SwinStage


class TemporalSwinMoEEncoder(nn.Module):
    """
    From-scratch Swin-like encoder with optional temporal modules and MoE FFN inside blocks.
    Returns {stride: [B*T, C_s, H_s, W_s]} and `aux_losses` for MoE.
    """

    def __init__(
        self,
        enc: Optional[EncoderCfg] = None,
        temporal: Optional[TemporalCfg] = None,
        moe: Optional[MoECfg] = None,
        img_channels: int = 3,
        patch_size: int = 4,
    ):
        super().__init__()
        self.enc = enc or EncoderCfg()
        self.temporal_cfg = temporal or TemporalCfg(enabled=False)
        self.moe_cfg = moe or MoECfg(enabled=False)

        # Patch embedding
        self.patch_embed = PatchEmbed(img_channels=img_channels, embed_dim=self.enc.embed_dim, patch_size=patch_size)

        # Drop path schedule
        import numpy as np

        total_blocks = sum(self.enc.depths)
        dpr = list(np.linspace(0, self.enc.drop_path_rate, total_blocks))

        # Build stages
        stages: List[nn.Module] = []
        dims = [self.enc.embed_dim * (2**i) for i in range(4)]
        window = self.enc.window_size
        heads = self.enc.num_heads
        dp_idx = 0
        self._stage_strides: List[int] = []

        def stride_for_stage(idx: int) -> int:
            # After patch embedding with patch_size p, stage 0 has stride p, each subsequent
            # downsampling doubles the stride.
            return patch_size * (2**idx)

        for i in range(4):
            depth = self.enc.depths[i]
            dim = dims[i]
            num_heads = heads[i] if i < len(heads) else heads[-1]
            # figure MoE blocks for this stage
            use_moe = self.moe_cfg.enabled and (self.moe_cfg.stages is None or (2 ** (i + 1)) in self.moe_cfg.stages)
            blk_indices = None
            if use_moe:
                blk_indices = _resolve_block_indices(depth, self.moe_cfg.blocks)

            def build_moe_ffn_factory(dim=dim):
                return lambda: SwitchFFN(
                    dim=dim,
                    hidden_mult=self.moe_cfg.hidden_mult,
                    num_experts=self.moe_cfg.num_experts,
                    drop=self.moe_cfg.dropout,
                    router_z_loss_coef=self.moe_cfg.router_z_loss_coef,
                    load_balance_coef=self.moe_cfg.load_balance_coef,
                )

            stage = SwinStage(
                dim=dim,
                depth=depth,
                num_heads=num_heads,
                window_size=window,
                mlp_ratio=4.0,
                drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[dp_idx : dp_idx + depth],
                downsample=(i < 3),
                use_moe=use_moe,
                moe_blocks=blk_indices,
                build_moe_ffn=build_moe_ffn_factory() if use_moe else None,
            )
            dp_idx += depth
            stages.append(stage)
            self._stage_strides.append(stride_for_stage(i))  # e.g., [4,8,16,32] when patch_size=4
        self.stages = nn.ModuleList(stages)

        # Temporal modules per selected stride
        self.temporal_blocks = nn.ModuleDict()
        if self.temporal_cfg.enabled:
            target = set(self.temporal_cfg.stages or [8, 16, 32])
            for i in range(4):
                stride = stride_for_stage(i)
                if int(stride) in target:
                    dim = dims[i]
                    self.temporal_blocks[str(int(stride))] = TemporalMSA(
                        dim=dim,
                        num_heads=self.temporal_cfg.heads,
                        window_T=self.temporal_cfg.window_T,
                        causal=self.temporal_cfg.causal,
                        attn_drop=self.temporal_cfg.attn_drop,
                        proj_drop=self.temporal_cfg.proj_drop,
                    )

        self.return_strides = (
            list(self.enc.out_strides) if self.enc.out_strides is not None else [stride_for_stage(i) for i in range(4)]
        )
        self._moe_modules: List[SwitchFFN] = [
            m
            for s in self.stages
            for m in getattr(s, "blocks", [])
            if hasattr(m, "ffn") and isinstance(getattr(m, "ffn"), SwitchFFN)
        ]

    def _collect_moe_aux(self) -> Dict[str, torch.Tensor]:
        aux = {
            "moe_router_z": torch.tensor(0.0, device=next(self.parameters()).device),
            "moe_load_balance": torch.tensor(0.0, device=next(self.parameters()).device),
        }
        for s in self.stages:
            for blk in s.blocks:
                ffn = getattr(blk, "ffn", None)
                if isinstance(ffn, SwitchFFN):
                    a = ffn.consume_aux()
                    for k, v in a.items():
                        aux[k] = aux.get(k, 0.0) + v
        return aux

    def forward(self, x: torch.Tensor) -> Dict:
        # Accept [B,3,H,W] or [B,T,3,H,W]
        is_seq = x.dim() == 5
        if is_seq:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            BT = B * T
        else:
            B, C, H, W = x.shape
            T = 1
            BT = B

        tokens, H0, W0 = self.patch_embed(x)
        Ht, Wt = H0, W0
        feats: Dict[int, torch.Tensor] = {}

        for i, stage in enumerate(self.stages):
            tokens, Ht, Wt = stage(tokens, Ht, Wt)
            dim_i = tokens.shape[-1]
            fmap = tokens.transpose(1, 2).reshape(BT, dim_i, Ht, Wt)
            stride_now = H // Ht  # assumes integer division

            # Temporal injection at selected strides
            key = str(int(stride_now))
            if is_seq and key in self.temporal_blocks and T > 1:
                fmap_bt = fmap.view(B, T, dim_i, Ht, Wt)
                fmap_bt = self.temporal_blocks[key](fmap_bt)
                fmap = fmap_bt.view(BT, dim_i, Ht, Wt)
                tokens = fmap.flatten(2).transpose(1, 2)

            if stride_now in self.return_strides:
                feats[stride_now] = fmap

        aux = self._collect_moe_aux()
        return {"features": feats, "aux_losses": aux, "meta": {"H0": H0, "W0": W0}}


if __name__ == "__main__":
    # Simple test
    model = TemporalSwinMoEEncoder(
        enc=EncoderCfg(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.1,
            out_strides=[4, 8, 16, 32],
        ),
        temporal=TemporalCfg(enabled=True, stages=[8, 16], heads=4, window_T=3),
        moe=MoECfg(enabled=True, num_experts=4, hidden_mult=1, stages=[16, 32], blocks=[1]),
        img_channels=3,
        patch_size=4,
    )
    print(model)

    B, T, C, H, W = 2, 5, 3, 224, 224
    x = torch.randn(B, T, C, H, W)
    out = model(x)
    for s in sorted(out["features"].keys()):
        print(f"Stride {s}: {out['features'][s].shape}")
    print("Aux losses:", out["aux_losses"])
