from typing import Dict, Optional, List

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .encoders import TemporalSwinMoEEncoder
from .common.config import EncoderCfg, TemporalCfg, MoECfg
from .decoders import FPNDecoder
from .heads import PerScaleHeads


class BallHeatmapModel(nn.Module):
    """
    Outputs:
      - heatmaps: List[Tensor[B,T,1,Hs,Ws]]
      - speed:    Tensor[B,T,2]                 (screen-normalized velocity)
      - vis_logits: Tensor[B,T,3]               (COCO visibility logits)
    If SSL enabled, also returns *_strong from the strong view.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # --- Encoder config ---
        backbone = getattr(config.model, "backbone", {})
        use_strides = list(config.model.deep_supervision_strides)

        enc_cfg = EncoderCfg(
            embed_dim=int(getattr(backbone, "embed_dim", 96)),
            depths=list(getattr(backbone, "depths", [2, 2, 6, 2])),
            num_heads=list(getattr(backbone, "num_heads", [3, 6, 12, 24])),
            window_size=int(getattr(backbone, "window_size", 7)),
            drop_path_rate=float(getattr(backbone, "drop_path_rate", 0.1)),
            out_strides=use_strides,
        )
        tecfg = getattr(config.model, "temporal_encoder", None)
        temporal_cfg = TemporalCfg(
            enabled=bool(getattr(tecfg, "enabled", False)) if tecfg is not None else False,
            stages=[int(s) for s in getattr(tecfg, "strides", [16, 32])] if tecfg is not None else None,
        )
        backbone_moe = getattr(backbone, "moe", None) or getattr(config, "moe", None)
        moe_cfg = MoECfg(
            enabled=bool(backbone_moe.get("enabled", False)) if backbone_moe is not None else False,
            stages=[int(s) for s in backbone_moe.get("strides", [])]
            if (backbone_moe and "strides" in backbone_moe)
            else None,
            blocks=backbone_moe.get("blocks", "all") if backbone_moe is not None else "all",
            num_experts=int(backbone_moe.get("num_experts", 4)) if backbone_moe is not None else 4,
            hidden_mult=float(backbone_moe.get("hidden_mult", 4.0)) if backbone_moe is not None else 4.0,
            dropout=float(backbone_moe.get("dropout", 0.0)) if backbone_moe is not None else 0.0,
            router_z_loss_coef=float(backbone_moe.get("router_z_loss_coef", 1e-2))
            if backbone_moe is not None
            else 1e-2,
            load_balance_coef=float(backbone_moe.get("load_balance_coef", 1e-2)) if backbone_moe is not None else 1e-2,
        )
        self.encoder = TemporalSwinMoEEncoder(enc=enc_cfg, temporal=temporal_cfg, moe=moe_cfg)

        base = enc_cfg.embed_dim
        enc_out_channels = {4: base, 8: base * 2, 16: base * 4, 32: base * 8}
        self.feat_strides = list(enc_out_channels.keys())
        self.feat_channels = list(enc_out_channels.values())

        # --- Decoder ---
        decoder_channels = list(getattr(config.model, "decoder_channels", [512, 256, 128, 64]))
        tcfg = getattr(config.model, "temporal", None)
        tcfg_dict = {k: v for k, v in (tcfg.items() if tcfg is not None else [])} if tcfg is not None else None
        self.decoder = FPNDecoder(
            in_channels_map=enc_out_channels,
            out_channels=decoder_channels[-1],
            use_strides=use_strides,
            temporal_cfg=tcfg_dict,
        )

        self.temporal_enabled = (
            getattr(self.encoder, "temporal_blocks", None) and len(self.encoder.temporal_blocks) > 0
        ) or (tcfg_dict is not None and bool(tcfg_dict.get("enabled", False)))

        # --- Heads ---
        strides = list(config.model.deep_supervision_strides)
        hmap_ch = int(config.model.heatmap_channels)

        ssl_cfg = getattr(config, "ssl", None)
        self.use_ssl = bool(ssl_cfg and ssl_cfg.get("enabled", False))

        self.heads_per_scale = PerScaleHeads(
            in_channels=decoder_channels[-1],
            strides=strides,
            heatmap_channels=hmap_ch,
            speed_out_channels=2,
            vis_classes=3,
            head_hidden=int(getattr(getattr(config.model, "heads", {}), "hidden", 256)),
        )

    def unfreeze_encoder_temporal(self):
        if hasattr(self, "encoder_temporal"):
            for p in self.encoder_temporal.parameters():
                p.requires_grad = True
        if hasattr(self.encoder, "temporal_block") and self.encoder.temporal_block is not None:
            for p in self.encoder.temporal_block.parameters():
                p.requires_grad = True
        if hasattr(self.encoder, "temporal_blocks") and self.encoder.temporal_blocks is not None:
            for m in self.encoder.temporal_blocks:
                for p in m.parameters():
                    p.requires_grad = True

    def _apply_heads(self, dec_feats: Dict[int, torch.Tensor], B: int, T: int, is_sequence: bool):
        # heads_per_scale returns:
        #   heatmaps: List[[B*T or B, 1, Hs, Ws]]
        #   speed:    [B*T or B, 2]
        #   vis:      [B*T or B, 3]
        h_list, speed_bt, vis_bt = self.heads_per_scale(dec_feats)

        # reshape heatmaps back to [B,T,...]
        def reshape_hlist(hlist: List[torch.Tensor]) -> List[torch.Tensor]:
            out = []
            for h in hlist:
                if is_sequence:
                    out.append(h.view(B, T, *h.shape[1:]))
                else:
                    out.append(h.unsqueeze(1))  # [B,1,...] -> [B,T=1,...]
            return out

        if is_sequence:
            speed = speed_bt.view(B, T, 2)
            vis_logits = vis_bt.view(B, T, 3)
        else:
            speed = speed_bt.view(B, 1, 2)
            vis_logits = vis_bt.view(B, 1, 3)

        return reshape_hlist(h_list), speed, vis_logits

    def forward(self, x: torch.Tensor, x_strong: Optional[torch.Tensor] = None) -> Dict:
        if self.use_ssl and x_strong is None:
            raise ValueError("Strongly augmented input `x_strong` is required when SSL is enabled.")

        # Accept [B,3,H,W] or [B,T,3,H,W]
        is_sequence = x.dim() == 5
        if is_sequence:
            B, T = x.shape[:2]
        else:
            B, T = x.shape[0], 1

        # --- Weak / standard path ---
        aux_losses: Dict = {}
        enc_out = self.encoder(x)  # encoder should manage [B,T,3,H,W] or [B,3,H,W]
        aux_losses.update(enc_out.get("aux_losses", {}))
        enc_feats_weak = enc_out["features"]

        if (
            hasattr(self.decoder, "forward_bt")
            and is_sequence
            and T > 1
            and getattr(self.decoder, "temporal_enabled", False)
        ):
            dec_feats_weak = self.decoder.forward_bt(enc_feats_weak, B, T)
        else:
            dec_feats_weak = self.decoder(enc_feats_weak)

        h_w, speed_w, vis_w = self._apply_heads(dec_feats_weak, B, T, is_sequence)

        if not self.use_ssl:
            return {
                "heatmaps": h_w,  # List[[B,T,1,Hs,Ws]]
                "speed": speed_w,  # [B,T,2]
                "vis_logits": vis_w,  # [B,T,3]
                "aux_losses": aux_losses,
            }

        # --- Strong path (SSL) ---
        x_s = x_strong
        if x_s.dim() == 4 and is_sequence:  # [B*T,3,H,W] coming in by mistake
            x_s = x_s.view(B, T, *x_s.shape[1:])
        enc_out_s = self.encoder(x_s if x_s.dim() == 5 else x_s)
        aux_losses.update(enc_out_s.get("aux_losses", {}))
        enc_feats_strong = enc_out_s["features"]

        if (
            hasattr(self.decoder, "forward_bt")
            and is_sequence
            and T > 1
            and getattr(self.decoder, "temporal_enabled", False)
        ):
            dec_feats_strong = self.decoder.forward_bt(enc_feats_strong, B, T)
        else:
            dec_feats_strong = self.decoder(enc_feats_strong)

        h_s, speed_s, vis_s = self._apply_heads(dec_feats_strong, B, T, is_sequence)

        return {
            "heatmaps": h_w,
            "speed": speed_w,
            "vis_logits": vis_w,
            "heatmaps_strong": h_s,
            "speed_strong": speed_s,
            "vis_logits_strong": vis_s,
            "aux_losses": aux_losses,
        }


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "model": {
            "backbone": {
                "embed_dim": 96,
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24],
                "window_size": 7,
                "drop_path_rate": 0.1,
                "moe": {"enabled": True, "strides": [16, 32], "num_experts": 4},
            },
            "temporal_encoder": {"enabled": False, "strides": [16, 32]},
            "temporal": {"enabled": True},
            "deep_supervision_strides": [8, 16, 32],
            "decoder_channels": [256, 128, 64],
            "heatmap_channels": 1,
            "heads": {"hidden": 256},
        },
        "ssl": {"enabled": False},
    })
    model = BallHeatmapModel(cfg)
    print(model)

    # Non-sequence
    x = torch.randn(2, 3, 7 * 32, 7 * 32)
    y = model(x)
    print("Non-seq:", len(y["heatmaps"]), y["heatmaps"][0].shape, y["speed"].shape, y["vis_logits"].shape)

    # Sequence
    cfg.model.temporal_encoder.enabled = True
    cfg.ssl.enabled = True
    model = BallHeatmapModel(cfg)
    xs = torch.randn(2, 4, 3, 7 * 32, 7 * 32)
    xs_str = torch.randn(2, 4, 3, 7 * 32, 7 * 32)
    ys = model(xs, xs_str)
    print(
        "Seq:",
        len(ys["heatmaps"]),
        ys["heatmaps"][0].shape,
        ys["speed"].shape,
        ys["vis_logits"].shape,
        len(ys["heatmaps_strong"]),
        ys["heatmaps_strong"][0].shape,
        ys["speed_strong"].shape,
        ys["vis_logits_strong"].shape,
    )
