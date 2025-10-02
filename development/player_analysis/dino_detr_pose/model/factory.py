"""Factory functions wiring the DINOv3 backbone with DETRPose components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

try:
    from hydra.utils import to_absolute_path as hydra_to_absolute_path
except Exception:  # pragma: no cover
    hydra_to_absolute_path = None

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover
    DictConfig = ()  # type: ignore
    OmegaConf = None  # type: ignore

from .backbone import Dinov3BackboneConfig, Dinov3PoseBackbone
from .utils import ensure_detrpose_imports

ensure_detrpose_imports()

from models.detrpose import (  # type: ignore  # pylint: disable=wrong-import-position
    Criterion,
    DETRPose,
    HybridEncoder,
    HungarianMatcher,
    PostProcess,
    Transformer,
)

from development.player_analysis.dino_detr_pose.lightning.lit_module import DinoDetrPoseLitModule


@dataclass
class DinoDetrPoseConfig:
    # Backbone
    repo_dir: str = "third_party/dinov3"
    entry: str = "dinov3_vits16"
    weights: str = "third_party/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    backbone_freeze: bool = True
    backbone_out_channels: int = 256

    # Encoder
    hidden_dim: int = 256
    nheads: int = 8
    dim_feedforward: int = 1024
    encoder_layers: int = 1
    encoder_dropout: float = 0.0
    encoder_act: str = "gelu"
    use_encoder_idx: tuple[int, ...] = (2,)
    expansion: float = 1.0
    depth_mult: float = 1.0
    eval_spatial_size: tuple[int, int] = (320, 640)
    feat_strides: tuple[int, int, int] = (8, 16, 32)

    # Transformer
    num_queries: int = 60
    dec_layers: int = 6
    dropout: float = 0.0
    activation: str = "relu"
    learnable_tgt_init: bool = True
    two_stage_type: str = "standard"
    aux_loss: bool = True
    dec_pred_class_embed_share: bool = False
    dec_pred_pose_embed_share: bool = False
    two_stage_class_embed_share: bool = False
    two_stage_bbox_embed_share: bool = False
    cls_no_bias: bool = False
    dec_n_points: int = 4
    enc_n_points: int = 4
    reg_max: int = 32
    reg_scale: float = 4.0

    # Task
    num_classes: int = 1
    num_body_points: int = 17

    # Matcher / loss
    set_cost_class: float = 2.0
    set_cost_keypoints: float = 10.0
    set_cost_oks: float = 4.0
    focal_alpha: float = 0.25
    mal_alpha: Optional[float] = None
    focal_gamma: float = 2.0
    weight_dict: Mapping[str, float] = None  # type: ignore[assignment]
    losses: tuple[str, ...] = ("vfl", "keypoints")
    postprocess_num_select: int = 60

    def __post_init__(self):
        if self.weight_dict is None:
            self.weight_dict = {"loss_vfl": 2.0, "loss_keypoints": 10.0, "loss_oks": 4.0}


def _to_dict(cfg_like: Any) -> Dict[str, Any]:
    if cfg_like is None:
        return {}
    if OmegaConf is not None and isinstance(cfg_like, DictConfig):  # type: ignore[arg-type]
        container = OmegaConf.to_container(cfg_like, resolve=True)
        return dict(container)
    if isinstance(cfg_like, Mapping):
        return dict(cfg_like)
    if hasattr(cfg_like, "__dict__"):
        return dict(vars(cfg_like))
    return {}


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if path in (None, "", "null"):
        return None
    if hydra_to_absolute_path is not None:
        return hydra_to_absolute_path(str(path))
    return str(path)


def _parse_model_config(cfg_like: Any) -> DinoDetrPoseConfig:
    data = _to_dict(cfg_like)
    cfg = DinoDetrPoseConfig()
    for key, value in data.items():
        if not hasattr(cfg, key):
            continue
        setattr(cfg, key, value)
    cfg.weights = _resolve_path(cfg.weights) or cfg.weights
    cfg.repo_dir = _resolve_path(cfg.repo_dir) or cfg.repo_dir
    return cfg


def create_model(cfg_like: Any) -> DETRPose:
    cfg = _parse_model_config(cfg_like)
    backbone = Dinov3PoseBackbone(
        Dinov3BackboneConfig(
            repo_dir=cfg.repo_dir,
            entry=cfg.entry,
            weights=cfg.weights,
            out_channels=cfg.backbone_out_channels,
            freeze=cfg.backbone_freeze,
        )
    )

    encoder = HybridEncoder(
        in_channels=[cfg.backbone_out_channels] * 3,
        feat_strides=list(cfg.feat_strides),
        n_levels=3,
        hidden_dim=cfg.hidden_dim,
        nhead=cfg.nheads,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.encoder_dropout,
        enc_act=cfg.encoder_act,
        use_encoder_idx=list(cfg.use_encoder_idx),
        num_encoder_layers=cfg.encoder_layers,
        expansion=cfg.expansion,
        depth_mult=cfg.depth_mult,
        act="silu",
        eval_spatial_size=tuple(cfg.eval_spatial_size),
        temperatureH=20,
        temperatureW=20,
    )

    transformer = Transformer(
        hidden_dim=cfg.hidden_dim,
        nhead=cfg.nheads,
        num_queries=cfg.num_queries,
        num_decoder_layers=cfg.dec_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        activation=cfg.activation,
        normalize_before=False,
        return_intermediate_dec=True,
        num_feature_levels=len(cfg.feat_strides),
        enc_n_points=cfg.enc_n_points,
        dec_n_points=cfg.dec_n_points,
        learnable_tgt_init=cfg.learnable_tgt_init,
        two_stage_type=cfg.two_stage_type,
        num_classes=cfg.num_classes,
        aux_loss=cfg.aux_loss,
        dec_pred_class_embed_share=cfg.dec_pred_class_embed_share,
        dec_pred_pose_embed_share=cfg.dec_pred_pose_embed_share,
        two_stage_class_embed_share=cfg.two_stage_class_embed_share,
        two_stage_bbox_embed_share=cfg.two_stage_bbox_embed_share,
        cls_no_bias=cfg.cls_no_bias,
        num_body_points=cfg.num_body_points,
        feat_strides=list(cfg.feat_strides),
        eval_spatial_size=tuple(cfg.eval_spatial_size),
        reg_max=cfg.reg_max,
        reg_scale=cfg.reg_scale,
    )
    model = DETRPose(backbone, encoder, transformer)
    return model


def create_matcher(cfg_like: Any) -> HungarianMatcher:
    cfg = _parse_model_config(cfg_like)
    return HungarianMatcher(
        cost_class=cfg.set_cost_class,
        focal_alpha=cfg.focal_alpha,
        cost_keypoints=cfg.set_cost_keypoints,
        cost_oks=cfg.set_cost_oks,
        num_body_points=cfg.num_body_points,
    )


def create_loss(cfg_like: Any) -> Criterion:
    cfg = _parse_model_config(cfg_like)
    matcher = create_matcher(cfg_like)
    criterion = Criterion(
        num_classes=cfg.num_classes,
        matcher=matcher,
        weight_dict=dict(cfg.weight_dict),
        losses=list(cfg.losses),
        num_body_points=cfg.num_body_points,
        focal_alpha=cfg.focal_alpha,
        mal_alpha=cfg.mal_alpha,
        gamma=cfg.focal_gamma,
    )
    return criterion


def create_postprocessors(cfg_like: Any) -> Dict[str, Any]:
    cfg = _parse_model_config(cfg_like)
    return {"pose": PostProcess(num_select=cfg.postprocess_num_select, num_body_points=cfg.num_body_points)}


def create_lit_module(
    *,
    cfg: Any,
    model: DETRPose,
    loss_fn: Criterion,
    metric_fns: Mapping[str, Any] | None = None,
) -> DinoDetrPoseLitModule:
    model_cfg = getattr(cfg, "model", cfg)
    postprocessors = create_postprocessors(model_cfg)
    return DinoDetrPoseLitModule(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        postprocessors=postprocessors,
    )


__all__ = [
    "create_model",
    "create_loss",
    "create_postprocessors",
    "create_matcher",
    "create_lit_module",
]
