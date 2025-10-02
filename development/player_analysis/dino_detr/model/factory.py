from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional


try:
    from hydra.utils import to_absolute_path as hydra_to_absolute_path
except Exception:  # pragma: no cover - hydra not installed
    hydra_to_absolute_path = None

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - OmegaConf not installed
    DictConfig = ()  # type: ignore
    OmegaConf = None  # type: ignore

from ..lightning.lit_module import DinoDetrLitModule
from .dino_detr import DINODETR, DETRsegm, PostProcess, PostProcessPanoptic, PostProcessSegm, SetCriterion
from .dino_backbone import build_dino_backbone
from .matcher import build_matcher
from .transformer import build_transformer


@dataclass
class DINODETRConfig:
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    dim_feedforward: int = 2048
    enc_layers: int = 6
    dec_layers: int = 6
    pre_norm: bool = False

    set_cost_class: float = 1.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0

    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    mask_loss_coef: float = 1.0
    dice_loss_coef: float = 1.0

    num_queries: int = 100
    aux_loss: bool = True
    masks: bool = False
    frozen_weights: Optional[str] = None
    lr_backbone: float = 0.0

    dataset_file: str = "coco"
    num_classes: Optional[int] = None
    eos_coef: float = 0.1
    dino_repo_dir: str = "third_party/dinov3"
    dino_model_name: str = "dinov3_vitl16"
    position_embedding: str = "sine"
    device: str = "cuda"


def _to_dict(cfg_like: Any) -> Dict[str, Any]:
    if cfg_like is None:
        return {}
    if OmegaConf is not None and isinstance(cfg_like, DictConfig):  # type: ignore[arg-type]
        container = OmegaConf.to_container(cfg_like, resolve=True)
        return dict(container) if isinstance(container, Mapping) else {}
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
    return str(Path(str(path)).expanduser().resolve())


def _parse_model_config(cfg_like: Any) -> DINODETRConfig:
    data = _to_dict(cfg_like)
    defaults = DINODETRConfig()

    def _get(key: str, cast, default):
        if key not in data or data[key] is None:
            return default
        return cast(data[key])

    num_classes = data.get("num_classes")
    parsed = DINODETRConfig(
        hidden_dim=int(data.get("hidden_dim", defaults.hidden_dim)),
        dropout=float(data.get("dropout", defaults.dropout)),
        nheads=int(data.get("nheads", defaults.nheads)),
        dim_feedforward=int(data.get("dim_feedforward", defaults.dim_feedforward)),
        enc_layers=int(data.get("enc_layers", defaults.enc_layers)),
        dec_layers=int(data.get("dec_layers", defaults.dec_layers)),
        pre_norm=bool(data.get("pre_norm", defaults.pre_norm)),
        set_cost_class=float(data.get("set_cost_class", defaults.set_cost_class)),
        set_cost_bbox=float(data.get("set_cost_bbox", defaults.set_cost_bbox)),
        set_cost_giou=float(data.get("set_cost_giou", defaults.set_cost_giou)),
        bbox_loss_coef=float(data.get("bbox_loss_coef", defaults.bbox_loss_coef)),
        giou_loss_coef=float(data.get("giou_loss_coef", defaults.giou_loss_coef)),
        mask_loss_coef=float(data.get("mask_loss_coef", defaults.mask_loss_coef)),
        dice_loss_coef=float(data.get("dice_loss_coef", defaults.dice_loss_coef)),
        num_queries=int(data.get("num_queries", defaults.num_queries)),
        aux_loss=bool(data.get("aux_loss", defaults.aux_loss)),
        masks=bool(data.get("masks", defaults.masks)),
        frozen_weights=_resolve_path(data.get("frozen_weights")),
        lr_backbone=float(data.get("lr_backbone", defaults.lr_backbone)),
        dataset_file=str(data.get("dataset_file", defaults.dataset_file)),
        num_classes=int(num_classes) if num_classes is not None else None,
        eos_coef=float(data.get("eos_coef", defaults.eos_coef)),
        dino_repo_dir=str(data.get("dino_repo_dir", defaults.dino_repo_dir)),
        dino_model_name=str(data.get("dino_model_name", defaults.dino_model_name)),
        position_embedding=str(data.get("position_embedding", defaults.position_embedding)),
        device=str(data.get("device", defaults.device)),
    )
    parsed.dino_repo_dir = _resolve_path(parsed.dino_repo_dir) or defaults.dino_repo_dir
    parsed.frozen_weights = _resolve_path(parsed.frozen_weights)
    return parsed


def _build_args(cfg: DINODETRConfig, device_override: Optional[str]) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        nheads=cfg.nheads,
        dim_feedforward=cfg.dim_feedforward,
        enc_layers=cfg.enc_layers,
        dec_layers=cfg.dec_layers,
        pre_norm=cfg.pre_norm,
        set_cost_class=cfg.set_cost_class,
        set_cost_bbox=cfg.set_cost_bbox,
        set_cost_giou=cfg.set_cost_giou,
        bbox_loss_coef=cfg.bbox_loss_coef,
        giou_loss_coef=cfg.giou_loss_coef,
        mask_loss_coef=cfg.mask_loss_coef,
        dice_loss_coef=cfg.dice_loss_coef,
        num_queries=cfg.num_queries,
        aux_loss=cfg.aux_loss,
        masks=cfg.masks,
        frozen_weights=cfg.frozen_weights,
        lr_backbone=cfg.lr_backbone,
        dataset_file=cfg.dataset_file,
        eos_coef=cfg.eos_coef,
        device=device_override or cfg.device,
        dino_repo_dir=cfg.dino_repo_dir,
        dino_model_name=cfg.dino_model_name,
        position_embedding=cfg.position_embedding,
    )


def _infer_num_classes(cfg: DINODETRConfig) -> int:
    if cfg.num_classes is not None:
        return int(cfg.num_classes)
    return 91 if cfg.dataset_file == "coco" else 20


def create_model(cfg_like: Any, *, device: Optional[str] = None) -> DINODETR:
    model_cfg = _parse_model_config(cfg_like)
    args = _build_args(model_cfg, device_override=device)
    backbone = build_dino_backbone(args)
    transformer = build_transformer(args)
    num_classes = _infer_num_classes(model_cfg)
    model = DINODETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=model_cfg.num_queries,
        aux_loss=model_cfg.aux_loss,
    )
    if model_cfg.masks:
        model = DETRsegm(model, freeze_detr=(model_cfg.frozen_weights is not None))
    return model


def create_loss(cfg_like: Any, *, device: Optional[str] = None):
    model_cfg = _parse_model_config(cfg_like)
    args = _build_args(model_cfg, device_override=device)
    matcher = build_matcher(args)

    weight_dict = {
        "loss_ce": 1.0,
        "loss_bbox": model_cfg.bbox_loss_coef,
        "loss_giou": model_cfg.giou_loss_coef,
    }
    if model_cfg.masks:
        weight_dict["loss_mask"] = model_cfg.mask_loss_coef
        weight_dict["loss_dice"] = model_cfg.dice_loss_coef

    if model_cfg.aux_loss:
        aux_weights = {}
        for idx in range(max(0, model_cfg.dec_layers - 1)):
            for key, value in weight_dict.items():
                aux_weights[f"{key}_{idx}"] = value
        weight_dict.update(aux_weights)

    losses = ["labels", "boxes", "cardinality"]
    if model_cfg.masks:
        losses.append("masks")

    criterion = SetCriterion(
        _infer_num_classes(model_cfg),
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=model_cfg.eos_coef,
        losses=losses,
    )
    criterion.to(device or model_cfg.device)
    return criterion


def create_postprocessors(cfg_like: DINODETRConfig) -> Dict[str, Any]:
    cfg = _parse_model_config(cfg_like)
    postprocessors: Dict[str, Any] = {"bbox": PostProcess()}
    if cfg.masks:
        postprocessors["segm"] = PostProcessSegm()
        if cfg.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)
    return postprocessors


def create_lit_module(
    *,
    cfg: Any,
    model: DINODETR,
    loss_fn,
    metric_fns: Mapping[str, Any] | None = None,
) -> DinoDetrLitModule:
    model_cfg = _parse_model_config(getattr(cfg, "model", {}))
    postprocessors = getattr(model, "postprocessors", None) or create_postprocessors(model_cfg)
    return DinoDetrLitModule(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        postprocessors=postprocessors,
    )


__all__ = ["create_model", "create_loss", "create_postprocessors"]
