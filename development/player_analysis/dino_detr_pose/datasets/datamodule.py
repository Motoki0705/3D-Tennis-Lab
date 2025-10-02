"""Lightning datamodule builder for the COCO MPPE experiment."""

from __future__ import annotations

from typing import Any, Mapping

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover
    DictConfig = ()  # type: ignore
    OmegaConf = None  # type: ignore

from development.core.lightning.base_datamodule import BaseDataModule
from .coco_pose import CocoPlayerPoseDataset
from ..collate import collate_pose_seq_t1


def _to_dict(cfg_like: Any) -> dict[str, Any]:
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


def build_datamodule(cfg_like: Any) -> BaseDataModule:
    cfg = _to_dict(cfg_like)
    dataset_cfg = cfg.get("dataset", {})
    dataset = CocoPlayerPoseDataset(**dataset_cfg)
    return BaseDataModule(
        config=cfg,
        dataset=dataset,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        collate_fn=collate_pose_seq_t1,
    )


__all__ = ["build_datamodule"]
