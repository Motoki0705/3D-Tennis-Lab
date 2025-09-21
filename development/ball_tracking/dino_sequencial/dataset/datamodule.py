"""Datamodule factory that reuses the core ball dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from development.core.data_core.build import build_dataset
from development.core.lightning.base_datamodule import BaseDataModule


@dataclass
class DataModuleConfig:
    cfg: Mapping[str, Any]


def _to_dict(cfg_like: Any) -> Mapping[str, Any]:
    if isinstance(cfg_like, DictConfig):
        return OmegaConf.to_container(cfg_like, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg_like, Mapping):
        return cfg_like
    raise TypeError("Datamodule config must be a mapping or DictConfig.")


def _resolve_path(path: str | None) -> str:
    if not path:
        raise ValueError("Dataset path must be specified.")
    return to_absolute_path(path)


def _heatmap_size(image_size: Sequence[int], stride: int) -> Sequence[int]:
    h, w = int(image_size[0]), int(image_size[1])
    stride = max(1, int(stride))
    return [h // stride, w // stride]


def build_datamodule(cfg_like: Any) -> BaseDataModule:
    data_cfg = _to_dict(cfg_like)

    paths_cfg = data_cfg.get("paths", {})
    train_paths = paths_cfg.get("train", {})
    images_root = _resolve_path(train_paths.get("images"))
    annotation_file = _resolve_path(train_paths.get("annotation"))

    dataset_cfg = data_cfg.get("dataset", {})
    sequence_cfg = dataset_cfg.get("sequence", {})
    normalization_cfg = dataset_cfg.get("normalization", {})

    image_size = [int(v) for v in dataset_cfg.get("image_size", [288, 512])]
    output_stride = int(dataset_cfg.get("output_stride", 4))
    heatmap_size = _heatmap_size(image_size, output_stride)

    dataset = build_dataset(
        "ball",
        annotation_file=annotation_file,
        image_dir=images_root,
        sequence_length=int(sequence_cfg.get("length", 3)),
        frame_stride=int(sequence_cfg.get("stride", 1)),
        heatmap_size=tuple(int(v) for v in heatmap_size),
        heatmap_sigma=float(dataset_cfg.get("heatmap_sigma", 2.0)),
        image_size=tuple(image_size),
        drop_short_clips=bool(sequence_cfg.get("drop_short_clips", False)),
        allow_partial_last=bool(sequence_cfg.get("allow_partial_last", False)),
        normalize_mean=normalization_cfg.get("mean", [0.485, 0.456, 0.406]),
        normalize_std=normalization_cfg.get("std", [0.229, 0.224, 0.225]),
        category_name="ball",
    )

    return BaseDataModule(config=data_cfg, dataset=dataset)


__all__ = ["build_datamodule", "DataModuleConfig"]
