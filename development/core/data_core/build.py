"""Dataset factory helpers for Hydra/Lightning builders."""

from __future__ import annotations

from typing import Any, Callable, Dict
from omegaconf import DictConfig, OmegaConf
from ..datasets.ball import BallSequenceDataset
from ..datasets.court import CourtKeypointDataset
from ..datasets.player import PlayerSequenceDataset
from ..collate import batch

from importlib import import_module
from typing import Iterable, Mapping

from ..lightning.base_datamodule import BaseDataModule
from ..augment.augmentations import (
    StandardAugmentations,
    LightAugmentations,
    EvalAugmentations,
    make_clip_replay_adapter,
)


DATASET_REGISTRY: Dict[str, Callable[..., Any]] = {
    "ball": BallSequenceDataset,
    "player": PlayerSequenceDataset,
    "court": CourtKeypointDataset,
}
COLLATE_FN_REGISTRY: Dict[str, Callable[[Iterable[Any]], Any]] = {
    "sequence_collate": batch.sequence_collate,
    "flatten_sequence_collate": batch.flatten_sequence_collate,
}


def register_dataset(name: str, builder: Callable[..., Any]) -> None:
    """Register a custom dataset builder for ``build_dataset``."""

    key = name.lower()
    DATASET_REGISTRY[key] = builder


def register_collate_fn(name: str, fn: Callable[[Iterable[Any]], Any]) -> None:
    """Register a custom collate function for ``build_datamodule``."""
    key = name.lower()
    COLLATE_FN_REGISTRY[key] = fn


def build_dataset(name: str, /, **kwargs: Any) -> Any:
    """Instantiate a dataset based on ``name`` and keyword arguments."""

    key = name.lower()
    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    builder = DATASET_REGISTRY[key]
    return builder(**kwargs)


def _ensure_dataset_registered(name: str, *, register: Mapping[str, Any] | None = None) -> None:
    key = name.lower()
    if key in DATASET_REGISTRY:
        return
    if not register:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Provide register={{...}} or use one of: {available}")
    target_path = register.get("target")
    module_path, _, attr = str(target_path).rpartition(":")
    if not module_path:
        module_path, _, attr = str(target_path).rpartition(".")
    cls = getattr(import_module(module_path), attr)
    reg_name = str(register.get("name", name))
    register_dataset(reg_name, cls)


def _ensure_collate_fn_registered(name: str, *, register: Mapping[str, Any] | None = None) -> None:
    key = name.lower() if name is not None else ""
    if key in COLLATE_FN_REGISTRY:
        return
    if not register:
        available = ", ".join(sorted(COLLATE_FN_REGISTRY))
        raise KeyError(f"Unknown collate fn '{name}'. Provide register={{...}} or use one of: {available}")
    target_path = register.get("target")
    module_path, _, attr = str(target_path).rpartition(":")
    if not module_path:
        module_path, _, attr = str(target_path).rpartition(".")
    cls = getattr(import_module(module_path), attr)
    reg_name = str(register.get("name", name))
    register_collate_fn(reg_name, cls)


def _build_augment_bundle(spec: Mapping[str, Any]):
    bundle_name = str(spec.get("bundle", "standard")).lower()
    image_size = spec.get("image_size", (256, 256))
    targets = tuple(spec.get("targets", ("image",)))
    use_replay = bool(spec.get("use_replay", False))
    kwargs = dict(
        image_size=image_size,
        targets=targets,
        use_replay=use_replay,
        normalize_mean=spec.get("normalize_mean", (0.485, 0.456, 0.406)),
        normalize_std=spec.get("normalize_std", (0.229, 0.224, 0.225)),
        remove_invisible_keypoints=spec.get("remove_invisible_keypoints", False),
        bbox_format=spec.get("bbox_format"),
        bbox_label_fields=tuple(spec.get("bbox_label_fields", ())),
    )
    if bundle_name == "standard":
        bundle = StandardAugmentations(**kwargs)
    elif bundle_name == "light":
        bundle = LightAugmentations(**kwargs)
    elif bundle_name == "eval":
        bundle = EvalAugmentations(**kwargs)
    else:
        raise ValueError(f"Unknown augment bundle '{bundle_name}'. Use standard|light|eval")
    tfms = bundle()
    keypoints_field = spec.get("keypoints_field", "keypoints")
    bbox_field = spec.get("bbox_field", "bboxes")
    class_field = spec.get("class_field", "classes")
    return {
        "train": make_clip_replay_adapter(
            tfms["train"], keypoints_field=keypoints_field, bboxes_field=bbox_field, classes_field=class_field
        ),
        "val": make_clip_replay_adapter(
            tfms["val"], keypoints_field=keypoints_field, bboxes_field=bbox_field, classes_field=class_field
        ),
        "test": make_clip_replay_adapter(
            tfms.get("test", tfms["val"]),
            keypoints_field=keypoints_field,
            bboxes_field=bbox_field,
            classes_field=class_field,
        ),
    }


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


def build_datamodule(
    *,
    cfg_like: Mapping[str, Any],
):
    """Build a LightningDataModule from a dataset spec using core factories.

    Parameters
    ----------
    name: registry name of the dataset (register via 'register' mapping if unknown)
    """
    data_cfg = _to_dict(cfg_like)
    dataset_name = str(data_cfg.get("dataset_name", "")).lower()
    dataset_register = data_cfg.get("dataset_register", None)
    collate_fn_name = data_cfg.get("collate_fn_name")
    collate_fn_name = str(collate_fn_name).lower() if collate_fn_name is not None else None
    collate_fn_register = data_cfg.get("collate_fn_register", None)
    dataset = data_cfg.get("dataset", {})
    augment = data_cfg.get("augment", {})

    if dataset_name is not None:
        _ensure_dataset_registered(dataset_name, register=dataset_register)
    if collate_fn_name is not None:
        _ensure_collate_fn_registered(collate_fn_name, register=collate_fn_register)
    aug = _build_augment_bundle(augment or {})

    # Build per-split datasets so transforms don't collide.
    ds_kwargs = dict(dataset)
    full_dataset = build_dataset(dataset_name, **ds_kwargs, transform=None)

    collate_fn = COLLATE_FN_REGISTRY.get(collate_fn_name) if collate_fn_name else None

    return BaseDataModule(
        config=data_cfg,
        dataset=full_dataset,
        train_transforms=aug["train"],
        val_transforms=aug["val"],
        test_transforms=aug["test"],
        collate_fn=collate_fn,
    )


__all__ = ["build_dataset", "register_dataset", "register_collate_fn", "build_datamodule"]
