"""
Generic Albumentations transform factories for image/keypoints/bboxes,
plus a clip-level replay adapter for time-series datasets (ball/player).

This module generalizes the former keypoint-only factory to support:
- targets=("image",), ("image","keypoints"), ("image","bboxes")
- use_replay: choose Compose vs ReplayCompose
- bbox/keypoint params only when needed
- make_clip_replay_adapter: apply one sampled augmentation consistently to all
  frames in a clip: inputs shape [T,C,H,W] (+ optional per-frame keypoints/bboxes).

Typical usages
--------------
Court (image+keypoints, deterministic on val/test):
    bundle = StandardAugmentations(
        image_size=(256,256), targets=("image","keypoints"), use_replay=False
    )
    tfms = bundle()  # {"train": Compose(...), "val": Compose(...), "test": Compose(...)}

Player (image+bboxes, replay to allow clip-level sync if needed):
    bundle = StandardAugmentations(
        image_size=(640,640), targets=("image","bboxes"), use_replay=True,
        bbox_format="coco", bbox_label_fields=("class_labels",)
    )
    tfms = bundle()

Ball (image only, clip-synchronized augmentations):
    ops_bundle = LightAugmentations(
        image_size=(256,256), targets=("image",), use_replay=True
    )
    tfms = ops_bundle()
    clip_transform = make_clip_replay_adapter(tfms["train"])  # dataset transform

Notes
-----
- Albumentations expects HWC/uint8 internally. For clip adapter we convert between
  [T,C,H,W] (float) and [T,H,W,C] (uint8) safely.
- If you pass use_replay=True but use Compose, the clip adapter will simply apply
  the pipeline per frame (no guarantee of sync). Prefer ReplayCompose for sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from ..data_core.replay import make_clip_replay_adapter

try:  # Optional dependency – defer failure until a factory is invoked.
    import albumentations as A
    from albumentations import ReplayCompose
    from albumentations.pytorch import ToTensorV2
except Exception:  # pragma: no cover - Albumentations not installed
    A = None  # type: ignore
    ReplayCompose = None  # type: ignore
    ToTensorV2 = None  # type: ignore


class AlbumentationsUnavailableError(RuntimeError):
    """Raised when a transform factory requires Albumentations at runtime."""


def _require_albumentations() -> Tuple[Any, Any, Any]:
    if A is None or ToTensorV2 is None:
        raise AlbumentationsUnavailableError("Albumentations with ToTensorV2 is required for these transforms.")
    return A, ReplayCompose, ToTensorV2


def _ensure_size_tuple(image_size: Iterable[int]) -> Tuple[int, int]:
    size = tuple(int(v) for v in image_size)
    if len(size) != 2:
        raise ValueError("'image_size' must contain exactly two integers (H, W).")
    return size  # type: ignore[return-value]


def _compose(
    ops: Sequence[Any],
    *,
    targets: Sequence[str] = ("image",),
    use_replay: bool = False,
    remove_invisible_keypoints: bool | None = None,
    bbox_format: str | None = None,
    bbox_label_fields: Sequence[str] = (),
    additional_targets: Mapping[str, str] | None = None,
) -> Any:
    """
    Build an Albumentations Compose/ReplayCompose with optional keypoint/bbox handling.

    Parameters
    ----------
    ops : list of albumentations transforms
    targets : tuple of {"image","keypoints","bboxes"}
    use_replay : if True, build ReplayCompose; else Compose
    remove_invisible_keypoints : passed to KeypointParams when "keypoints" in targets
    bbox_format : e.g., "coco" when "bboxes" in targets
    bbox_label_fields : label fields aligned with bboxes (e.g., ("class_labels",))
    additional_targets : Albumentations additional_targets
    """
    albumentations, replay_cls, _ = _require_albumentations()
    compose_cls = replay_cls if use_replay else albumentations.Compose
    if use_replay and replay_cls is None:
        raise AlbumentationsUnavailableError("ReplayCompose is not available in this Albumentations build.")

    kwargs: Dict[str, Any] = {"additional_targets": additional_targets}

    if "keypoints" in targets:
        if remove_invisible_keypoints is None:
            remove_invisible_keypoints = False
        kwargs["keypoint_params"] = albumentations.KeypointParams(
            format="xy",
            remove_invisible=remove_invisible_keypoints,
        )

    if "bboxes" in targets:
        if not bbox_format:
            raise ValueError("bbox_format is required when targets include 'bboxes'.")
        kwargs["bbox_params"] = albumentations.BboxParams(
            format=bbox_format,
            label_fields=list(bbox_label_fields),
        )

    return compose_cls(list(ops), **kwargs)


def _post_process_ops(
    *,
    normalize_mean: Sequence[float],
    normalize_std: Sequence[float],
    to_tensor: bool,
) -> Sequence[Any]:
    albumentations, _, to_tensor_cls = _require_albumentations()
    ops: list[Any] = [
        albumentations.Normalize(mean=list(normalize_mean), std=list(normalize_std)),
    ]
    if to_tensor:
        ops.append(to_tensor_cls())
    return ops


@dataclass
class BaseAugmentations:
    """Reusable skeleton for (image|keypoints|bboxes) transform bundles.

    Subclasses customise :meth:`augmentation_ops` to define their training
    augmentations while evaluation/test pipelines stick to deterministic
    preprocessing. ``__call__`` returns a dict with ``train``, ``val`` and
    ``test`` transforms for direct use in datamodules/datasets.
    """

    image_size: Sequence[int]
    targets: Sequence[str] = ("image",)  # e.g., ("image","keypoints"), ("image","bboxes")
    normalize_mean: Sequence[float] = (0.485, 0.456, 0.406)
    normalize_std: Sequence[float] = (0.229, 0.224, 0.225)
    remove_invisible_keypoints: bool | None = None
    bbox_format: str | None = None
    bbox_label_fields: Sequence[str] = ()
    additional_targets: Mapping[str, str] | None = None
    to_tensor: bool = True
    use_replay: bool = False

    def __call__(self) -> Dict[str, Any]:
        size = _ensure_size_tuple(self.image_size)
        base_ops = self._base_ops(size)
        post_ops = _post_process_ops(
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            to_tensor=self.to_tensor,
        )

        train_ops = [*base_ops, *self.augmentation_ops(), *post_ops]
        eval_ops = [*base_ops, *self.eval_ops(), *post_ops]
        test_ops = [*base_ops, *self.test_ops(), *post_ops]

        compose_kwargs = dict(
            targets=self.targets,
            use_replay=self.use_replay,
            remove_invisible_keypoints=self.remove_invisible_keypoints,
            bbox_format=self.bbox_format,
            bbox_label_fields=self.bbox_label_fields,
            additional_targets=self.additional_targets,
        )
        return {
            "train": _compose(train_ops, **compose_kwargs),
            "val": _compose(eval_ops, **compose_kwargs),
            "test": _compose(test_ops, **compose_kwargs),
        }

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def augmentation_ops(self) -> Sequence[Any]:  # pragma: no cover - override hook
        return ()

    def eval_ops(self) -> Sequence[Any]:  # pragma: no cover - override hook
        return ()

    def test_ops(self) -> Sequence[Any]:  # pragma: no cover - override hook
        return self.eval_ops()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _base_ops(self, image_size: Tuple[int, int]) -> Sequence[Any]:
        albumentations, _, _ = _require_albumentations()
        h, w = image_size
        return (albumentations.Resize(h, w),)


@dataclass
class StandardAugmentations(BaseAugmentations):
    """A moderately strong defaults set: rotate/flip/brightness/perspective/blur."""

    rotate_limit: float = 10.0
    rotate_prob: float = 0.5
    brightness_contrast_prob: float = 0.3
    horizontal_flip_prob: float = 0.5
    perspective_prob: float = 0.0
    perspective_scale: Tuple[float, float] = (0.05, 0.1)
    gaussian_blur_prob: float = 0.0
    gaussian_blur_limit: Tuple[int, int] = (3, 7)

    def augmentation_ops(self) -> Sequence[Any]:
        albumentations, _, _ = _require_albumentations()
        ops: list[Any] = []
        if self.rotate_prob > 0:
            ops.append(
                albumentations.Rotate(
                    limit=self.rotate_limit,
                    p=self.rotate_prob,
                    border_mode=0,
                )
            )
        if self.horizontal_flip_prob > 0:
            ops.append(albumentations.HorizontalFlip(p=self.horizontal_flip_prob))
        if self.brightness_contrast_prob > 0:
            ops.append(albumentations.RandomBrightnessContrast(p=self.brightness_contrast_prob))
        if self.perspective_prob > 0:
            ops.append(
                albumentations.Perspective(
                    scale=self.perspective_scale,
                    p=self.perspective_prob,
                )
            )
        if self.gaussian_blur_prob > 0:
            ops.append(
                albumentations.GaussianBlur(
                    blur_limit=self.gaussian_blur_limit,
                    p=self.gaussian_blur_prob,
                )
            )
        return ops


@dataclass
class LightAugmentations(BaseAugmentations):
    """Lightweight variant: mirror & brightness jitter."""

    horizontal_flip_prob: float = 0.5
    brightness_contrast_prob: float = 0.1

    def augmentation_ops(self) -> Sequence[Any]:
        albumentations, _, _ = _require_albumentations()
        ops: list[Any] = []
        if self.horizontal_flip_prob > 0:
            ops.append(albumentations.HorizontalFlip(p=self.horizontal_flip_prob))
        if self.brightness_contrast_prob > 0:
            ops.append(albumentations.RandomBrightnessContrast(p=self.brightness_contrast_prob))
        return ops


@dataclass
class EvalAugmentations(BaseAugmentations):
    """Deterministic preprocessing shared across validation/test splits."""

    def augmentation_ops(self) -> Sequence[Any]:  # pragma: no cover - no-op
        return ()


def prepare_keypoint_transforms(image_size: Iterable[int]):
    """Backward-compat shim for legacy keypoint-only API.

    Returns (train, val) Compose transforms with keypoint support.
    """
    bundle = StandardAugmentations(
        image_size=image_size,
        targets=("image", "keypoints"),
        use_replay=False,
    )
    transforms = bundle()
    return transforms["train"], transforms["val"]


__all__ = [
    "AlbumentationsUnavailableError",
    "BaseAugmentations",
    "StandardAugmentations",
    "LightAugmentations",
    "EvalAugmentations",
    "prepare_keypoint_transforms",
    "make_clip_replay_adapter",
    "_compose",
]
