from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

_BUCKET_NAMES = ("low", "medium", "high")


def _normalize_distribution(dist: Sequence[float]) -> Tuple[float, float, float]:
    values = np.asarray(dist, dtype=np.float32)
    if values.size != 3:
        raise ValueError("Distribution must contain exactly three values for low/medium/high buckets.")
    if np.any(values < 0):
        raise ValueError("Distribution values must be non-negative.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Distribution must have a positive sum.")
    normalized = (values / total).astype(np.float32)
    return float(normalized[0]), float(normalized[1]), float(normalized[2])


def _ensure_multiple_of_16(value: float) -> int:
    rounded = int(np.round(value / 16.0) * 16)
    return max(16, rounded)


@dataclass(frozen=True)
class ProgressiveResolutionConfig:
    enabled: bool = False
    low_long_side: int = 256
    medium_long_side: int = 384
    high_long_side: int = 640
    start_distribution: Tuple[float, float, float] = (0.7, 0.2, 0.1)
    end_distribution: Tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    transition_portion: float = 0.7
    seed: int = 42

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ProgressiveResolutionConfig":
        data = dict(mapping)
        start = _normalize_distribution(data.get("start_distribution", cls.start_distribution))
        end = _normalize_distribution(data.get("end_distribution", cls.end_distribution))
        transition_portion = float(data.get("transition_portion", cls.transition_portion))
        transition_portion = float(np.clip(transition_portion, 0.0, 1.0))
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            low_long_side=int(data.get("low_long_side", cls.low_long_side)),
            medium_long_side=int(data.get("medium_long_side", cls.medium_long_side)),
            high_long_side=int(data.get("high_long_side", cls.high_long_side)),
            start_distribution=start,
            end_distribution=end,
            transition_portion=transition_portion,
            seed=int(data.get("seed", cls.seed)),
        )


class ProgressiveResolutionController:
    """Maintains epoch-aware sampling weights for progressive multi-resolution training."""

    def __init__(self, cfg: ProgressiveResolutionConfig, *, eval_mode: bool = False) -> None:
        self.cfg = cfg
        self._eval_mode = eval_mode
        self._enabled = bool(cfg.enabled) and not eval_mode
        self._max_epochs = 1
        self._distribution = np.array(cfg.start_distribution, dtype=np.float32)
        self._bucket_long_sides = np.array(
            [cfg.low_long_side, cfg.medium_long_side, cfg.high_long_side], dtype=np.int32
        )
        self._rng = np.random.default_rng(cfg.seed)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_eval_mode(self, enabled: bool = True) -> None:
        self._eval_mode = enabled
        if enabled:
            self._enabled = False

    def set_max_epochs(self, max_epochs: int) -> None:
        self._max_epochs = max(1, int(max_epochs))

    def on_epoch_start(self, epoch: int) -> None:
        if self._eval_mode:
            return
        if not self._enabled:
            return
        self._distribution = self._compute_distribution(epoch)
        self._rng = np.random.default_rng(self.cfg.seed + int(epoch))

    def sample_target_shape(self, original_shape: Tuple[int, int]) -> Tuple[int, int]:
        long_side = self.sample_long_side()
        return self._scale_to_long_side(original_shape, long_side)

    def sample_long_side(self) -> int:
        if self._eval_mode or not self._enabled:
            return int(self.cfg.high_long_side)
        bucket_index = int(self._rng.choice(len(_BUCKET_NAMES), p=self._distribution))
        return int(self._bucket_long_sides[bucket_index])

    def _compute_distribution(self, epoch: int) -> np.ndarray:
        transition_epochs = max(1, int(math.ceil(self.cfg.transition_portion * self._max_epochs)))
        if transition_epochs <= 1:
            progress = 1.0 if epoch > 0 else 0.0
        else:
            progress = min(1.0, float(epoch) / float(transition_epochs - 1))
        start = np.array(self.cfg.start_distribution, dtype=np.float32)
        end = np.array(self.cfg.end_distribution, dtype=np.float32)
        interpolated = start + (end - start) * progress
        interpolated = np.clip(interpolated, 0.0, None)
        total = float(interpolated.sum())
        if total <= 0:
            return np.array(self.cfg.end_distribution, dtype=np.float32)
        return interpolated / total

    @staticmethod
    def _scale_to_long_side(original_shape: Tuple[int, int], desired_long_side: int) -> Tuple[int, int]:
        orig_h, orig_w = original_shape
        desired = max(16, int(desired_long_side))
        if orig_h <= 0 or orig_w <= 0:
            return desired, desired
        if orig_h >= orig_w:
            scale = desired / float(orig_h)
        else:
            scale = desired / float(orig_w)
        scaled_h = _ensure_multiple_of_16(orig_h * scale)
        scaled_w = _ensure_multiple_of_16(orig_w * scale)
        return scaled_h, scaled_w


__all__ = [
    "ProgressiveResolutionConfig",
    "ProgressiveResolutionController",
]
