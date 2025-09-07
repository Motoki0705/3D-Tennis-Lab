from __future__ import annotations
from typing import Callable, List

import torch


class ReplayCompose:
    """Minimal replayable transform that applies the same op to a list of frames.

    This is a lightweight stand-in for albumentations' ReplayCompose. It supports
    a user-provided callable transform operating on CHW tensors and returns the
    list transformed identically.
    """

    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor] | None = None):
        self.transform = transform

    def __call__(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.transform is None:
            return frames
        return [self.transform(f) for f in frames]


def default_weak_aug(x: torch.Tensor) -> torch.Tensor:
    # Placeholder: identity; you can add light noise/color jitter here.
    return x


def default_strong_aug(x: torch.Tensor) -> torch.Tensor:
    # Placeholder: identity; replace with stronger augs if desired.
    return x
