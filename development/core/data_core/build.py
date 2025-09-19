"""Dataset factory helpers for Hydra/Lightning builders."""

from __future__ import annotations

from typing import Any, Callable, Dict

from ..datasets.ball import BallSequenceDataset
from ..datasets.court import CourtKeypointDataset
from ..datasets.player import PlayerSequenceDataset

DATASET_REGISTRY: Dict[str, Callable[..., Any]] = {
    "ball": BallSequenceDataset,
    "player": PlayerSequenceDataset,
    "court": CourtKeypointDataset,
}


def register_dataset(name: str, builder: Callable[..., Any]) -> None:
    """Register a custom dataset builder for ``build_dataset``."""

    key = name.lower()
    DATASET_REGISTRY[key] = builder


def build_dataset(name: str, /, **kwargs: Any) -> Any:
    """Instantiate a dataset based on ``name`` and keyword arguments."""

    key = name.lower()
    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    builder = DATASET_REGISTRY[key]
    return builder(**kwargs)


__all__ = ["build_dataset", "register_dataset", "DATASET_REGISTRY"]
