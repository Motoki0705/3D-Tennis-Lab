from __future__ import annotations

from development.core.data_core.build import register_dataset
from .dataset import CocoDetectionDataset


def register(name: str = "dino_detr_coco") -> str:
    """Register the CocoDetectionDataset into the core dataset registry.

    Returns the registered name so configs can reference it if desired.
    """
    register_dataset(name, CocoDetectionDataset)
    return name
