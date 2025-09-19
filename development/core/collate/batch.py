"""Collate helpers for core datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch


def sequence_collate(batch: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of samples with shape-preserving temporal stacking."""

    inputs = torch.stack([_ensure_tensor(sample["inputs"]) for sample in batch], dim=0)
    targets_batch: Dict[str, List[Any]] = {}
    metadata: List[Any] = []

    for sample in batch:
        metadata.append(sample.get("metadata", {}))
        sample_targets = sample.get("targets", {})
        for key, value in sample_targets.items():
            targets_batch.setdefault(key, []).append(value)

    targets = _collate_targets(targets_batch)
    return {
        "inputs": inputs,
        "targets": targets,
        "metadata": metadata,
    }


def flatten_sequence_collate(batch: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate and flatten the temporal dimension into the channel axis."""

    collated = sequence_collate(batch)
    inputs = collated["inputs"]
    if inputs.ndim != 5:
        raise ValueError("Expected inputs with shape [B,T,C,H,W] before flattening.")
    b, t, c, h, w = inputs.shape
    collated["inputs"] = inputs.reshape(b, t * c, h, w)
    return collated


def _collate_targets(targets_batch: Dict[str, List[Any]]) -> Dict[str, Any]:
    collated: Dict[str, Any] = {}
    for key, values in targets_batch.items():
        first = values[0]
        if torch.is_tensor(first):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated


def _ensure_tensor(tensor: Any) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError("Expected each sample to provide 'inputs' as a torch.Tensor.")
    return tensor


__all__ = ["sequence_collate", "flatten_sequence_collate"]
