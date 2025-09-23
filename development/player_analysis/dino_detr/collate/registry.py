from __future__ import annotations

from development.core.data_core.build import register_collate_fn
from .batch import collate_seq_t1


def register(name: str = "collate_seq_t1") -> str:
    """Register the collate_seq_t1 into the core collate_fn registry.

    Returns the registered name so configs can reference it if desired.
    """
    register_collate_fn(name, collate_seq_t1)
    return name
