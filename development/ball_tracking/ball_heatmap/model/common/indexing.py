from __future__ import annotations

from typing import Dict, List, Union


def _resolve_block_indices(num_blocks: int, spec: Union[str, List[int], Dict[str, int]]) -> List[int]:
    """
    Resolve a block-selection spec into explicit indices.
    spec:
      - "all"
      - {"every": k}  -> [0, k, 2k, ...]
      - {"last": k}   -> last k indices
      - [i, j, ...]   -> explicit list
    """
    if isinstance(spec, str):
        s = spec.lower()
        if s == "all":
            return list(range(num_blocks))
        raise ValueError(f"Unknown spec string: {spec}")
    if isinstance(spec, dict):
        if "every" in spec:
            k = int(spec["every"])
            return list(range(0, num_blocks, max(1, k)))
        if "last" in spec:
            k = int(spec["last"])
            k = max(0, min(k, num_blocks))
            return list(range(num_blocks - k, num_blocks))
        raise ValueError(f"Unknown spec dict keys: {list(spec.keys())}")
    # assume list-like of ints
    idx = sorted(set(int(i) for i in spec))
    for i in idx:
        if i < 0 or i >= num_blocks:
            raise ValueError(f"Block index out of range: {i} (num_blocks={num_blocks})")
    return idx
