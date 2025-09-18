from __future__ import annotations

from typing import Dict, Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler

from .resolution import ProgressiveResolutionController


class ProgressiveResolutionSampler(Sampler[int]):
    """Sampler that groups indices per batch and assigns a long-side resolution."""

    def __init__(
        self,
        subset_size: int,
        subset_indices: Sequence[int],
        batch_size: int,
        controller: ProgressiveResolutionController,
        *,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if subset_size != len(subset_indices):
            raise ValueError("subset_size must match length of subset_indices")
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.controller = controller
        self._seed = int(seed)
        self._subset_positions = np.arange(subset_size)
        self._actual_indices = np.asarray(subset_indices, dtype=np.int64)
        self._current_order: List[int] = self._subset_positions.tolist()
        self._index_long_side: Dict[int, int] = {}
        self._epoch = 0

    def __iter__(self) -> Iterator[int]:
        return iter(self._current_order)

    def __len__(self) -> int:
        if self.drop_last:
            return (self.subset_size // self.batch_size) * self.batch_size
        return self.subset_size

    @property
    def index_long_side(self) -> Dict[int, int]:
        return self._index_long_side

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        rng = np.random.default_rng(self._seed + self._epoch)
        perm = rng.permutation(self.subset_size)
        if self.drop_last:
            usable = (self.subset_size // self.batch_size) * self.batch_size
            perm = perm[:usable]
        self._current_order = perm.tolist()
        self._assign_long_sides()

    def _assign_long_sides(self) -> None:
        self._index_long_side = {}
        if self.subset_size == 0:
            return
        actual_perm = self._actual_indices[self._current_order]
        total = len(actual_perm)
        for start in range(0, total, self.batch_size):
            batch_indices = actual_perm[start : start + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                break
            long_side = self.controller.sample_long_side()
            for data_index in batch_indices:
                self._index_long_side[int(data_index)] = int(long_side)


__all__ = ["ProgressiveResolutionSampler"]
