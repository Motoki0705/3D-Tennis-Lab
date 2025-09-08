from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRunner(ABC):
    """Unified interface for train/infer runners."""

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def run(self):
        raise NotImplementedError
