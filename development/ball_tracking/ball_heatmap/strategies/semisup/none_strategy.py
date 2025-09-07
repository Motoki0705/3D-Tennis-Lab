from __future__ import annotations
from typing import Dict, Any

from .base import SemisupStrategy


class NoneSemisup(SemisupStrategy):
    def training_step(self, module, batch_unsup: Dict[str, Any], global_step: int) -> Dict[str, Any]:
        return {}
