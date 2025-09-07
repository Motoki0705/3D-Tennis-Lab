from __future__ import annotations
from typing import Dict, Any


class SemisupStrategy:
    def training_step(self, module, batch_unsup: Dict[str, Any], global_step: int) -> Dict[str, Any]:
        """Return dict with loss terms and logs; may be empty if disabled."""
        return {}
