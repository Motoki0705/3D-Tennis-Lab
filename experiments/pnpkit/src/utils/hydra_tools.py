from __future__ import annotations

from typing import List

from hydra.utils import instantiate

from ..pipeline.base import Stage


def instantiate_stages(stage_cfg_list) -> List[Stage]:
    stages: List[Stage] = []
    for sc in stage_cfg_list:
        stages.append(instantiate(sc))
    return stages
