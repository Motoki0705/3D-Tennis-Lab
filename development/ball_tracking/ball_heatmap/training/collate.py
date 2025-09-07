from __future__ import annotations
from typing import Any, Dict, List

import torch


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    sup_list = [s["sup"] for s in samples if "sup" in s]
    if sup_list:
        video = torch.stack([x["video"] for x in sup_list], dim=0)
        targets = {}
        # heatmaps as list per scale
        hm_list = list(zip(*[x["targets"]["hm"] for x in sup_list]))
        targets["hm"] = [torch.stack(hms, dim=0) for hms in hm_list]
        targets["speed"] = torch.stack([x["targets"]["speed"] for x in sup_list], dim=0)
        targets["vis_state"] = torch.stack([x["targets"]["vis_state"] for x in sup_list], dim=0)
        targets["vis_mask_hm"] = torch.stack([x["targets"]["vis_mask_hm"] for x in sup_list], dim=0)
        targets["vis_mask_speed"] = torch.stack([x["targets"]["vis_mask_speed"] for x in sup_list], dim=0)
        meta = [x["meta"] for x in sup_list]
        batch["sup"] = {"video": video, "targets": targets, "meta": meta}

    unsup_list = [s["unsup"] for s in samples if "unsup" in s]
    if unsup_list:
        weak = torch.stack([x["weak"] for x in unsup_list], dim=0)
        strong = torch.stack([x["strong"] for x in unsup_list], dim=0)
        meta = [x["meta"] for x in unsup_list]
        batch["unsup"] = {"weak": weak, "strong": strong, "meta": meta}

    return batch
