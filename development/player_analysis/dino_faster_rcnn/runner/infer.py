from __future__ import annotations


import torch
from omegaconf import DictConfig

from ..model.dino_faster_rcnn import DinoFasterRCNN


class InferRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @torch.no_grad()
    def run(self):
        model = DinoFasterRCNN(**self.cfg.model)
        model.eval()

        # Simple single-image inference demo if a path is provided via cfg.data.test.images
        img_path = self.cfg.data.test.images
        if not img_path:
            print("No test image provided (data.test.images). Inference runner exiting.")
            return

        import cv2
        from torchvision.transforms.functional import to_tensor

        img = cv2.imread(img_path)
        assert img is not None, f"Failed to read: {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = to_tensor(img)

        outputs = model([x])  # type: ignore
        print({k: v.shape if isinstance(v, torch.Tensor) else len(v) for k, v in outputs[0].items()})
