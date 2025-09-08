from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork


class DINOv3ViTAdapter(nn.Module):
    """Adapt a frozen DINOv3 ViT into multi-scale feature maps for Faster R-CNN.

    - Loads a pretrained ViT from local `third_party/dinov3` via torch.hub.
    - Freezes all ViT parameters (foundation model usage).
    - Projects patch tokens to a 2D feature map and builds a small FPN.
    - Exposes `out_channels` as required by torchvision detectors.
    """

    def __init__(
        self,
        *,
        repo_dir: str,
        entry: str,
        weights: str,
        out_channels: int = 256,
        fpn_levels: int = 3,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.vit = torch.hub.load(repo_dir, entry, source="local", weights=weights)

        # Determine patch size and embedding dim from DINOv3 ViT
        patch_size = getattr(self.vit, "patch_size", None)
        embed_dim = getattr(self.vit, "embed_dim", None) or getattr(self.vit, "num_features", None)
        if patch_size is None or embed_dim is None:
            raise RuntimeError("Could not infer patch_size/embed_dim from DINOv3 ViT")
        self.patch_size: int = int(patch_size)
        self.embed_dim: int = int(embed_dim)

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        # 1x1 projection from token dim to detector channel dim
        self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)

        # Build a minimal FPN over 3 levels by downsampling the base map
        self.make_coarse4 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.make_coarse5 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[out_channels, out_channels, out_channels], out_channels=out_channels
        )

        self.out_channels = out_channels
        self._fpn_levels = max(1, int(fpn_levels))

    @torch.no_grad()
    def _vit_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # DINOv3 ViT returns a dict for single tensor input from forward_features
        feats = self.vit.forward_features(x)
        if not isinstance(feats, dict):
            # Some implementations may return list[dict]; handle gracefully
            if isinstance(feats, list) and len(feats) > 0 and isinstance(feats[0], dict):
                feats = feats[0]
            else:
                raise RuntimeError("Unexpected DINOv3 feature output format")
        tokens = feats.get("x_norm_patchtokens", None)
        if tokens is None:
            raise RuntimeError("DINOv3 features missing 'x_norm_patchtokens'")
        # tokens: [B, N, C] where N = (H/patch)*(W/patch)
        return tokens

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract ViT patch tokens and reshape into BCHW
        with torch.no_grad():
            tokens = self._vit_patch_tokens(x)
        b, n, c = tokens.shape
        h = x.shape[-2] // self.patch_size
        w = x.shape[-1] // self.patch_size
        assert n == h * w, f"Token count {n} mismatch with HxW {h}x{w} (patch={self.patch_size})"
        feat_2d = tokens.transpose(1, 2).contiguous().view(b, c, h, w)

        c3 = self.proj(feat_2d)
        c4 = nn.functional.max_pool2d(c3, kernel_size=2, stride=2)
        c5 = nn.functional.max_pool2d(c4, kernel_size=2, stride=2)

        lat3 = c3
        lat4 = self.make_coarse4(c4)
        lat5 = self.make_coarse5(c5)

        fpn_out = self.fpn({"c3": lat3, "c4": lat4, "c5": lat5})
        return fpn_out


def build_dinov3_vit_backbone(backbone_cfg: Dict) -> DINOv3ViTAdapter:
    return DINOv3ViTAdapter(
        repo_dir=backbone_cfg.get("repo_dir", "third_party/dinov3"),
        entry=backbone_cfg.get("entry", "dinov3_vits16"),
        weights=backbone_cfg.get("weights", "third_party/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
        out_channels=int(backbone_cfg.get("out_channels", 256)),
        fpn_levels=int(backbone_cfg.get("fpn_levels", 3)),
        freeze=bool(backbone_cfg.get("freeze", True)),
    )


class DinoFasterRCNN(nn.Module):
    """Faster R-CNN with a frozen DINOv3 ViT backbone (foundation model)."""

    def __init__(
        self,
        backbone: Dict,
        num_classes: int,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        min_size: int = 800,
        max_size: int = 1333,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
    ) -> None:
        super().__init__()

        backbone_module = build_dinov3_vit_backbone(backbone)

        # 3 FPN levels -> 3 anchor levels
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 3,
        )

        self.model = FasterRCNN(
            backbone=backbone_module,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
            image_mean=image_mean,
            image_std=image_std,
            min_size=min_size,
            max_size=max_size,
        )

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        return self.model(images, targets)


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    from torchvision import transforms

    def get_img():
        import requests

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        return image

    REPO_DIR = "third_party/dinov3"
    weights = "third_party/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    backbone_cfg = {
        "repo_dir": REPO_DIR,
        "entry": "dinov3_vits16_dd",
        "weights": weights,
        "out_channels": 256,
        "fpn_levels": 3,
        "freeze": True,
    }
    model = DinoFasterRCNN(backbone=backbone_cfg, num_classes=91)
    model.eval()

    img = get_img()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((800, 800), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)[0]

    print("Predicted boxes:", outputs["boxes"])
    print("Predicted labels:", outputs["labels"])
    print("Predicted scores:", outputs["scores"])

    # Visualize top-5 predictions
    plt.imshow(img)
    ax = plt.gca()
    for box, label, score in zip(outputs["boxes"][:5], outputs["labels"][:5], outputs["scores"][:5]):
        x1, y1, x2, y2 = box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2))
        ax.text(x1, y1, f"{label.item()}:{score:.2f}", bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()
