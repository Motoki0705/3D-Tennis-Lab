from __future__ import annotations

import os
from dataclasses import asdict
from typing import Tuple

import torch
import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path as abspath

from ..training.datamodule import BallDataModule, DataModuleConfig, AugmentationConfig


def _to_np_img(img_chw: torch.Tensor) -> np.ndarray:
    # img_chw: (3, H, W) in [0,1] or normalized; returns (H, W, 3)
    img = img_chw.detach().cpu().float().clamp(-10, 10)
    return np.moveaxis(img.numpy(), 0, -1)


def _denorm(img_chw: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=img_chw.dtype, device=img_chw.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=img_chw.dtype, device=img_chw.device).view(3, 1, 1)
    return (img_chw * std_t + mean_t).clamp(0.0, 1.0)


def _overlay_heatmap(base_img_chw: torch.Tensor, heatmap_hw: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    # base_img_chw: (3,H,W) in [0,1]; heatmap_hw: (H,W) in [0,1]
    h, w = base_img_chw.shape[-2:]
    hm = heatmap_hw.detach().cpu().float()
    if hm.shape[-2:] != (h, w):
        hm = torch.nn.functional.interpolate(
            hm.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze()
    hm = hm.clamp(0, 1)
    # colorize heatmap in red
    color = torch.stack([hm, torch.zeros_like(hm), torch.zeros_like(hm)], dim=0)
    over = (1 - alpha) * base_img_chw + alpha * color
    return over.clamp(0, 1)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Prepare output directory
    out_dir = abspath("outputs/dataset_preview")
    _ensure_dir(out_dir)

    # Build DataModuleConfig from cfg.data
    data_cfg = cfg.data
    aug_cfg = AugmentationConfig(**data_cfg.get("augmentation", {}))
    dm_cfg = DataModuleConfig(
        images_root=abspath(data_cfg.images_root),
        labeled_json=abspath(data_cfg.labeled_json),
        img_size=tuple(data_cfg.img_size),
        T=int(data_cfg.T),
        frame_stride=int(data_cfg.frame_stride),
        output_stride=int(data_cfg.output_stride),
        sigma_px=float(data_cfg.sigma_px),
        batch_size=int(data_cfg.batch_size),
        num_workers=int(data_cfg.num_workers),
        augmentation=aug_cfg,
        val_ratio=float(getattr(data_cfg, "val_ratio", 0.1)),
        split_seed=int(getattr(data_cfg, "split_seed", 42)),
    )

    print("DataModuleConfig:", asdict(dm_cfg))

    dm = BallDataModule(dm_cfg)
    dm.setup("fit")

    # Choose which split to preview
    splits = {
        "train": dm.train_dataloader(),
        "val": dm.val_dataloader(),
    }

    # Denorm parameters
    mean = tuple(getattr(data_cfg.get("normalize", {}), "mean", [0.485, 0.456, 0.406]))  # type: ignore
    std = tuple(getattr(data_cfg.get("normalize", {}), "std", [0.229, 0.224, 0.225]))  # type: ignore

    n_samples_per_split = int(os.environ.get("PREVIEW_SAMPLES", 2))

    for split_name, loader in splits.items():
        if loader is None:
            continue
        out_split = os.path.join(out_dir, split_name)
        _ensure_dir(out_split)

        batch = next(iter(loader))
        videos, heatmaps = batch  # videos: (B, C, T, H, W), heatmaps: (B,1,h_s,w_s)
        B, C, T, H, W = videos.shape
        print(f"[{split_name}] batch: video {videos.shape}, heatmap {heatmaps.shape}")

        for i in range(min(B, n_samples_per_split)):
            vid = videos[i]  # (C,T,H,W)
            hm = heatmaps[i, 0]  # (h_s, w_s)

            # Build a figure with T frames horizontally
            fig, axes = plt.subplots(1, T + 1, figsize=(4 * (T + 1), 4))
            if T == 1:
                axes = [axes]

            for t in range(T):
                frame_chw = vid[:, t]
                frame_denorm = _denorm(frame_chw, mean, std)
                axes[t].imshow(_to_np_img(frame_denorm))
                axes[t].set_title(f"{split_name} t={t}")
                axes[t].axis("off")

            # Overlay heatmap on last frame
            frame_last = vid[:, T - 1]
            frame_last_denorm = _denorm(frame_last, mean, std)
            over = _overlay_heatmap(frame_last_denorm, hm, alpha=0.5)
            axes[T].imshow(_to_np_img(over))
            axes[T].set_title("last + GT heatmap")
            axes[T].axis("off")

            out_path = os.path.join(out_split, f"sample_{i}.png")
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
