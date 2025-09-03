# -*- coding: utf-8 -*-
"""
End-to-end test for development.ball_tracking.ball_heatmap.train

This test:
- Creates a tiny synthetic COCO-like dataset in a temp dir
- Monkeypatches heavy components (ViT model, transforms) to lightweight versions
- Overrides Hydra/Trainer configs to run 1 epoch on CPU
- Calls train() and verifies it runs and produces log output
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import importlib

import pytest
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2


MODULE_BASE = "development.ball_tracking.ball_heatmap"


def _write_img(path: Path, hw=(64, 80)):
    import cv2
    import numpy as np

    H, W = hw
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.circle(arr, (5, 5), 2, (255, 0, 0), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), arr)


def _build_coco(tmpdir: Path, clips, vis_seq_per_clip):
    images, annotations = [], []
    img_id, ann_id = 1, 1
    for clip in clips:
        seq = vis_seq_per_clip[clip]
        for t, kv in enumerate(seq):
            file_name = f"{clip}/img_{t:04d}.jpg"
            _write_img(tmpdir / file_name, hw=(64, 80))
            images.append({"id": img_id, "file_name": file_name, "height": 64, "width": 80})
            if kv is not None:
                x, y, v = kv
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "keypoints": [float(x), float(y), float(v)],
                    "num_keypoints": 1,
                })
                ann_id += 1
            img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "ball", "keypoints": ["ball"]}],
    }
    ann_path = tmpdir / "ann.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    return ann_path


def _fake_prepare_transforms(*, img_size):
    # lightweight and deterministic
    tf = A.Compose(
        [A.Resize(img_size[0], img_size[1]), ToTensorV2()],
        keypoint_params=A.KeypointParams(format="xy"),
    )
    return tf, tf


class DummyBallHeatmapModel(nn.Module):
    """
    Lightweight stand-in for BallHeatmapModel used to keep the e2e fast and CPU-only.
    Provides encoder/decoder/heads params for optimizer grouping and returns zero maps
    with correct shapes based on the configured deep supervision strides.
    """

    def __init__(
        self,
        vit_name: str = "vit_tiny_patch16_224",
        pretrained: bool = False,
        decoder_channels: list[int] | None = None,
        deep_supervision_strides: list[int] | None = None,
        heatmap_channels: int = 1,
        offset_channels: int = 2,
    ):
        super().__init__()
        self.strides = list(deep_supervision_strides or [8])
        self.hc = int(heatmap_channels)
        self.oc = int(offset_channels)
        # minimal params so optimizers have groups to work with
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU())
        self.heads = nn.ModuleDict(dict(h=nn.Conv2d(4, self.hc, 1), o=nn.Conv2d(4, self.oc, 1)))

    def forward(self, images: torch.Tensor):
        B, _, H, W = images.shape
        heatmaps, offsets = [], []
        for s in self.strides:
            hs, ws = H // int(s), W // int(s)
            heatmaps.append(torch.zeros(B, self.hc, hs, ws, device=images.device))
            offsets.append(torch.zeros(B, self.oc, hs, ws, device=images.device))
        return {"heatmaps": heatmaps, "offsets": offsets}


@pytest.mark.timeout(60)
def test_train_end2end_runs_single_epoch(tmp_path, monkeypatch):
    # Import target submodules to patch
    mod_dm = importlib.import_module(f"{MODULE_BASE}.datamodule")
    mod_lm = importlib.import_module(f"{MODULE_BASE}.lit_module")

    # Patch heavy components
    monkeypatch.setattr(mod_dm, "prepare_transforms", _fake_prepare_transforms)
    monkeypatch.setattr(mod_lm, "BallHeatmapModel", DummyBallHeatmapModel)

    # Build a tiny dataset: 1 clip x 6 frames (some visible, some not)
    seq = [(10, 10, 1), (15, 15, 1), (20, 20, 1), (25, 25, 2), None, (30, 30, 1)]
    img_root = tmp_path / "imgs"
    ann_path = _build_coco(img_root, ["clipA"], {"clipA": seq})

    # Where Hydra will put run artifacts
    run_dir = tmp_path / "hydra_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Import train AFTER monkeypatching model/transforms
    mod_train = importlib.import_module(f"{MODULE_BASE}.train")

    # Compose Hydra overrides for fast CPU run
    overrides = [
        # dataset locations and loader
        f"dataset.img_dir={str(img_root)}",
        f"dataset.annotation_file={str(ann_path)}",
        "dataset.img_size=[64,80]",
        "dataset.loader.batch_size=2",
        "dataset.loader.num_workers=0",
        "dataset.loader.sampler=uniform",
        # simple split ensuring we have val for checkpoint monitor
        "dataset.split.train_ratio=0.5",
        "dataset.split.val_ratio=0.25",
        # sequence disabled (single frame)
        "dataset.sequence.length=1",
        "dataset.sequence.frame_stride=1",
        # model and training runtime
        "model.deep_supervision_strides=[8]",
        "model.pretrained=false",
        "training.max_epochs=1",
        "training.accelerator=cpu",
        "training.devices=1",
        "training.precision=32",
        # keep callbacks light
        "callbacks.heatmap_logger.num_samples=2",
        # hydra output directory for assertions
        f"hydra.run.dir={str(run_dir)}",
    ]

    # Hydra reads sys.argv in hydra.main-decorated functions
    argv_backup = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]] + overrides
        # Enable full errors from Hydra to ease debugging if it fails
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")

        # Run training end-to-end (1 epoch) and testing with best ckpt
        mod_train.train()
    finally:
        sys.argv = argv_backup

    # Assert TensorBoard logs directory was created under our hydra run dir
    tb_dir = run_dir / "tb_logs" / "ball_heatmap_v1"
    assert tb_dir.exists(), f"TensorBoard log dir not found: {tb_dir}"
