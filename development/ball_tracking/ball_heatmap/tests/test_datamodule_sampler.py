from types import SimpleNamespace as NS
import sys
import types
import numpy as np


def _ns(obj):
    if isinstance(obj, dict):
        return NS(**{k: _ns(v) for k, v in obj.items()})
    return obj


def test_weighted_sampler_unknown_fallback(tmp_path):
    # Prepare a tiny COCO with 3 images; reuse structure from dataset test
    from .test_dataset import _make_toy_coco

    img_dir, ann_path = _make_toy_coco(tmp_path)

    cfg = _ns({
        "model": {"deep_supervision_strides": [8, 4]},
        "dataset": {
            "img_dir": str(img_dir),
            "annotation_file": str(ann_path),
            "img_size": [32, 32],
            "heatmap": {"strides": [8, 4], "sigmas": [1.0, 1.0], "offset_mask_tau": 0.3},
            "negatives": "use",
            "version_field": "view",
            "version_weights": {"v1": 0.7, "v2": 0.3},
            "v_mix_schedule": [],
            "loader": {
                "batch_size": 2,
                "num_workers": 0,
                "sampler": "weighted",
                "pin_memory": False,
                "persistent_workers": False,
            },
            "split": {"train_ratio": 1.0, "val_ratio": 0.0},
        },
        "training": {
            "max_epochs": 1,
            "precision": 16,
            "accelerator": "cpu",
            "devices": 1,
            "lr_head": 1e-3,
            "vit_lr": 1e-5,
            "weight_decay": 0.0,
            "freeze_vit_epochs": 0,
            "loss": {
                "lambda_hmap": 1.0,
                "lambda_offset": 1.0,
                "lambda_coord": 0.0,
                "deep_supervision_weights": [0.5, 0.5],
                "focal": False,
                "pos_weight": 25.0,
                "pos_mask_tau": 0.3,
            },
        },
        "evaluation": {"pck_thresholds": [0.05]},
    })

    # Stub out transform dependency to avoid albumentations requirement
    import importlib

    try:
        importlib.import_module("development.utils.transformers.keypoint_transformer")
    except Exception:
        sys.modules["development.utils.transformers.keypoint_transformer"] = types.SimpleNamespace(
            prepare_transforms=lambda img_size: (None, None)
        )
    try:
        importlib.import_module("pytorch_lightning")
    except Exception:
        sys.modules["pytorch_lightning"] = types.SimpleNamespace(LightningModule=object)
    from development.ball_tracking.ball_heatmap.datamodule import BallDataModule

    dm = BallDataModule(cfg)
    # Don't call setup (which uses real dataset/transforms). Provide minimal train_dataset stub
    dm.train_dataset = NS(indices=[0, 1, 2], dataset=NS(versions=["v1", "v2", "unknown"]))

    # Build weights; ensure unknown gets non-zero weight
    indices = dm.train_dataset.indices
    weights = dm._build_sample_weights(indices)

    assert np.all(weights > 0), "All samples (incl. unknown) must have non-zero sampling weight"
