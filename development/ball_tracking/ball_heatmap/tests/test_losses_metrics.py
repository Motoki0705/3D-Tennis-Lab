import sys
import types
import torch


def test_weighted_mse_pos_mask_threshold_effect():
    # Construct a target with one strong positive and many small values
    target = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    target[:, :, 3, 4] = 1.0
    target[:, :, 2:5, 3:6] += 0.1  # small positive neighborhood
    pred = torch.zeros_like(target)

    # Case A: tau=0.0 -> everything counted as positive, scaled by pos_weight
    # Import lazily with PL stub
    import importlib

    try:
        importlib.import_module("pytorch_lightning")
    except Exception:
        sys.modules["pytorch_lightning"] = types.SimpleNamespace(LightningModule=object)
    from development.ball_tracking.ball_heatmap.lit_module import weighted_mse_loss

    loss_a = weighted_mse_loss(pred, target, pos_weight=10.0, pos_mask_tau=0.0)
    # Case B: tau=0.5 -> only the center pixel is positive; others are negative (no pos_weight)
    loss_b = weighted_mse_loss(pred, target, pos_weight=10.0, pos_mask_tau=0.5)

    assert loss_a.item() > loss_b.item(), "Thresholding positives should reduce weighted loss vs all-positive mask"


def _ns(**kwargs):
    from types import SimpleNamespace

    for k, v in kwargs.items():
        if isinstance(v, dict):
            kwargs[k] = _ns(**v)
    return SimpleNamespace(**kwargs)


def test_pck_offset_img_perfect_match():
    # Build minimal config
    import importlib

    try:
        importlib.import_module("pytorch_lightning")
    except Exception:
        sys.modules["pytorch_lightning"] = types.SimpleNamespace(LightningModule=object)
    from development.ball_tracking.ball_heatmap.lit_module import BallLitModule

    cfg = _ns(
        model={"deep_supervision_strides": [8, 4]},
        training={
            "loss": {
                "lambda_hmap": 1.0,
                "lambda_offset": 1.0,
                "lambda_coord": 0.0,
                "deep_supervision_weights": [0.5, 0.5],
                "focal": False,
                "pos_weight": 25.0,
                "pos_mask_tau": 0.3,
            },
            "lr_head": 1e-3,
            "vit_lr": 1e-5,
            "weight_decay": 0.0,
            "freeze_vit_epochs": 0,
        },
        dataset={"heatmap": {"offset_mask_tau": 0.3}},
        evaluation={"pck_thresholds": [0.05]},
    )
    mod = BallLitModule(cfg)

    # Highest-resolution heatmap size (H/stride, W/stride) with stride=4 for img 32x32 -> 8x8
    H, W, stride = 32, 32, 4
    Hs, Ws = H // stride, W // stride
    b = 2

    hmap = torch.zeros((b, 1, Hs, Ws), dtype=torch.float32)
    offs = torch.zeros((b, 2, Hs, Ws), dtype=torch.float32)
    # Put peak at (2,3) and set offsets dx=0.5, dy=0.25
    y, x = 2, 3
    hmap[:, :, y, x] = 1.0
    offs[:, 0, y, x] = 0.5
    offs[:, 1, y, x] = 0.25

    # GT equals predicted location in image pixels
    gt_xy = torch.tensor([[(x + 0.5) * stride, (y + 0.25) * stride] for _ in range(b)], dtype=torch.float32)
    valid = torch.ones((b,), dtype=torch.float32)

    pck = mod._pck_offset_img(hmap, offs, gt_xy, valid, (H, W), threshold_ratio=0.01)
    assert torch.isclose(pck, torch.tensor(1.0)), "Perfect match should yield PCK=1"
