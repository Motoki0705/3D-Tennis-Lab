# tests/test_ball_lit_module.py
# -*- coding: utf-8 -*-
import math
import importlib
from types import SimpleNamespace
from argparse import Namespace

import pytest
import torch
import torch.nn as nn

# 🔧 ここをあなたの実モジュールパスに差し替えてください
# 例: MODULE_PATH = "experiments.ball.lit_module"
MODULE_PATH = "development.ball_tracking.ball_heatmap.lit_module"  # <-- CHANGE ME


# -----------------------------
# Helpers: synthetic data
# -----------------------------
def make_config(
    img_hw=(128, 160),
    strides=(16, 8, 4),
    pos_weight=25.0,
    pos_mask_tau=0.3,
    offset_mask_tau=0.2,
    focal=False,
    freeze_vit_epochs=1,
):
    H, W = img_hw
    return Namespace(
        model=Namespace(
            vit_name="tiny",
            pretrained=False,
            deep_supervision_strides=list(strides),
            heatmap_channels=1,
            offset_channels=2,
        ),
        dataset=Namespace(
            heatmap=Namespace(
                offset_mask_tau=offset_mask_tau,
            )
        ),
        training=Namespace(
            loss=Namespace(
                lambda_hmap=1.0,
                lambda_offset=1.0,
                lambda_coord=0.0,
                deep_supervision_weights=[0.25, 0.5, 1.0][: len(strides)],
                focal=focal,
                pos_weight=pos_weight,
                pos_mask_tau=pos_mask_tau,
            ),
            lr_head=1e-3,
            vit_lr=1e-5,
            weight_decay=0.01,
            freeze_vit_epochs=freeze_vit_epochs,
        ),
        evaluation=Namespace(
            pck_thresholds=[0.05, 0.1],
        ),
        dataset_img_hw=(H, W),
    )


def _one_hot_heatmap(Hs, Ws, cx, cy):
    """Put 1.0 at integer cell (cy, cx)."""
    hm = torch.zeros(1, Hs, Ws, dtype=torch.float32)
    y = int(round(cy))
    x = int(round(cx))
    y = max(0, min(Hs - 1, y))
    x = max(0, min(Ws - 1, x))
    hm[0, y, x] = 1.0
    return hm


def make_batch(B=2, img_hw=(128, 160), strides=(16, 8, 4)):
    """Create a consistent synthetic batch matching the module's expectations."""
    H, W = img_hw
    images = torch.randn(B, 3, H, W, dtype=torch.float32)

    # Choose deterministic GT points in image pixels
    gt_xy = []
    for i in range(B):
        x = 30.0 + 15.0 * i
        y = 40.0 + 12.0 * i
        gt_xy.append(torch.tensor([x, y], dtype=torch.float32))
    gt_xy = torch.stack(gt_xy, dim=0)  # [B,2]
    valid_mask = torch.ones(B, 1, dtype=torch.float32)

    # Build multi-scale heatmaps and offsets
    heatmaps = []
    offsets = []
    for s in strides:
        Hs, Ws = H // s, W // s
        hm_s = []
        off_s = []
        for b in range(B):
            xs = gt_xy[b, 0] / s
            ys = gt_xy[b, 1] / s
            cx, cy = math.floor(xs), math.floor(ys)
            dx, dy = xs - cx, ys - cy  # in-cell offsets

            hm = _one_hot_heatmap(Hs, Ws, cx, cy)  # [1,Hs,Ws]
            off = torch.zeros(2, Hs, Ws, dtype=torch.float32)
            off[0, cy, cx] = dx
            off[1, cy, cx] = dy
            hm_s.append(hm)
            off_s.append(off)

        heatmaps.append(torch.stack(hm_s, dim=0))  # [B,1,Hs,Ws]
        offsets.append(torch.stack(off_s, dim=0))  # [B,2,Hs,Ws]

    batch = {
        "image": images,
        "heatmaps": heatmaps,
        "offsets": offsets,
        "coord": gt_xy,  # [B,2] in image pixels
        "valid_mask": valid_mask,  # [B,1]
    }
    return batch


# -----------------------------
# Dummy model to avoid heavy nets
# -----------------------------
class DummyBallHeatmapModel(nn.Module):
    """
    Mimics BallHeatmapModel interface:
      - attributes: encoder, decoder, heads (with params)
      - forward(images) -> {"heatmaps": [...], "offsets": [...]}
    The outputs are externally set via set_ground_truth().
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Tiny params to exercise param groups & freezing
        self.encoder = nn.Sequential(nn.Conv2d(3, 3, 1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(3, 3, 1), nn.ReLU())
        self.heads = nn.ModuleDict(dict(h=nn.Conv2d(3, 1, 1), o=nn.Conv2d(3, 2, 1)))
        self._tgt_heatmaps = None
        self._tgt_offsets = None

    def set_ground_truth(self, tgt_hmaps, tgt_offs):
        # Store precomputed multi-scale targets to return on forward
        self._tgt_heatmaps = [t.detach().clone() for t in tgt_hmaps]
        self._tgt_offsets = [t.detach().clone() for t in tgt_offs]

    def forward(self, images):
        assert (
            self._tgt_heatmaps is not None and self._tgt_offsets is not None
        ), "Call set_ground_truth() before forward"
        # Return exact targets, ensuring correct shapes
        return {"heatmaps": self._tgt_heatmaps, "offsets": self._tgt_offsets}


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture(scope="module")
def mod():
    try:
        return importlib.import_module(MODULE_PATH)
    except Exception as e:
        pytest.skip(f"Import failed. Set MODULE_PATH correctly. Error: {e}")


@pytest.fixture()
def patched_module(monkeypatch, mod):
    # Patch BallHeatmapModel symbol in the target module to our Dummy
    monkeypatch.setattr(mod, "BallHeatmapModel", DummyBallHeatmapModel)
    return mod


# -----------------------------
# Tests
# -----------------------------
def test_weighted_mse_loss_basic(patched_module):
    wmse = patched_module.weighted_mse_loss
    device = torch.device("cpu")

    # Simple 1x1 map with one positive >= tau and one negative < tau
    pred = torch.tensor([[[[0.0, 0.0]]]], device=device)  # [B=1,C=1,H=1,W=2]
    tgt = torch.tensor([[[[1.0, 0.0]]]], device=device)
    loss = wmse(pred, tgt, pos_weight=10.0, pos_mask_tau=0.3)

    # pos pixel diff^2 = 1, neg pixel diff^2 = 0
    # avg_pos = 1, avg_neg = 0 => total = 10*1 + 0 = 10
    assert torch.isfinite(loss)
    assert abs(loss.item() - 10.0) < 1e-5


def test_configure_optimizers_and_param_groups(patched_module):
    cfg = make_config()
    m = patched_module.BallLitModule(cfg)
    # After monkeypatch, module created a DummyBallHeatmapModel inside
    opt_cfg = m.configure_optimizers()
    opt = opt_cfg["optimizer"]
    # two param groups: head/decoder + encoder
    assert len(opt.param_groups) == 2
    # LR per group matches config
    lrs = {pg["lr"] for pg in opt.param_groups}
    assert cfg.training.lr_head in lrs
    assert cfg.training.vit_lr in lrs


def test_freeze_then_unfreeze_encoder(patched_module):
    cfg = make_config(freeze_vit_epochs=1)
    module = patched_module.BallLitModule(cfg)

    # Initially frozen
    assert all(not p.requires_grad for p in module.model.encoder.parameters())

    # Simulate epoch 1 start -> unfreeze
    called = []

    module.trainer = SimpleNamespace(
        current_epoch=1,  # ← ここに現在エポックを置く
        datamodule=SimpleNamespace(update_sampling_for_epoch=lambda e: called.append(e)),
    )

    module.on_train_epoch_start()

    assert all(p.requires_grad for p in module.model.encoder.parameters())
    assert called == [1]


def test_forward_and_loss_shapes_and_values(patched_module):
    cfg = make_config()
    module = patched_module.BallLitModule(cfg)

    batch = make_batch(B=3, img_hw=cfg.dataset_img_hw, strides=tuple(cfg.model.deep_supervision_strides))
    # Provide perfect predictions via dummy model
    module.model.set_ground_truth(batch["heatmaps"], batch["offsets"])

    out = module._forward_and_loss(batch)

    # keys exist
    for k in ["loss", "loss_h", "loss_o", "pred_heatmaps", "pred_offsets", "tgt_heatmaps"]:
        assert k in out

    # lists length == #scales
    n_scales = len(cfg.model.deep_supervision_strides)
    assert len(out["pred_heatmaps"]) == n_scales
    assert len(out["pred_offsets"]) == n_scales

    # loss should be ~0 because pred == target (offset L1 masked at center is 0)
    assert out["loss"].item() < 1e-6
    assert out["loss_h"].item() < 1e-6
    assert out["loss_o"].item() < 1e-6


def test_validation_step_returns_logging_payload(patched_module):
    cfg = make_config()
    module = patched_module.BallLitModule(cfg)

    batch = make_batch(B=2, img_hw=cfg.dataset_img_hw, strides=tuple(cfg.model.deep_supervision_strides))
    module.model.set_ground_truth(batch["heatmaps"], batch["offsets"])

    payload = module.validation_step(batch, batch_idx=0)
    # For HeatmapLoggerV2 payload
    for k in [
        "images",
        "pred_heatmaps",
        "pred_offsets",
        "target_heatmaps",
        "gt_coords_img",
        "valid_mask",
    ]:
        assert k in payload


def test_pck_hmap_correct_when_same_peak(patched_module):
    cfg = make_config()
    module = patched_module.BallLitModule(cfg)

    # single-scale tensors [B,1,Hs,Ws]
    B = 2
    Hs, Ws = 16, 20
    h_pred = torch.zeros(B, 1, Hs, Ws)
    h_tgt = torch.zeros(B, 1, Hs, Ws)

    # place same peaks
    h_pred[:, 0, 5, 7] = 1.0
    h_tgt[:, 0, 5, 7] = 1.0

    pck = module._pck(h_pred, h_tgt, threshold_ratio=0.01)
    assert torch.isfinite(pck)
    assert abs(pck.item() - 1.0) < 1e-6


def test_pck_hmap_zero_when_far(patched_module):
    cfg = make_config()
    module = patched_module.BallLitModule(cfg)

    B = 1
    Hs, Ws = 16, 20
    h_pred = torch.zeros(B, 1, Hs, Ws)
    h_tgt = torch.zeros(B, 1, Hs, Ws)
    h_pred[:, 0, 1, 1] = 1.0
    h_tgt[:, 0, 14, 18] = 1.0

    pck = module._pck(h_pred, h_tgt, threshold_ratio=0.05)
    assert abs(pck.item() - 0.0) < 1e-6


def test_pck_offset_img_uses_stride_and_offsets(patched_module):
    cfg = make_config(strides=(4,))
    module = patched_module.BallLitModule(cfg)

    # Image size 128x160, stride 4 -> map 32x40
    H, W = cfg.dataset_img_hw
    s = cfg.model.deep_supervision_strides[-1]
    Hs, Ws = H // s, W // s

    # Build a heatmap with a single peak at (cy, cx) and offsets (dx, dy)
    h = torch.zeros(1, 1, Hs, Ws)
    cx, cy = 7, 10
    h[:, 0, cy, cx] = 1.0

    offs = torch.zeros(1, 2, Hs, Ws)
    dx, dy = 0.25, 0.5
    offs[:, 0, cy, cx] = dx
    offs[:, 1, cy, cx] = dy

    # Expected image coords
    x_img = (cx + dx) * s
    y_img = (cy + dy) * s
    gt_xy_img = torch.tensor([[x_img, y_img]], dtype=torch.float32)
    valid = torch.tensor([[1.0]], dtype=torch.float32)

    pck = module._pck_offset_img(h, offs, gt_xy_img, valid, (H, W), threshold_ratio=0.01)
    assert abs(pck.item() - 1.0) < 1e-6


def test_offset_mask_threshold_zero_pos_is_safe(patched_module):
    # If heatmap is all below offset_tau, mask.sum()==0 -> offset loss branch should be 0
    cfg = make_config(offset_mask_tau=0.99, strides=(8,))
    module = patched_module.BallLitModule(cfg)

    B = 2
    H, W = cfg.dataset_img_hw
    s = cfg.model.deep_supervision_strides[0]
    Hs, Ws = H // s, W // s

    images = torch.randn(B, 3, H, W)
    # below tau
    wh = torch.full((B, 1, Hs, Ws), 0.1, dtype=torch.float32)
    wo = torch.zeros(B, 2, Hs, Ws)
    batch = {
        "image": images,
        "heatmaps": [wh],
        "offsets": [wo],
        "coord": torch.zeros(B, 2),
        "valid_mask": torch.ones(B, 1),
    }

    # Ensure dummy returns something (not used for loss_o due to mask=0)
    module.model.set_ground_truth([wh], [wo])
    out = module._forward_and_loss(batch)
    assert torch.isfinite(out["loss"])
    # offset loss part should be ~0 (mask zero)
    assert out["loss_o"].item() < 1e-8
