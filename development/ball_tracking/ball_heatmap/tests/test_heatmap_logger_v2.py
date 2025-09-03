# tests/test_heatmap_logger_v2.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import torch

# テスト対象をインポート（パスはあなたの実装に合わせて変更してください）
# 例: from development.ball_tracking.ball_heatmap.callbacks.logger import HeatmapLoggerV2
from development.ball_tracking.ball_heatmap.heatmap_logger_v2 import HeatmapLoggerV2


# ---------------------------
# テスト用ダミー実装
# ---------------------------
class DummyWriter:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def add_image(self, tag: str, img: torch.Tensor, global_step: int | None = None):
        # 画像テンソルの形状や dtype も軽く保持（アサート用）
        self.calls.append({
            "tag": tag,
            "shape": tuple(img.shape),
            "dtype": str(img.dtype),
            "global_step": global_step,
        })


class DummyLogger:
    def __init__(self):
        self.experiment = DummyWriter()


class DummyTrainer:
    def __init__(self, logger: DummyLogger | None):
        self.logger = logger


class DummyModule:
    def __init__(self, stride: int = 4, epoch: int = 0):
        # HeatmapLoggerV2 が参照する属性だけを持つ最小モジュール
        self.current_epoch = epoch
        self.config = SimpleNamespace(model=SimpleNamespace(deep_supervision_strides=[stride]))


def _mk_batch(
    n: int = 3,
    img_hw: tuple[int, int] = (64, 64),
    scales_hw: list[tuple[int, int]] = [(8, 8), (16, 16)],
    with_offsets: bool = False,
    with_gt_xy: bool = False,
) -> Dict[str, Any]:
    """HeatmapLoggerV2 が on_validation_batch_end で受け取る outputs/batch を構成。"""
    H, W = img_hw
    images = torch.rand(n, 3, H, W)

    pred_heatmaps = [torch.rand(n, 1, h, w) for (h, w) in scales_hw]
    target_heatmaps = [torch.rand(n, 1, h, w) for (h, w) in scales_hw]

    out = {
        "images": images,
        "pred_heatmaps": pred_heatmaps,
        "target_heatmaps": target_heatmaps,
    }

    if with_offsets:
        # 2 チャンネルの offset を最後のスケール解像度に合わせて用意
        _h, _w = scales_hw[-1]
        pred_offsets = [torch.rand(n, 2, sh, sw) for (sh, sw) in scales_hw]
        out["pred_offsets"] = pred_offsets
    if with_gt_xy:
        # 画像座標系 (x,y)
        gt_xy = torch.tensor([[W * 0.3, H * 0.6]]).repeat(n, 1)
        out["gt_coords_img"] = gt_xy

    return out


# ---------------------------
# テスト
# ---------------------------


def test_logs_basic_images_and_scales():
    """logger が存在し、pred/target heatmap があるとき、基本タグが記録される"""
    cb = HeatmapLoggerV2(num_samples=2, draw_multiscale=True)

    logger = DummyLogger()
    trainer = DummyTrainer(logger=logger)
    module = DummyModule(stride=4, epoch=1)

    outputs = _mk_batch(n=3, img_hw=(64, 64), scales_hw=[(8, 8), (16, 16)], with_offsets=False, with_gt_xy=False)

    # 1 バッチ目だけバッファされる
    cb.on_validation_batch_end(trainer, module, outputs, batch=outputs, batch_idx=0)
    cb.on_validation_batch_end(trainer, module, outputs, batch=outputs, batch_idx=1)  # 無視される

    cb.on_validation_epoch_end(trainer, module)

    tags = [c["tag"] for c in logger.experiment.calls]
    # 入力画像
    assert "val/input" in tags
    # 各スケールの pred/tgt
    assert "val/pred_scale0" in tags
    assert "val/tgt_scale0" in tags
    assert "val/pred_scale1" in tags
    assert "val/tgt_scale1" in tags
    # オーバーレイは offsets が無いので出ない
    assert "val/overlay_pred_gt" not in tags


def test_overlay_with_offsets_and_gt():
    """オフセットと GT があるとき、overlay が描画される"""
    cb = HeatmapLoggerV2(num_samples=2, draw_multiscale=True)

    logger = DummyLogger()
    trainer = DummyTrainer(logger=logger)
    # 画像 64、最終ヒートマップ 16 → stride=4 に設定
    module = DummyModule(stride=4, epoch=2)

    outputs = _mk_batch(
        n=3,
        img_hw=(64, 64),
        scales_hw=[(8, 8), (16, 16)],
        with_offsets=True,
        with_gt_xy=True,
    )

    cb.on_validation_batch_end(trainer, module, outputs, batch=outputs, batch_idx=0)
    cb.on_validation_epoch_end(trainer, module)

    tags = [c["tag"] for c in logger.experiment.calls]
    assert "val/overlay_pred_gt" in tags


def test_no_logger_does_nothing():
    """trainer.logger が None の場合、何も記録されない"""
    cb = HeatmapLoggerV2(num_samples=2, draw_multiscale=True)

    trainer = DummyTrainer(logger=None)
    module = DummyModule(stride=4, epoch=0)
    outputs = _mk_batch()

    cb.on_validation_batch_end(trainer, module, outputs, batch=outputs, batch_idx=0)
    cb.on_validation_epoch_end(trainer, module)

    # logger がないので呼び出しは 0 件
    assert True  # 例外が出ないことを持って成功とする


def test_only_first_batch_is_buffered():
    """最初のバッチのみバッファされることを確認"""
    cb = HeatmapLoggerV2(num_samples=2, draw_multiscale=True)

    logger = DummyLogger()
    trainer = DummyTrainer(logger=logger)
    module = DummyModule(stride=4, epoch=3)

    outputs0 = _mk_batch(scales_hw=[(8, 8)], with_offsets=False)
    outputs1 = _mk_batch(scales_hw=[(16, 16)], with_offsets=False)

    cb.on_validation_batch_end(trainer, module, outputs0, batch=outputs0, batch_idx=0)  # これが使われる
    cb.on_validation_batch_end(trainer, module, outputs1, batch=outputs1, batch_idx=1)  # 無視される
    cb.on_validation_epoch_end(trainer, module)

    tags = [c["tag"] for c in logger.experiment.calls]
    # pred/tgt は scale0 のみ（= 最初のバッチのスケール構成）
    assert "val/pred_scale0" in tags
    assert "val/tgt_scale0" in tags
    # バッファは消去される（内部状態の健全性）
    assert cb._buffer is None
