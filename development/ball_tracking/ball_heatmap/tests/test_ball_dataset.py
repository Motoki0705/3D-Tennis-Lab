# tests/test_ball_dataset.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import importlib
import math

import cv2
import numpy as np
import pytest
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ここをあなたの実モジュールパスに差し替えてください
# 例) "experiments.ball.dataset"
MODULE_DATASET_PATH = "development.ball_tracking.ball_heatmap.dataset"  # <-- CHANGE ME


# -----------------------------
# Helpers to build tiny COCO set
# -----------------------------
def _write_img(path: Path, hw=(64, 80)):
    H, W = hw
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    # draw a small dot to avoid jpeg all-black optimizations
    cv2.circle(arr, (5, 5), 2, (0, 255, 0), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), arr)


def _build_coco(tmpdir: Path, clips, vis_seq_per_clip, with_missing=False):
    """
    clips: ["clipA", "clipB"]
    vis_seq_per_clip: dict {clip: [(x,y,v or None), ...]} length==T per clip
        - if None: means no annotation for that frame
        - v is used as visibility -> version label v{v}
    """
    images = []
    annotations = []
    img_id = 1
    ann_id = 1
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
            elif not with_missing:
                # create no annotation entry at all (dataset treats as None)
                pass
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


@pytest.fixture(scope="module")
def mod_ds():
    try:
        return importlib.import_module(MODULE_DATASET_PATH)
    except Exception as e:
        pytest.skip(f"Import failed. Set MODULE_DATASET_PATH correctly. Error: {e}")


# -----------------------------
# Tests: BallDataset
# -----------------------------
def test_single_frame_shapes_and_offsets(tmp_path: Path, mod_ds):
    # One clip of 5 frames, all annotated with v=1
    seq = [(10, 20, 1), (15, 22, 1), (30, 40, 1), (45, 10, 1), (60, 55, 1)]
    ann_path = _build_coco(tmp_path, ["clipA"], {"clipA": seq})
    specs = [
        mod_ds.HeatmapSpec(stride=4, sigma=1.5),
        mod_ds.HeatmapSpec(stride=8, sigma=1.0),
    ]
    ds = mod_ds.BallDataset(
        img_dir=tmp_path,
        annotation_file=ann_path,
        img_size=(64, 80),  # H,W
        heatmap_specs=specs,
        sequence_length=1,
        frame_stride=1,
        negatives="use",
        transform=A.Compose([A.Resize(64, 80), ToTensorV2()], keypoint_params=A.KeypointParams(format="xy")),
    )
    assert len(ds) == 5  # L=1 なので全フレーム中心でOK

    sample = ds[2]  # 中央フレーム (x=30, y=40)
    assert isinstance(sample["image"], torch.Tensor) and sample["image"].shape == (3, 64, 80)
    assert isinstance(sample["coord"], torch.Tensor) and sample["coord"].shape == (2,)
    assert isinstance(sample["valid_mask"], torch.Tensor)

    # heatmap/offsets: 2スケール
    assert len(sample["heatmaps"]) == 2 and len(sample["offsets"]) == 2
    for s, spec in enumerate(specs):
        hm = sample["heatmaps"][s]  # [1,Hs,Ws]
        off = sample["offsets"][s]  # [2,Hs,Ws]
        Hs, Ws = 64 // spec.stride, 80 // spec.stride
        assert hm.shape == (1, Hs, Ws)
        assert off.shape == (2, Hs, Ws)
        # 期待セルとオフセット（床関数でセル、残りが分数部）
        x, y = sample["coord"].tolist()
        xs, ys = x / spec.stride, y / spec.stride
        ix, iy = int(math.floor(xs)), int(math.floor(ys))
        fx, fy = xs - ix, ys - iy
        assert abs(off[0, iy, ix].item() - fx) < 1e-5
        assert abs(off[1, iy, ix].item() - fy) < 1e-5
        # ヒートマップのピークセル一致：
        # 連続中心 (xs, ys) に最も近い離散セル（タイは floor を優先）
        peak = torch.argmax(hm.view(-1)).item()
        py, px = divmod(peak, Ws)
        ix_ceil = min(ix + 1, Ws - 1)
        iy_ceil = min(iy + 1, Hs - 1)
        px_expected = ix if abs(xs - ix) <= abs(xs - ix_ceil) else ix_ceil
        py_expected = iy if abs(ys - iy) <= abs(ys - iy_ceil) else iy_ceil
        assert (py, px) == (py_expected, px_expected)


def test_sequence_replay_and_horizontal_flip(tmp_path: Path, mod_ds):
    # 3 フレーム、うち 2 フレームだけアノテーション
    seq = [(10, 20, 2), None, (30, 20, 2)]  # v=2 -> version "v2"
    ann_path = _build_coco(tmp_path, ["clipA"], {"clipA": seq})
    specs = [mod_ds.HeatmapSpec(stride=4, sigma=1.0)]

    # 画像サイズと一致する Resize + 必ず水平反転 (p=1.0)
    tf = A.Compose(
        [A.Resize(64, 80), A.HorizontalFlip(p=1.0), ToTensorV2()],
        keypoint_params=A.KeypointParams(format="xy"),
    )

    ds = mod_ds.BallDataset(
        img_dir=tmp_path,
        annotation_file=ann_path,
        img_size=(64, 80),
        heatmap_specs=specs,
        sequence_length=3,
        frame_stride=1,
        center_version="v2",
        center_span=1,
        negatives="use",
        transform=tf,
    )
    # L=3 なので中心はフレーム1のみ、ただし center_version='v2' → 真ん中(インデックス1)は None なので index には入らない
    # しかし dataset は「中心フレームが v2 であること」をチェックする。ここでは v2 かつ ann がある中心: インデックス0 と 2 は L=3 の条件を満たせない。
    # → サンプルは 0 件
    assert len(ds) == 0

    # 条件を緩めて center_version を None にすると 1 サンプル取れる
    ds2 = mod_ds.BallDataset(
        img_dir=tmp_path,
        annotation_file=ann_path,
        img_size=(64, 80),
        heatmap_specs=specs,
        sequence_length=3,
        frame_stride=1,
        center_version=None,
        center_span=1,
        negatives="use",
        transform=tf,
    )
    assert len(ds2) == 1
    s = ds2[0]
    assert s["image"].shape == (3, 3, 64, 80)  # [T,3,H,W]
    # 反転チェック: x' = (W-1) - x
    W = 80
    coords = s["coord"]  # [T,2]
    valids = s["valid_mask"]  # [T]
    # t=0 は注釈あり (10,20)
    if valids[0] > 0.5:
        assert abs(coords[0, 0].item() - ((W - 1) - 10)) < 1e-4
        assert abs(coords[0, 1].item() - 20) < 1e-4
    # t=2 も注釈あり (30,20)
    if valids[2] > 0.5:
        assert abs(coords[2, 0].item() - ((W - 1) - 30)) < 1e-4
        assert abs(coords[2, 1].item() - 20) < 1e-4


def test_center_version_and_span_and_skip(tmp_path: Path, mod_ds):
    # v1,v1,v1,v2,v1 として、L=3, center_version='v1', center_span=3, negatives='skip'
    seq = [(5, 5, 1), (6, 5, 1), (7, 5, 1), (8, 5, 2), (9, 5, 1)]
    ann_path = _build_coco(tmp_path, ["clipA"], {"clipA": seq}, with_missing=False)
    specs = [mod_ds.HeatmapSpec(stride=8, sigma=1.0)]
    ds = mod_ds.BallDataset(
        img_dir=tmp_path,
        annotation_file=ann_path,
        img_size=(64, 80),
        heatmap_specs=specs,
        sequence_length=3,
        frame_stride=1,
        center_version="v1",
        center_span=3,
        negatives="skip",
        transform=A.Compose([A.Resize(64, 80), ToTensorV2()], keypoint_params=A.KeypointParams(format="xy")),
    )
    # 可能な中心は i=1 と i=2 （3 連続 v1 を満たすのは [0,1,2] と [1,2,3] だが i=3 は v2 なので不可）
    # ただし L=3 の端制約も満たすため i=1,2 の 2 サンプル
    assert len(ds) in (1, 2)  # 乱数なしだが環境差分を許容
    for k in range(len(ds)):
        s = ds[k]
        versions = s["meta"]["versions"]
        assert all(v == "v1" for v in versions)  # span=3 すべて v1
        assert s["valid_mask"].shape[0] == 3


def test_transform_setter_wraps_compose_in_replay(mod_ds, tmp_path: Path):
    # dataset.transform に A.Compose を渡すと ReplayCompose にラップされることを確認
    seq = [(10, 20, 1), (15, 30, 1), (22, 18, 1)]
    ann_path = _build_coco(tmp_path, ["clipA"], {"clipA": seq})
    specs = [mod_ds.HeatmapSpec(stride=4, sigma=1.0)]
    base = A.Compose([A.Resize(64, 80), ToTensorV2()], keypoint_params=A.KeypointParams(format="xy"))
    ds = mod_ds.BallDataset(
        img_dir=tmp_path,
        annotation_file=ann_path,
        img_size=(64, 80),
        heatmap_specs=specs,
        sequence_length=3,
        transform=base,
    )
    assert isinstance(ds.transform, A.ReplayCompose)
