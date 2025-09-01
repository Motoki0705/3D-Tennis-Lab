# tests/test_ball_datamodule.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import importlib
from argparse import Namespace
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pytest
from torch.utils.data import WeightedRandomSampler, RandomSampler, DataLoader

# ここをあなたの実モジュールパスに差し替えてください
# 例) "experiments.ball.datamodule"
MODULE_DATAMODULE_PATH = "development.ball_tracking.ball_heatmap.datamodule"  # <-- CHANGE ME
MODULE_DATASET_PATH = "development.ball_tracking.ball_heatmap.dataset"  # <-- CHANGE ME


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture(scope="module")
def mod_dm():
    try:
        return importlib.import_module(MODULE_DATAMODULE_PATH)
    except Exception as e:
        pytest.skip(f"Import failed. Set MODULE_DATAMODULE_PATH correctly. Error: {e}")


@pytest.fixture(scope="module")
def mod_ds():
    try:
        return importlib.import_module(MODULE_DATASET_PATH)
    except Exception as e:
        pytest.skip(f"Import failed. Set MODULE_DATASET_PATH correctly. Error: {e}")


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
    # 簡易で決定論的な Transform を返す
    tf = A.Compose(
        [A.Resize(img_size[0], img_size[1]), ToTensorV2()],
        keypoint_params=A.KeypointParams(format="xy"),
    )
    return tf, tf


def _make_config(tmpdir: Path, mod_ds):
    # SimpleNamespace で必要項目のみ用意

    return Namespace(
        dataset=Namespace(
            img_dir=str(tmpdir),
            annotation_file=str(tmpdir / "ann.json"),
            img_size=(64, 80),
            negatives="use",
            heatmap=Namespace(strides=[8, 4], sigmas=[1.2, 0.8]),
            sequence=Namespace(length=1, frame_stride=1, center_version=None, center_span=1),
            version_weights={"v1": 1.0, "v2": 0.1},
            v_mix_schedule=[
                {"epoch_le": 0, "v1": 0.5, "v2": 0.5},
                {"epoch_le": 2, "v1": 0.8, "v2": 0.2},
            ],
            split=Namespace(train_ratio=0.5, val_ratio=0.25),
            loader=Namespace(
                sampler="weighted",  # "uniform" / "balanced" / others->weighted
                batch_size=2,
                num_workers=0,  # テスト簡略化のため 0
                pin_memory=False,
                persistent_workers=False,
            ),
        )
    )


# -----------------------------
# Tests: BallDataModule
# -----------------------------
def test_setup_and_transforms_applied(tmp_path: Path, mod_dm, mod_ds, monkeypatch):
    # データ: 1 クリップ 8 フレーム (v1多め, v2少なめ)
    seq = [(10, 10, 1), (15, 15, 1), (20, 20, 1), (25, 25, 2), (30, 30, 1), (35, 35, 1), (40, 40, 2), (45, 45, 1)]
    ann_path = _build_coco(tmp_path, ["clipA"], {"clipA": seq})
    cfg = _make_config(tmp_path, mod_ds)
    cfg.dataset.annotation_file = str(ann_path)

    # prepare_transforms を軽量なフェイクに差替え
    monkeypatch.setattr(mod_dm, "prepare_transforms", _fake_prepare_transforms)

    dm = mod_dm.BallDataModule(cfg)
    dm.setup()

    # 分割長の整合性
    n_total = len(dm.full_dataset)  # provided by BaseDataModule
    n_train = len(dm.train_dataset)
    n_val = len(dm.val_dataset)
    n_test = len(dm.test_dataset)
    assert n_train + n_val + n_test == n_total

    # それぞれ transform が ReplayCompose 化されていること
    import albumentations as A

    assert isinstance(dm.train_dataset.dataset.transform, A.ReplayCompose)
    assert isinstance(dm.val_dataset.dataset.transform, A.ReplayCompose)
    assert isinstance(dm.test_dataset.dataset.transform, A.ReplayCompose)


def test_train_dataloader_sampler_variants(tmp_path: Path, mod_dm, mod_ds, monkeypatch):
    # データ: v1:6, v2:2 くらいの分布
    seq = [(10, 10, 1), (15, 15, 1), (20, 20, 1), (25, 25, 2), (30, 30, 1), (35, 35, 1), (40, 40, 2), (45, 45, 1)]
    ann_path = _build_coco(tmp_path, ["clipA"], {"clipA": seq})

    # 共通 config
    cfg = _make_config(tmp_path, mod_ds)
    cfg.dataset.annotation_file = str(ann_path)
    monkeypatch.setattr(mod_dm, "prepare_transforms", _fake_prepare_transforms)

    # 1) uniform -> RandomSampler
    cfg.dataset.loader.sampler = "uniform"
    dm1 = mod_dm.BallDataModule(cfg)
    dm1.setup()
    loader1: DataLoader = dm1.train_dataloader()
    assert isinstance(loader1.sampler, RandomSampler)

    # 2) balanced -> WeightedRandomSampler（バージョン数で逆比例重み）
    cfg.dataset.loader.sampler = "balanced"
    dm2 = mod_dm.BallDataModule(cfg)
    dm2.setup()
    loader2: DataLoader = dm2.train_dataloader()
    assert isinstance(loader2.sampler, WeightedRandomSampler)
    # 重み配列の長さ == サブセット長
    assert len(loader2.sampler.weights) == len(dm2.train_dataset)

    # 3) weighted（明示重み） -> WeightedRandomSampler
    cfg.dataset.loader.sampler = "weighted"
    dm3 = mod_dm.BallDataModule(cfg)
    dm3.setup()
    loader3: DataLoader = dm3.train_dataloader()
    assert isinstance(loader3.sampler, WeightedRandomSampler)
    assert len(loader3.sampler.weights) == len(dm3.train_dataset)


def test_build_sample_weights_policies(tmp_path: Path, mod_dm, mod_ds, monkeypatch):
    # v1: 6, v2: 2
    seq = [(10, 10, 1), (15, 15, 1), (20, 20, 1), (25, 25, 2), (30, 30, 1), (35, 35, 1), (40, 40, 2), (45, 45, 1)]
    ann_path = _build_coco(tmp_path, ["clipA"], {"clipA": seq})
    cfg = _make_config(tmp_path, mod_ds)
    cfg.dataset.split.train_ratio = 1.0
    cfg.dataset.split.val_ratio = 0.0
    cfg.dataset.annotation_file = str(ann_path)
    monkeypatch.setattr(mod_dm, "prepare_transforms", _fake_prepare_transforms)

    # balanced
    cfg.dataset.loader.sampler = "balanced"
    dm = mod_dm.BallDataModule(cfg)
    dm.setup()
    # Subset の indices を取得
    idx = dm.train_dataset.indices
    w_bal = dm._build_sample_weights(idx)
    # v1 と v2 で重みが違う（件数の逆数）
    ds = dm.train_dataset.dataset
    vs = np.array([ds.versions[i] for i in idx])
    assert w_bal.shape[0] == len(idx)
    assert not np.allclose(w_bal[vs == "v1"].mean(), w_bal[vs == "v2"].mean())

    # weighted（バージョン重みを尊重）
    cfg.dataset.loader.sampler = "weighted"
    dm2 = mod_dm.BallDataModule(cfg)
    dm2.setup()
    idx2 = dm2.train_dataset.indices
    w_w = dm2._build_sample_weights(idx2)
    # v1 の方が重みが大（v1:1.0, v2:0.1）
    ds2 = dm2.train_dataset.dataset
    vs2 = np.array([ds2.versions[i] for i in idx2])
    assert w_w[vs2 == "v1"].mean() > w_w[vs2 == "v2"].mean()


def test_update_sampling_for_epoch_schedule(tmp_path: Path, mod_dm, mod_ds, monkeypatch):
    seq = [(10, 10, 1), (15, 15, 1), (20, 20, 2), (25, 25, 2)]
    ann_path = _build_coco(tmp_path, ["clipA"], {"clipA": seq})
    cfg = _make_config(tmp_path, mod_ds)
    cfg.dataset.annotation_file = str(ann_path)
    cfg.dataset.loader.sampler = "weighted"
    monkeypatch.setattr(mod_dm, "prepare_transforms", _fake_prepare_transforms)

    dm = mod_dm.BallDataModule(cfg)
    dm.setup()

    # epoch 0 -> schedule[0] 適用
    dm.update_sampling_for_epoch(0)
    assert dm._current_version_weights == {"v1": 0.5, "v2": 0.5}

    # epoch 2 -> schedule[1] 適用
    dm.update_sampling_for_epoch(2)
    assert dm._current_version_weights == {"v1": 0.8, "v2": 0.2}

    # epoch 99 -> fallback to static
    dm.update_sampling_for_epoch(99)
    assert dm._current_version_weights == cfg.dataset.version_weights
