# filename: development/court_pose/vit_heatmap_01/tests/test_callbacks.py
import torch

from development.court_pose.vit_heatmap_01.lit_module import CourtLitModule
from development.utils.callbacks.heatmap_logger import HeatmapImageLogger


class _FakeWriter:
    """TensorBoard SummaryWriter 互換の最小スタブ"""

    def __init__(self):
        self.logged = []  # (tag, tensor, step)

    def add_image(self, tag, img_tensor, global_step=None):
        assert isinstance(img_tensor, torch.Tensor)
        self.logged.append((tag, img_tensor, global_step))


class _FakeLogger:
    """trainer.logger.experiment に対応する最小スタブ"""

    def __init__(self):
        self.experiment = _FakeWriter()


class _FakeTrainer:
    """Callback引数のうち必要最小限のみを持つスタブ"""

    def __init__(self, with_logger=True):
        self.logger = _FakeLogger() if with_logger else None


def _make_dummy_batch(cfg):
    b = max(3, cfg.dataset.batch_size)  # num_samples の検証のため >=3 を確保
    c = 3
    h, w = cfg.dataset.img_size
    k = cfg.model.heatmap_channels
    hh, ww = cfg.model.output_size
    images = torch.randn(b, c, h, w)
    target_heatmaps = torch.rand(b, k, hh, ww)
    return (images, target_heatmaps)


def test_heatmap_image_logger_logs_first_batch_and_tags(dummy_config):
    """最初のバッチのみ保持し、Input + KPごとの画像が記録されることを検証"""
    model = CourtLitModule(dummy_config)
    cb = HeatmapImageLogger(num_samples=2)
    trainer = _FakeTrainer(with_logger=True)

    # epoch start
    cb.on_validation_epoch_start(trainer, model)
    assert cb.ready is True
    assert cb.images == []
    assert cb.pred_heatmaps == []
    assert cb.target_heatmaps == []

    # batch 0 の outputs を生成 (実運用同様に validation_step の戻り値を渡す)
    batch0 = _make_dummy_batch(dummy_config)
    out0 = model.validation_step(batch0, 0)

    # 最初のバッチを保存
    cb.on_validation_batch_end(trainer, model, out0, batch0, batch_idx=0)
    assert cb.ready is False
    assert isinstance(cb.images, torch.Tensor)
    assert cb.images.shape[0] == 2  # num_samples が効いている

    # batch 1 は無視される (上書きされない)
    batch1 = _make_dummy_batch(dummy_config)
    out1 = model.validation_step(batch1, 1)
    snapshot = cb.images.clone()
    cb.on_validation_batch_end(trainer, model, out1, batch1, batch_idx=1)
    assert torch.equal(cb.images, snapshot)

    # epoch end でログが書かれる
    cb.on_validation_epoch_end(trainer, model)

    writer = trainer.logger.experiment
    k = dummy_config.model.heatmap_channels
    # 入力画像1件 + KPごとの予測ヒートマップk件
    assert len(writer.logged) == 1 + k

    # 先頭は入力画像
    tag0, img0, step0 = writer.logged[0]
    assert tag0 == "Validation/Input_Images"
    assert isinstance(img0, torch.Tensor)
    assert isinstance(step0, int)
    # 残りは KP ごとの予測ヒートマップ
    for i in range(1, 1 + k):
        tag, img, step = writer.logged[i]
        assert tag == f"Validation_Pred/Heatmap_KP{(i - 1):02d}"
        assert isinstance(img, torch.Tensor)
        assert isinstance(step, int)
        assert step == step0


def test_heatmap_image_logger_no_logger_is_noop(dummy_config):
    """trainer.logger が無い場合は例外なく何もしない (早期 return)"""
    model = CourtLitModule(dummy_config)
    cb = HeatmapImageLogger(num_samples=1)
    trainer = _FakeTrainer(with_logger=False)

    cb.on_validation_epoch_start(trainer, model)
    out = model.validation_step(_make_dummy_batch(dummy_config), 0)
    cb.on_validation_batch_end(trainer, model, out, None, batch_idx=0)

    # ロガーが無いので、この呼び出しは副作用なく終了することのみ確認
    cb.on_validation_epoch_end(trainer, model)


def test_heatmap_image_logger_resets_each_epoch(dummy_config):
    """各エポック開始時に ready フラグと内部バッファがリセットされることを検証"""
    model = CourtLitModule(dummy_config)
    cb = HeatmapImageLogger(num_samples=2)
    trainer = _FakeTrainer(with_logger=True)

    # 1エポック目
    cb.on_validation_epoch_start(trainer, model)
    out0 = model.validation_step(_make_dummy_batch(dummy_config), 0)
    cb.on_validation_batch_end(trainer, model, out0, None, batch_idx=0)
    assert cb.ready is False
    assert isinstance(cb.images, torch.Tensor) and cb.images.shape[0] == 2
    cb.on_validation_epoch_end(trainer, model)

    # 2エポック目開始でリセットされる
    cb.on_validation_epoch_start(trainer, model)
    assert cb.ready is True
    assert cb.images == []
    assert cb.pred_heatmaps == []
    assert cb.target_heatmaps == []
