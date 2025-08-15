# filename: development/court_pose/01_vit_heatmap/tests/test_lit_module.py
import torch

from development.court_pose.vit_heatmap_01.lit_module import CourtLitModule


def test_lit_module_forward(dummy_config):
    """モデルのフォワードパスが正しい形状の出力を返すかテスト"""
    model = CourtLitModule(dummy_config)

    batch_size = dummy_config.dataset.batch_size
    img_size = dummy_config.dataset.img_size

    # ダミー入力テンソル
    dummy_input = torch.randn(batch_size, 3, img_size[0], img_size[1])

    # フォワードパスを実行
    output = model(dummy_input)

    heatmap_channels = dummy_config.model.heatmap_channels
    heatmap_size = dummy_config.model.output_size

    assert output.shape == (batch_size, heatmap_channels, heatmap_size[0], heatmap_size[1])


def test_lit_module_loss(dummy_config):
    """training_step がスカラー、validation_step が所定の辞書を返すことを検査"""
    model = CourtLitModule(dummy_config)

    b = dummy_config.dataset.batch_size
    c = 3
    h, w = dummy_config.dataset.img_size
    k = dummy_config.model.heatmap_channels
    hh, ww = dummy_config.model.output_size

    images = torch.randn(b, c, h, w)
    target_heatmaps = torch.rand(b, k, hh, ww)
    batch = (images, target_heatmaps)

    # training_step
    train_loss = model.training_step(batch, 0)
    assert isinstance(train_loss, torch.Tensor)
    assert train_loss.shape == torch.Size([])  # スカラー
    assert train_loss.requires_grad is True

    # validation_step
    out = model.validation_step(batch, 0)
    # 返却構造
    assert set(out.keys()) == {"loss", "images", "pred_heatmaps", "target_heatmaps"}
    # 型・形状
    assert isinstance(out["loss"], torch.Tensor) and out["loss"].shape == torch.Size([])
    assert out["images"].shape == (b, c, h, w)
    assert out["pred_heatmaps"].shape == (b, k, hh, ww)
    assert out["target_heatmaps"].shape == (b, k, hh, ww)
    # CPUへ移っていること(Callbackの期待に一致)
    assert out["images"].device.type == "cpu"
    assert out["pred_heatmaps"].device.type == "cpu"
    assert out["target_heatmaps"].device.type == "cpu"


def test_configure_optimizers(dummy_config):
    """configure_optimizersがoptimizerとschedulerを返すかテスト"""
    model = CourtLitModule(dummy_config)
    optimizers = model.configure_optimizers()

    assert "optimizer" in optimizers
    assert "lr_scheduler" in optimizers
    assert isinstance(optimizers["optimizer"], torch.optim.Optimizer)


def test_pck_calculation(dummy_config):
    """PCK計算ロジックをテスト"""
    model = CourtLitModule(dummy_config)
    k, h, w = 15, 56, 56

    # ケース1: 予測とターゲットが完全に一致
    pred = torch.zeros(1, k, h, w)
    target = torch.zeros(1, k, h, w)
    pred[0, 0, 10, 20] = 1.0  # 予測
    target[0, 0, 10, 20] = 1.0  # 正解
    pck_perfect = model.calculate_pck(pred, target)
    assert torch.isclose(pck_perfect, torch.tensor(1.0))

    # ケース2: 予測が閾値外にずれている
    pred[0, 1, 30, 30] = 1.0
    target[0, 1, 10, 10] = 1.0
    pck_imperfect = model.calculate_pck(pred, target)
    # 2つのキーポイントのうち1つが正解なので0.5
    assert torch.isclose(pck_imperfect, torch.tensor(0.5))

    # ケース3: ターゲットがない(v=0)場合は評価対象外
    pred = torch.rand(1, k, h, w)
    target = torch.zeros(1, k, h, w)  # 全てラベルなし
    pck_no_target = model.calculate_pck(pred, target)
    assert pck_no_target == 0.0 or torch.isnan(pck_no_target)
