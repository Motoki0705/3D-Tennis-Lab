**Overview**

- **目的**: テニスコートの15キーポイントをヒートマップ回帰で推定。
- **中核**: ViTエンコーダ + デコーダ + 1x1 Convヘッド。PyTorch Lightning で学習管理。Hydra で設定管理。
- **対象コード**: `development/court_pose/vit_heatmap/`, `development/utils/`。

**Model**

- **エンコーダ**: `VitEncoder` (`model/vit_encoder.py`)
  - `timm.create_model(vit_name, pretrained)`でViTを構築。
  - `forward_features`出力からCLSトークンを除去し、パッチ列を`[B, D, H/P, W/P]`へ整形。
- **デコーダ**: `decoder_name`で切替（`model/vit_heatmap.py`）
  - `simple`: 転置畳み込み + Conv ブロックの逐次アップサンプリング（`decoder/simple_decoder.py`）。
  - `pixel_shuffle_attention`: 1x1拡張→PixelShuffle→DW/PW Conv→CBAM 注意（`decoder/pixel_shuffle_attention_decoder.py`）。
  - `context_pyramid`: Non-Local で文脈注入 + 大カーネル + ASPP + SE（`decoder/context_pyramid_decoder.py`）。
- **ヘッド**: `HeatmapHead`（`model/heatmap_head.py`）で `Conv2d(in, num_keypoints, 1x1)` によりKチャネルのヒートマップ出力。
- **全体**: `VitHeatmapModel` が Encoder→Decoder→Head を接続し、`[B, K, H, W]` ヒートマップを返す。

**Data**

- **データセット**: `CourtKeypointDataset`（`dataset.py`）
  - 入力: 画像ディレクトリとCOCO風JSON（`images`/`annotations`/`categories.keypoints`）。
  - 前処理: 画像をRGB化、リサイズ（`img_size`）、Albumentations変換（後述）。
  - キーポイント: `v>0`のみ拡張対象。最終的に`heatmap_size`へスケールし、各KPに2Dガウシアンを描画（`sigma`）。
  - 出力: 画像`float32`(0–1, Optionで正規化) と K×H×W のターゲットヒートマップ`float32`。
- **変換**: `prepare_transforms`（`development/utils/transformers/keypoint_transformer.py`）
  - Train: `Resize`→`Rotate`→`RandomBrightnessContrast`→`HorizontalFlip`→`Normalize`→`ToTensorV2`。`KeypointParams(format="xy")`。
  - Val/Test: `Resize`→`Normalize`→`ToTensorV2`。
- **DataModule**: `CourtDataModule`（`datamodule.py`）
  - `BaseDataModule`（`development/utils/lightning/base_datamodule.py`）を継承。
  - 全データを`train/val/test`に比率分割（`train_ratio/val_ratio`）。各Subsetの`dataset.transform`を適切に差し替え。
  - DataLoaderは`batch_size/num_workers/pin_memory/persistent_workers`を設定。

**Training**

- **LightningModule**: `CourtLitModule`（`lit_module.py`）
  - モデル: `VitHeatmapModel` を構築（Hydra設定を反映）。
  - 損失: `utils/loss/`のレジストリから取得（`_get_loss_fn`）。
    - `mse`: `nn.MSELoss`、`bce`: `nn.BCEWithLogitsLoss`、`focal`: カスタムFocal、`kldiv`: KLDiv（log_softmax入力）。
    - `focal`選択時は`sigmoid`後に損失計算（`_calculate_loss`）。
  - 指標: PCK（`calculate_pck`）
    - 予測/教師ヒートマップのargmax座標を比較。`threshold_ratio * sqrt(H^2+W^2)`以内を正解。
    - 教師側の各KPチャネル最大値が`>1e-4`のもののみ評価（可視KPマスク）。
  - 事前学習ViTの段階的学習: `freeze_vit_epochs`間はViTをfreezeし、それ以降unfreeze（`on_train_epoch_start`）。
  - 最適化: `AdamW`。ViTとその他のパラメータで別LR（`training.lr`と`training.vit_lr`）。`ReduceLROnPlateau(monitor="val/loss")`。
  - 検証/テスト: `val/loss`・`test/loss`とPCKを`self.log`。可視化用に画像とヒートマップを返却。
- **コールバック/ロギング**
  - `ModelCheckpoint`: `val/loss`監視、bestを保存、ファイル名は`epoch={epoch}-pck={val/PCK@0.05:.4f}`。
  - `EarlyStopping`: `val/loss`監視、`patience`に達したら停止。
  - `LearningRateMonitor`、`RichProgressBar`。
  - `HeatmapImageLogger`（`development/utils/callbacks/heatmap_logger.py`）
    - 各検証エポックで最初のバッチから`num_samples`件をTensorBoardへ画像/ヒートマップとして記録。

**Config**

- **Hydra構成**: `configs/vit_heatmap_v1.yaml`
  - `model`: `vit_name`, `pretrained`, `decoder_channels`, `decoder_name`, `heatmap_channels`, `output_size`。
  - `dataset`: `img_dir`, `annotation_file`, `img_size`, `heatmap_size`, `heatmap_sigma`, `batch_size`, 比率/Loader設定。
  - `training`: `max_epochs`, `lr`, `vit_lr`, `weight_decay`, `precision`, `accelerator`, `devices`, `loss`, `freeze_vit_epochs`。
  - `callbacks`: `checkpoint`, `early_stopping`, `heatmap_logger`。
  - `evaluation`: `pck_threshold`。

**Entry Points**

- **学習**: `development/court_pose/vit_heatmap/train.py`
  - Hydra: `@hydra.main(config_path="configs", config_name="vit_heatmap_v1")`。
  - 実行例（プロジェクトルートで）:
    - `python -m development.court_pose.vit_heatmap.train`
    - 例: デコーダ/損失を変更して実験
      - `python -m development.court_pose.vit_heatmap.train model.decoder_name=context_pyramid training.loss.name=focal`
- **推論/評価**: `infer.py`/`evaluate.py` は雛形（現状未実装）。将来的に`development/utils/loading/model_loader.py`で学習済み重みを読み込み可能。

**Utilities**

- **Loss Registry**: `development/utils/loss/`
  - `loss_registry`へ標準/カスタム損失を登録し、設定名で取得可能。
- **Model Loader**: `development/utils/loading/model_loader.py`
  - 呼び出し元パスから実験名を推定し、`tb_logs/<exp>/<version>/`の`hparams.yaml`と`.ckpt`を用いてモデルを復元。
  - `state_dict`の`model.`接頭辞を除去してロード。Evalモードへ設定。
  - テスト: `development/utils/tests/test_model_loader.py`。

**Training Flow**

- 1. 変換作成: `prepare_transforms(img_size)`
- 2. DataModule初期化: `CourtDataModule(config, train/val/test_transforms)`
- 3. LightningModule初期化: `CourtLitModule(config)`
- 4. Trainer準備: コールバック/ロガー設定
- 5. 実行: `trainer.fit(model, datamodule)` → `trainer.test(ckpt_path="best")`

**Notes**

- `run_experiments.sh` はモジュールパス/変数名にいくつか不整合があり、現状の構成では要修正です。
- `evaluate.py`/`infer.py` は空ファイルのため、使用時は実装が必要です。

**References**

- モデル/学習: `development/court_pose/vit_heatmap/`
- Lightning基盤/損失/変換: `development/utils/`
