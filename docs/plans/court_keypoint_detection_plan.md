# コートキーポイント検出モデル開発プラン（PyTorch Lightning 版）

## 1. 目的

テニスコートの15個のキーポイントを検出するモデルを **PyTorch Lightning** で開発する。
本モデルは上流工程「3D情報抽出」における「コート姿勢推定」の基盤となる。

---

## 2. モデル仕様

* **アーキテクチャ**

  * **エンコーダ**: Vision Transformer (ViT, timm)
  * **デコーダ**: シンプルなアップサンプリング（ConvTranspose + Conv）
  * **出力**: 15チャンネルのヒートマップ（各チャンネルが1つのキーポイントに対応）

* **損失関数**: Mean Squared Error (MSE)

* **評価指標**: PCK (Percentage of Correct Keypoints)

---

## 3. ディレクトリ構成

```
development/court_pose/01_vit_heatmap/
├─ configs/
│   └─ vit_heatmap_v1.yml       # ハイパーパラメータ・Trainer設定
├─ datamodule.py                # LightningDataModule
├─ dataset.py                   # PyTorch Dataset（個別実装）
├─ model_components/
│   ├─ vit_encoder.py            # ViTエンコーダ構築
│   ├─ decoder.py                # アップサンプリングデコーダ
│   └─ heatmap_head.py           # 最終15ch出力
├─ lit_module.py                 # LightningModule（モデル+学習ロジック）
├─ train.py                      # エントリポイント（学習）
├─ evaluate.py                   # 推論・PCK評価
└─ tests/
    ├─ test_dataset.py
    ├─ test_datamodule.py
    └─ test_lit_module.py
```

---

## 4. コンポーネント別タスク

### 4.1 Dataset（`dataset.py`）

* **入力**: `data/processed/court/annotation.json`
* **出力**: `(image_tensor, heatmap_tensor)`
* **機能**

  * 画像ロード & リサイズ（configでサイズ指定）
  * キーポイント座標 → ガウシアンヒートマップ生成（15チャンネル）
  * Albumentations による拡張（train/valで異なる設定）
* **インターフェース**

  ```python
  __getitem__(self, idx) -> (Tensor[B, 3, H, W], Tensor[B, 15, Hh, Wh])
  ```

---

### 4.2 DataModule（`datamodule.py`）

* LightningDataModule を継承
* **責務**

  * train/val/test Dataset の構築
  * DataLoaderの返却
  * augmentationパイプラインの定義
* **主要メソッド**

  ```python
  prepare_data(self): pass  # 必要ならダウンロード
  setup(self, stage): ...   # Datasetの初期化
  train_dataloader(self)
  val_dataloader(self)
  test_dataloader(self)
  ```

---

### 4.3 モデルコンポーネント（`model_components/`）

* **`vit_encoder.py`**

  * timmで事前学習済みViTをロード
  * 出力はパッチ埋め込み → feature map に変換
* **`decoder.py`**

  * ConvTranspose2d / Upsample を使って指定ヒートマップ解像度まで拡大
* **`heatmap_head.py`**

  * 1x1 Convで出力チャンネルを15に固定

---

### 4.4 LightningModule（`lit_module.py`）

* **責務**

  * モデル構築（encoder + decoder + head）
  * 損失計算（MSE）
  * 評価指標（PCK）
  * Optimizer / Scheduler の設定
* **主要メソッド**

  ```python
  training_step(self, batch, batch_idx)
  validation_step(self, batch, batch_idx)
  configure_optimizers(self)
  ```
* **ログ**

  * `self.log('train/loss', loss, prog_bar=True)`
  * `self.log('val/loss', loss, prog_bar=True)`
  * `self.log('val/PCK', pck, prog_bar=True)`

---

### 4.5 学習スクリプト（`train.py`）

* **処理フロー**

  1. `OmegaConf` で config 読み込み
  2. Logger（TensorBoard/WandB）設定
  3. Callback 設定（ModelCheckpoint, EarlyStopping, LearningRateMonitor）
  4. `CourtDataModule` & `CourtLitModule` 初期化
  5. `Trainer` 起動

  ```python
  trainer.fit(model, datamodule=dm)
  ```
* **Trainer 設定例（YAMLで管理）**

  ```yaml
  trainer:
    max_epochs: 100
    accelerator: gpu
    devices: 1
    precision: 16
  ```

---

### 4.6 評価スクリプト（`evaluate.py`）

* **処理フロー**

  1. チェックポイントから `CourtLitModule.load_from_checkpoint()` でモデル復元
  2. `trainer.test(...)` または `trainer.predict(...)`
  3. 出力ヒートマップから座標抽出（argmax or subpixel refinement）
  4. PCK計算・可視化・保存

---

### 4.7 テスト（`tests/`）

* **`test_dataset.py`**

  * Dataset が正しい shape のテンソルを返すか
  * ヒートマップがキーポイント位置にピークを持つか
* **`test_datamodule.py`**

  * DataLoader が batch サイズ通り動作するか
* **`test_lit_module.py`**

  * forward が期待形状の出力を返すか
  * training\_step がスカラー loss を返すか

---

## 5. Config ファイル例（`configs/vit_heatmap_v1.yml`）

```yaml
model:
  vit_name: vit_base_patch16_224
  pretrained: true
  decoder_channels: 256
  heatmap_channels: 15
  heatmap_size: [56, 56]

data:
  img_size: [224, 224]
  batch_size: 32
  num_workers: 8
  augmentation:
    train:
      - name: RandomRotate
        max_angle: 10
      - name: RandomBrightnessContrast
        p: 0.3
    val: []

training:
  max_epochs: 100
  lr: 1e-4
  weight_decay: 0.01
  precision: 16
  accelerator: gpu
  devices: 1

checkpoint:
  monitor: val/PCK
  mode: max
  save_top_k: 3
```

---

## 6. 開発ステップ

1. **雛形作成**

   * ディレクトリ・ファイルスケルトン生成
2. **Dataset実装**

   * ヒートマップ生成ロジック含む
3. **DataModule実装**
4. **モデルコンポーネント実装**

   * encoder / decoder / head
5. **LightningModule実装**
6. **train.py実装**

   * config読み込み → Trainer起動
7. **テスト実装**

   * dataset / datamodule / lit\_module
8. **小規模データで動作確認**
9. **本学習実行**
10. **evaluate.pyで評価・可視化**
11. **成果物を保存**
