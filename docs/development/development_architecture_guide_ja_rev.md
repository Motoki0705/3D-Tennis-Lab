# Development Architecture Guide（改訂版・日本語）

このドキュメントは、3D Tennis Lab プロジェクトにおける実験実装の共通基盤 **`development/core`** の設計と使い方を解説します。新しい実験を**作成・実行・管理**する開発者向けです。

> **重要な改訂点（本版）**
> 依存関係は **`development/<project>/<experiment> -> core`** の一方向です。
> ここで **`experiments` はディレクトリ名ではありません**。過去表記の「`experiments -> core`」は誤解を招くため、本版では明示的に **`development/<project>/<experiment>`** というパス表記に統一します。

---

## 1. コアコンセプト

本アーキテクチャは、**再利用性の最大化**と**実験構造の一貫性**を狙いとしています。各実験はコアのボイラープレートに依存し、開発者はモデルや損失などの差分に集中できます。

- **依存の明確化**: **`development/<project>/<experiment> -> development/core`**。コアは実験に依存しません。
- **設定駆動**: 構成は **[Hydra](https://hydra.cc/)** で管理し、実験はコアの既定設定を **上書き・拡張** して定義します。
- **単一エントリポイント**: 学習・評価・推論は **`development/core/run.py`** から実行します。

---

## 2. `development/core` 詳説

`development/core` は、実験を組み立て・実行するための**共通部品**を提供します。

### 2.1. エントリポイント: `run.py`

本スクリプトはワークフロー全体の**オーケストレータ**です。

1. Hydra により、コアのベース設定と実験の差分設定を**合成**して最終設定を得る。
2. 設定に基づき必要コンポーネントを**インスタンス化**する：
   - DataModule（`development.core.lightning.base_datamodule.BaseDataModule`）
   - Model（`torch.nn.Module`）
   - Loss Function
   - Metrics
   - LightningModule（`development.core.lightning.base_lit_module.BaseLitModule`）
   - Callbacks
3. PyTorch Lightning の `Trainer` を初期化。
4. 指定タスク（`fit` / `validate` / `test` / `predict`）を実行。

### 2.2. 設定: `configs/`

コア側の **デフォルト Hydra 設定** を提供します。

- `config.yaml`: 主要構造と include（`data` / `training` / `callbacks` を束ねる）
- `training/lightning_default.yaml`: Lightning `Trainer` と Optimizer / Scheduler の既定
- `callbacks/lightning_default.yaml`: `ModelCheckpoint` / `LearningRateMonitor` / `EarlyStopping` など標準コールバック
- `data/*.yaml`: データセットごとの既定（例: `ball.yaml`, `court.yaml`）。実験で上書きします。

### 2.3. Lightning コンポーネント: `lightning/`

- **`base_lit_module.py`**: 汎用 `LightningModule`。学習/検証/テスト手順、モデル・損失・指標の配線、最適化/スケジューラ設定を実装。実験側は継承で拡張可能。
- **`base_datamodule.py`**: 汎用 `LightningDataModule`。学習/検証/テスト分割と各 `DataLoader` を提供。

### 2.4. データ処理: `datasets/` と `data_core/`

データパイプラインの中核です。

- **`datasets/`**: タスク高レベルの `torch.utils.data.Dataset` 実装群。
  - `base_sequence.py`: 連続フレーム（クリップ）を COCO 形式アノテーションから列挙・読込する `BaseSequenceDataset` 抽象基底。
  - `ball.py` / `player.py` / `court.py`: タスク別実装（例: ヒートマップ生成、バウンディングボックス抽出）。
- **`data_core/`**: 低レベルの再利用ユーティリティ。
  - `build.py`: `build_dataset(name, ...)` ファクトリ
  - `coco_io.py`: COCO JSON 読込/解析
  - `grouping.py`: メタデータやディレクトリ構造に基づくクリップ化
  - `targets.py`: 監督信号生成（例: `make_heatmaps_xy`, `extract_player_bboxes_classes`）
  - `replay.py`: Albumentations を**クリップ全フレームへ一貫適用**するアダプタ

### 2.5. その他コアモジュール

- **`augment/`**: Albumentations 変換パイプラインのファクトリ
- **`callbacks/`**: `build_callbacks` と、予測ヒートマップを TensorBoard に可視化する `HeatmapLogger` など
- **`loss/`**: 損失関数レジストリ（名前指定で生成）
- **`loading/`**: 事前学習済みチェックポイントの読込 `load_model_from_checkpoint`

---

## 3. 実験実装ガイド（例：`development/ball_tracking/dino_sequencial`）

### 3.1. ディレクトリ構成（**改訂**）

実験は**自己完結パッケージ**として配置します。**`experiments/` というディレクトリは存在しません。**

```
development/<project>/<experiment>/
├── model/                  # ★ 必須: モデル定義とファクトリ
│   ├── architecture.py
│   └── factory.py
├── dataset/                # 任意: データローディングの実験側拡張
│   └── datamodule.py
├── loss/                   # 任意: 実験固有の損失
│   └── strategy.py
└── configs/                # ★ 必須: core との差分のみを配置
    ├── config.yaml
    ├── data/
    └── training/
```

- `<project>` 例: `ball_tracking` / `player_detection` など
- `<experiment>` 例: `dino_sequencial` など

### 3.2. 設定（`configs/`）

実験側 `configs/` には、**コアの既定設定との差分のみ**を置きます。

**`configs/config.yaml`（例）**: 最上位の実験設定。コア既定を `override` し、実験のファクトリにルーティングします。

```yaml
# @package _global_

defaults:
  - override /data: dino_sequencial
  - override /training: dino_sequencial
  - override /callbacks: dino_sequencial
  - _self_

project: "ball_tracking"
experiment: "dino_sequencial"

# 実験側ファクトリを指す
datamodule:
  _target_: development.ball_tracking.dino_sequencial.dataset.datamodule.build_datamodule
  cfg: ${data}

model:
  _target_: development.ball_tracking.dino_sequencial.model.factory.create_model
  # ... model-specific params

loss:
  _target_: development.ball_tracking.dino_sequencial.loss.strategy.build_loss
  # ... loss-specific params

lit_module:
  _target_: development.ball_tracking.dino_sequencial.model.factory.create_lit_module
  # ... lit_module-specific params
```

### 3.3. データセット（`dataset/datamodule.py`）

多くの実験は `development/core` の Dataset を再利用できます。`dino_sequencial` では次のパターンを示します：

1. Hydra の data 設定を受け取る。
2. `development.core.data_core.build.build_dataset` を通じて、コアの `BallSequenceDataset` を**名前で生成**。
3. 画像パス、系列長、ヒートマップサイズ等を設定から渡す。
4. 生成した Dataset をコアの `BaseDataModule` で**ラップ**する。

これにより、データセットの複雑な実装を再度書く必要がなく、実験は**設定中心**に保たれます。

### 3.4. モデル（`model/`）

実験固有の**必須コンポーネント**です。

- **`architecture.py`**: 純粋な `nn.Module`（例: `SequenceHeatmapNet`）。Lightning 依存は持たない。
- **`factory.py`**: `run.py` から呼ばれるファクトリ関数群。
  - `create_model()`: `architecture.py` のネットワークを生成。
  - `create_lit_module()`: `BaseLitModule` を継承した実験用 `LightningModule` を生成（例: 可視化を強化した検証出力など）。モデルと損失の配線を担う。
- **`lit_module.py`**（任意）: 実験特化のロギング/評価などを追加。

### 3.5. 損失（`loss/`）

必要に応じて独自損失を実装します。例として `dino_sequencial` では `HeatmapBCELoss` を `strategy.py` に定義し、`build_loss` ファクトリで生成します。

---

## 4. 実験の実行方法（**改訂**）

シンプルなシェルスクリプトでの起動を推奨します。

**`run_train.sh`（例）**

```bash
#!/bin/bash
# 使い方: bash run_train.sh <project> <experiment>
PROJECT=$1
EXPERIMENT=$2

EXP_PATH="development/${PROJECT}/${EXPERIMENT}"
HYDRA_RUN_DIR="outputs/${PROJECT}/${EXPERIMENT}"

python -m development.core.run \
  project=${PROJECT} \
  experiment=${EXPERIMENT} \
  +experiment_config_dir=${EXP_PATH}/configs \
  hydra.run.dir=${HYDRA_RUN_DIR} \
  --multirun
```

**実行例（`dino_sequencial`）**

```bash
bash run_train.sh ball_tracking dino_sequencial
```

- `python -m development.core.run`: コアのエントリポイントを実行。
- `project` / `experiment`: ログやルーティング用の識別子。
- `+experiment_config_dir`: **最重要**。Hydra の検索パスに **`development/<project>/<experiment>/configs`** を追加し、**実験側の差分設定**を見つけられるようにします。
- `hydra.run.dir`: 出力ディレクトリをプロジェクト/実験ごとに分離。
- `--multirun`: Hydra のマルチラン有効化。

> 互換メモ: 旧構成で `experiments/<exp_name>` を想定していたスクリプトは、`EXP_PATH` と `HYDRA_RUN_DIR` を上記形式に置き換えてください。

---

## 5. まとめ

本アーキテクチャは、**共通ロジックを `development/core` に集約**し、**設定駆動**で差分のみを実装することで、実験の**迅速な立ち上げ**と**コードの一貫性**を両立します。依存方向を **`development/<project>/<experiment> -> core`** に固定し、配置パスと設定のルーティングを明確化することで、チーム全体でのスケールと保守性を高めます。
