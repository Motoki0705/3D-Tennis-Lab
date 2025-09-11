# Ball Tracking: シーケンシャル特徴量ベースの実装計画

## 1. 概要

このドキュメントは、DINOv3で事前計算した特徴量シーケンスを用いて、ボールの軌跡を予測する新しい学習パイプラインの実装計画を定義します。既存のフレーム単位の検出アプローチを補完し、時間的な連続性を考慮した軌跡予測モデルの構築を目指します。

## 2. フェーズ1: 特徴量の事前計算

**目的:** ビデオデータセットからDINOv3特徴量シーケンスを抽出し、高速な学習を可能にするための前処理済みデータセットを構築する。

### 2.1. 汎用エンコードスクリプトの作成

- **場所:** `tools/`
- **ファイル名:** `encode_video_features.py`
- **機能:**
  - コマンドライン引数で以下を受け取る。
    - `--video_path`: 入力ビデオまたはビデオが格納されたディレクトリのパス。
    - `--label_path`: 対応するアノテーションデータ（ボール座標）のパス。
    - `--output_dir`: 特徴量とラベルの出力先ディレクトリ。
    - `--encoder_model`: 使用するエンコーダーモデル名（例: `dinov3_vits14`）。
    - `--sequence_length`: 分割するシーケンスの長さ。
  - 内部処理:
    1. ビデオとアノテーションを読み込む。
    2. 指定された`sequence_length`でクリップに分割する。
    3. 各フレームをDINOv3でエンコードする。
    4. 特徴量シーケンスと座標ラベルシーケンスをペアにし、PyTorchテンソルとして保存する。
- **出力:**
  - `output_dir`（推奨: `data/processed/ball_tracking/sequential_features/`）に、クリップごとに以下のファイルを保存する。
    - `<clip_name>_features.pt`
    - `<clip_name>_labels.pt`

## 3. フェーズ2: 新規学習パイプラインの実装

既存のアーキテクチャガイドに準拠し、`development/ball_tracking/`内に以下のコンポーネントを**新規追加**します。

### 3.1. 提案ファイル構造

```
development/ball_tracking/
├── model/
│   └── lstm_tracker.py           # (新規追加)
├── training/
│   ├── sequential_dataset.py     # (新規追加)
│   ├── sequential_datamodule.py  # (新規追加)
│   └── lit_sequential_tracker.py # (新規追加)
└── configs/
    ├── data/
    │   └── sequential_ball.yaml    # (新規追加)
    ├── model/
    │   └── lstm_tracker.yaml       # (新規追加)
    ├── training/
    │   └── sequential.yaml         # (新規追加)
    └── experiment/
        └── ball_sequential_lstm.yaml # (新規追加)
```

### 3.2. 各コンポーネントの実装詳細

1.  **データパイプライン (`training/`)**

    - `sequential_dataset.py`: `sequential_features`ディレクトリから`.pt`ファイルを読み込む`torch.utils.data.Dataset`を実装。
    - `sequential_datamodule.py`: `SequentialDataset`を使用する`LightningDataModule`を実装。`configs/data/sequential_ball.yaml`から特徴量パスなどを受け取る。

2.  **モデル (`model/`)**

    - `lstm_tracker.py`: `nn.Module`を継承。DINOv3の特徴量シーケンス（次元: `(sequence_length, feature_dim)`）を入力とし、ボールの座標シーケンス（次元: `(sequence_length, 2)`）を出力する`nn.LSTM`と`nn.Linear`を組み合わせたモデルを定義。`configs/model/lstm_tracker.yaml`から`hidden_size`や`num_layers`を受け取る。

3.  **学習ロジック (`training/`)**

    - `lit_sequential_tracker.py`: `LightningModule`を継承。
      - `__init__`: `LSTMTracker`モデルと、`configs/training/sequential.yaml`で定義された学習率やオプティマイザ設定を受け取る。
      - `training_step`/`validation_step`: バッチデータを受け取り、モデルで予測を行い、`nn.MSELoss`などで損失を計算する。

4.  **設定ファイル (`configs/`)**
    - `data/sequential_ball.yaml`: `_target_`で`SequentialDataModule`を指定し、特徴量パスを定義。
    - `model/lstm_tracker.yaml`: `_target_`で`LSTMTracker`を指定し、モデルのハイパーパラメータを定義。
    - `training/sequential.yaml`: `_target_`で`LitSequentialTracker`を指定し、学習パラメータを定義。
    - `experiment/ball_sequential_lstm.yaml`: 上記3つの設定を`defaults`リストで組み合わせる実験ファイル。

## 4. 実行ワークフロー

1.  `python tools/encode_video_features.py --video_path <...> --output_dir data/processed/ball_tracking/sequential_features/` を実行し、特徴量を生成する。
2.  `development/ball_tracking/configs/config.yaml`の`defaults`リストを以下のように編集し、実験対象を切り替える。
    ```yaml
    defaults:
      - experiment: ball_sequential_lstm
      - _self_
    ```
3.  プロジェクトルートから `python -m development.ball_tracking.main` を実行して学習を開始する。
