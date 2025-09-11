# Court Pose: シーケンシャル特徴量ベースの実装計画

## 1. 概要

このドキュメントは、DINOv3で事前計算した特徴量シーケンスを用いて、ビデオ内のカメラポーズを時系列で推定する新しい学習パイプラインの実装計画を定義します。フレーム間の時間的連続性を活用し、より滑らかで安定したカメラポーズ推定を実現することを目的とします。

## 2. フェーズ1: 特徴量の事前計算

**目的:** ビデオデータセットからDINOv3特徴量シーケンスを抽出し、高速な学習を可能にするための前処理済みデータセットを構築する。

### 2.1. 汎用エンコードスクリプトの活用

- **スクリプト:** `tools/encode_video_features.py` を流用します。
- **処理のカスタマイズ:**
  - `--label_path` には、カメラポーズのアノテーションデータを指定します。
  - スクリプトは、ビデオの各フレームに対応するカメラポーズパラメータ（例: 回転行列、並進ベクトル）を読み込み、特徴量シーケンスとペアで保存するようにします。
- **出力:**
  - `output_dir`（推奨: `data/processed/court_pose/sequential_features/`）に、クリップごとに以下のファイルを保存します。
    - `<clip_name>_features.pt`
    - `<clip_name>_labels.pt`

## 3. フェーズ2: 新規学習パイプラインの実装

既存のアーキテクチャガイドに準拠し、`development/court_pose/`内に以下のコンポーネントを**新規追加**します。

### 3.1. 提案ファイル構造

```
development/court_pose/
├── model/
│   └── lstm_pose_estimator.py    # (新規追加)
├── training/
│   ├── sequential_dataset.py     # (新規追加)
│   ├── sequential_datamodule.py  # (新規追加)
│   └── lit_sequential_pose.py    # (新規追加)
└── configs/
    ├── data/
    │   └── sequential_pose.yaml      # (新規追加)
    ├── model/
    │   └── lstm_pose_estimator.yaml  # (新規追加)
    ├── training/
    │   └── sequential.yaml           # (新規追加)
    └── experiment/
        └── court_sequential_lstm.yaml # (新規追加)
```

### 3.2. 各コンポーネントの実装詳細

1.  **データパイプライン (`training/`)**

    - `sequential_dataset.py`: `sequential_features`ディレクトリから特徴量とポーズラベルの`.pt`ファイルを読み込む`Dataset`を実装。
    - `sequential_datamodule.py`: 上記`Dataset`を使用する`LightningDataModule`を実装。

2.  **モデル (`model/`)**

    - `lstm_pose_estimator.py`: `nn.Module`を継承。DINOv3特徴量シーケンスを入力とし、カメラポーズパラメータのシーケンス（例: 次元 `(sequence_length, num_pose_params)`）を出力する`nn.LSTM`ベースの回帰モデルを定義。

3.  **学習ロジック (`training/`)**

    - `lit_sequential_pose.py`: `LightningModule`を継承。
      - `__init__`: `LSTMPoseEstimator`モデルと、学習率などの設定を受け取る。
      - `training_step`/`validation_step`: 予測されたポーズと正解ポーズとの間で損失（例: `nn.MSELoss`）を計算する。

4.  **設定ファイル (`configs/`)**
    - `data/sequential_pose.yaml`: `SequentialDataModule`の`_target_`と特徴量パスを定義。
    - `model/lstm_pose_estimator.yaml`: `LSTMPoseEstimator`の`_target_`とハイパーパラメータを定義。
    - `training/sequential.yaml`: `LitSequentialPose`の`_target_`と学習パラメータを定義。
    - `experiment/court_sequential_lstm.yaml`: 上記を組み合わせた実験ファイル。

## 4. 実行ワークフロー

1.  `python tools/encode_video_features.py --video_path <...> --label_path <...> --output_dir data/processed/court_pose/sequential_features/` を実行し、特徴量を生成する。
2.  `development/court_pose/configs/config.yaml`の`defaults`を`experiment: court_sequential_lstm`に設定する。
3.  プロジェクトルートから `python -m development.court_pose.main` を実行して学習を開始する。
