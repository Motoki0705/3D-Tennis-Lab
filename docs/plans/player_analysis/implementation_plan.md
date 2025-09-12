# Player Analysis: シーケンシャル特徴量ベースの実装計画

## 1. 概要

このドキュメントは、DINOv3で事前計算した特徴量シーケンスを用いて、プレーヤーの連続的な姿勢（ポーズ）を推定する新しい学習パイプラインの実装計画を定義します。時間的情報を活用することで、単一フレームの推定よりも滑らかで、文脈に沿った自然な動作解析を実現することを目的とします。

## 2. フェーズ1: 特徴量の事前計算

**目的:** ビデオデータセットからDINOv3特徴量シーケンスを抽出し、高速な学習を可能にするための前処理済みデータセットを構築する。

### 2.1. 汎用エンコードスクリプトの活用

- **スクリプト:** `tools/encode_video_features.py` を流用します。
- **処理のカスタマイズ:**
  - `--label_path` には、プレーヤーの姿勢アノテーションデータ（例: 3Dキーポイント座標）を指定します。
  - スクリプトは、ビデオの各フレームに対応する姿勢パラメータを読み込み、特徴量シーケンスとペアで保存するようにします。
- **出力:**
  - `output_dir`（推奨: `data/processed/player_analysis/sequential_features/`）に、クリップごとに以下のファイルを保存します。
    - `<clip_name>_features.pt`
    - `<clip_name>_labels.pt`

## 3. フェーズ2: 新規学習パイプラインの実装

既存のアーキテクチャガイドに準拠し、`development/player_analysis/sequential_dino`内に以下のコンポーネントを**新規追加**します。

### 3.1. 提案ファイル構造

```
development/player_analysis/sequential_dino
├── model/
│   └── lstm_player_pose.py       # (新規追加)
├── training/
│   ├── sequential_dataset.py     # (新規追加)
│   ├── sequential_datamodule.py  # (新規追加)
│   └── lit_sequential_player_pose.py # (新規追加)
└── configs/
    ├── data/
    │   └── sequential_player_pose.yaml # (新規追加)
    ├── model/
    │   └── lstm_player_pose.yaml     # (新規追加)
    ├── training/
    │   └── sequential.yaml             # (新規追加)
    └── experiment/
        └── player_sequential_lstm.yaml # (新規追加)
```

### 3.2. 各コンポーネントの実装詳細

1.  **データパイプライン (`training/`)**

    - `sequential_dataset.py`: `sequential_features`ディレクトリから特徴量と姿勢ラベルの`.pt`ファイルを読み込む`Dataset`を実装。
    - `sequential_datamodule.py`: 上記`Dataset`を使用する`LightningDataModule`を実装。

2.  **モデル (`model/`)**

    - `lstm_player_pose.py`: `nn.Module`を継承。DINOv3特徴量シーケンスを入力とし、プレーヤーの姿勢パラメータのシーケンス（例: 次元 `(sequence_length, num_keypoints, 3)`）を出力する`nn.LSTM`ベースの回帰モデルを定義。

3.  **学習ロジック (`training/`)**

    - `lit_sequential_player_pose.py`: `LightningModule`を継承。
      - `__init__`: `LSTMPlayerPose`モデルと、学習率などの設定を受け取る。
      - `training_step`/`validation_step`: 予測されたキーポイントと正解キーポイントとの間で損失（例: `nn.MSELoss`）を計算する。

4.  **設定ファイル (`configs/`)**
    - `data/sequential_player_pose.yaml`: `SequentialDataModule`の`_target_`と特徴量パスを定義。
    - `model/lstm_player_pose.yaml`: `LSTMPlayerPose`の`_target_`とハイパーパラメータを定義。
    - `training/sequential.yaml`: `LitSequentialPlayerPose`の`_target_`と学習パラメータを定義。
    - `experiment/player_sequential_lstm.yaml`: 上記を組み合わせた実験ファイル。

## 4. 実行ワークフロー

1.  `python tools/encode_video_features.py --video_path <...> --label_path <...> --output_dir data/processed/player_analysis/sequential_features/` を実行し、特徴量を生成する。
2.  `development/player_analysis/configs/config.yaml`の`defaults`を`experiment: player_sequential_lstm`に設定する。
3.  プロジェクトルートから `python -m development.player_analysis.main` を実行して学習を開始する。
