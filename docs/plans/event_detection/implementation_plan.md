# Event Detection: シーケンシャル特徴量ベースの実装計画

## 1. 概要

このドキュメントは、DINOv3で事前計算した特徴量シーケンスを用いて、テニスの試合における特定のイベント（例: サーブ、フォアハンド、バックハンド、バウンド）を検出・分類する新しい学習パイプラインの実装計画を定義します。これは、特徴量シーケンスを入力とし、各タイムステップのイベントカテゴリを出力する**シーケンス分類タスク**としてモデル化します。

## 2. フェーズ1: 特徴量の事前計算

**目的:** ビデオデータセットからDINOv3特徴量シーケンスを抽出し、高速な学習を可能にするための前処理済みデータセットを構築する。

### 2.1. 汎用エンコードスクリプトの活用

- **スクリプト:** `tools/encode_video_features.py` を流用します。
- **処理のカスタマイズ:**
  - `--label_path` には、イベントのアノテーションデータ（例: `(frame_index, event_class_id)`) を指定します。
  - スクリプトは、ビデオの各フレームに対応するイベントクラスを読み込み、特徴量シーケンスとペアで保存するようにします。
- **出力:**
  - `output_dir`（推奨: `data/processed/event_detection/sequential_features/`）に、クリップごとに以下のファイルを保存します。
    - `<clip_name>_features.pt`
    - `<clip_name>_labels.pt` (ラベルはクラスIDのテンソル)

## 3. フェーズ2: 新規学習パイプラインの実装

既存のアーキテクチャガイドに準拠し、`development/event_detection/`内に以下のコンポーネントを**新規追加**します。

### 3.1. 提案ファイル構造

```
development/event_detection/
├── model/
│   └── lstm_event_classifier.py    # (新規追加)
├── training/
│   ├── sequential_dataset.py         # (新規追加)
│   ├── sequential_datamodule.py      # (新規追加)
│   └── lit_sequential_classifier.py  # (新規追加)
└── configs/
    ├── data/
    │   └── sequential_event.yaml       # (新規追加)
    ├── model/
    │   └── lstm_event_classifier.yaml  # (新規追加)
    ├── training/
    │   └── sequential.yaml             # (新規追加)
    └── experiment/
        └── event_sequential_lstm.yaml  # (新規追加)
```

### 3.2. 各コンポーネントの実装詳細

1.  **データパイプライン (`training/`)**

    - `sequential_dataset.py`: `sequential_features`ディレクトリから特徴量とイベントラベルの`.pt`ファイルを読み込む`Dataset`を実装。
    - `sequential_datamodule.py`: 上記`Dataset`を使用する`LightningDataModule`を実装。

2.  **モデル (`model/`)**

    - `lstm_event_classifier.py`: `nn.Module`を継承。DINOv3特徴量シーケンスを入力とし、各タイムステップにおけるイベントクラスの確率（次元: `(sequence_length, num_event_classes)`）を出力する`nn.LSTM`と`nn.Linear`を組み合わせた分類モデルを定義。

3.  **学習ロジック (`training/`)**

    - `lit_sequential_classifier.py`: `LightningModule`を継承。
      - `__init__`: `LSTMEventClassifier`モデルと、学習率などの設定を受け取る。
      - `training_step`/`validation_step`: 予測されたクラス確率と正解クラスIDとの間で、分類損失（例: `nn.CrossEntropyLoss`）を計算する。

4.  **設定ファイル (`configs/`)**
    - `data/sequential_event.yaml`: `SequentialDataModule`の`_target_`と特徴量パス、クラス数などを定義。
    - `model/lstm_event_classifier.yaml`: `LSTMEventClassifier`の`_target_`とハイパーパラメータを定義。
    - `training/sequential.yaml`: `LitSequentialClassifier`の`_target_`と学習パラメータを定義。
    - `experiment/event_sequential_lstm.yaml`: 上記を組み合わせた実験ファイル。

## 4. 実行ワークフロー

1.  `python tools/encode_video_features.py --video_path <...> --label_path <...> --output_dir data/processed/event_detection/sequential_features/` を実行し、特徴量を生成する。
2.  `development/event_detection/configs/config.yaml`の`defaults`を`experiment: event_sequential_lstm`に設定する。
3.  プロジェクトルートから `python -m development.event_detection.main` を実行して学習を開始する。
