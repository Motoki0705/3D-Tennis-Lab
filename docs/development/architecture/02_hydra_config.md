# 2. Hydraによる設定ファイル駆動開発

本アーキテクチャの要は**Hydra**です。コードの変更を一切行わずに、YAMLファイルを組み合わせるだけで実験内容を柔軟に変更できます。

---

## 2.1. メイン設定ファイル (`configs/config.yaml`)

`config.yaml`は、実験の構成を定義する「設計図」の役割を果たします。`defaults`リストを使って、どのモジュール設定（`data`, `model`など）を読み込むかを指定します。

**例: `configs/config.yaml`**

```yaml
defaults:
  # --- モジュール選択栓 --- #
  - data: <dataset_name> #例: `configs/data/tennis_match_sequence.yaml` を読み込む
  - model: <model_name> #例: `configs/model/vit_heatmap.yaml` を読み込む
  - training: default # `configs/training/default.yaml` を読み込む
  # -------------------- #
  - _self_ # このファイル内の設定を最後に読み込む

# --- グローバル設定 ---
seed: 42
project: "<project_name>" # 例: ball_tracking, player_analysis
experiment: "<experiment_name>" # 例: baseline, with_augmentation
task: "train" # 実行アクション (train, infer, testなど)

# --- 実験名の自動生成 ---
# <project>_<experiment>_<model> の形式を徹底する
experiment_name: "${project}_${experiment}_${model.name}"

# --- Hydra自体の設定 ---
hydra:
  run:
    dir: outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}
```

---

## 2.2. モジュール別設定ファイル

各コンポーネントのパラメータは、対応するディレクトリ内のYAMLファイルに記述します。ファイル名は`config.yaml`の`defaults`リストで指定したものと一致させます。

**例1: データセット設定 (`configs/data/<dataset_name>.yaml`)**

```yaml
# @package _group_

# この設定グループの名前（config.yamlから参照される）
name: "<dataset_name>" # 例: tennis_match_sequence

# --- データセット固有のパラメータ ---
path: "/path/to/your/dataset"
sequence_length: 16
batch_size: 8
num_workers: 4
```

**例2: モデル設定 (`configs/model/<model_name>.yaml`)**

```yaml
# @package _group_

# この設定グループの名前
name: "<model_name>" # 例: vit_heatmap

# --- モデル固有のパラメータ ---
pretrained: true
architecture:
  num_layers: 12
  num_heads: 8
```

**例3: コールバック設定 (`configs/callbacks/default.yaml`)**

Hydraの`_target_`機能を使えば、YAML上でインスタンス化するクラスを指定できます。これを利用して、使用するコールバックとそのパラメータを柔軟に管理します。

```yaml
# @package _group_

# 使用したいコールバックをキーとして列挙する
model_checkpoint:
  _target_: pytorch_lightning.callbacks.ModelCheckpoint
  dirpath: "checkpoints/"
  filename: "epoch{epoch:03d}-val_loss{val_loss:.3f}"
  monitor: "val_loss"
  mode: "min"
  save_top_k: 1

early_stopping:
  _target_: pytorch_lightning.callbacks.EarlyStopping
  monitor: "val_loss"
  patience: 5
  mode: "min"
```

---

## 2.3. 使い方

- **実験の切り替え**: `config.yaml`の`defaults`リストを編集するだけで、使用するデータセットやモデルを簡単に入れ替えられます。

  ```yaml
  defaults:
    - data: another_dataset
    - model: cnn_baseline
    - training: default
    - _self_
  ```

- **コマンドラインからの上書き**: `config.yaml`を直接編集せず、コマンドラインから一時的に設定値を変更することも強力な機能です。

  ```bash
  # 学習率とバッチサイズを変更して実験を実行
  python main.py training.optimizer.lr=1e-5 data.batch_size=32
  ```

---

## 2.4. 命名規則ガイドライン

一貫性のある命名は、プロジェクトの見通しを良くするために非常に重要です。Hydraの設定ファイル名には、以下の規則を推奨します。

- **ブランチ名との連携**: `config.yaml`の`project`と`experiment`の値は、Gitのブランチ名 `feature/<project>/<experiment>` と対応させることが推奨されます。これにより、どのブランチの実験設定かが一目瞭然になります。

  - **フォーマット**: ブランチ名では`kebab-case`、設定ファイル内では`snake_case`と、それぞれの慣例に従い使い分けます。

- **設定名の付け方**:

  - **`<dataset_name>`**: データセットの内容や形式が分かる名前（例: `tennis_match_sequence`, `player_trajectory`）。
  - **`<model_name>`**: モデルのアーキテクチャや特徴が分かる名前（例: `vit_heatmap`, `hrnet_3d_pose`）。

- **実験名 (`experiment_name`)**: `${project}_${experiment}_${model.name}` の形式を推奨します。これにより、Hydraの出力ディレクトリ名から実験内容が一目でわかるようになります。

- **フォーマット**: ファイル名や設定名には、小文字のスネークケース（`snake_case`）を使用することを推奨します。（`snake_case`）を使用することを推奨します。
