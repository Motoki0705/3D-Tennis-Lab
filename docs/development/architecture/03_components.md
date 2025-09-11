# 3. コンポーネントの役割と依存関係

このアーキテクチャの各コンポーネントがどのように連携し、それぞれの役割を果たしているかを、依存関係のフロー（Dependency Flow）を通じて解説します。

---

## 3.1. 依存関係フロー

`main.py`を起点として、各コンポーネントは以下のように連携します。重要なのは、**責務が明確に分離されている**点です。

1.  **エントリーポイント (`main.py`)**: **司令塔**
    `main.py`の責務は、Hydraを起動して設定オブジェクト(`cfg`)を生成し、それを`runner`に渡すことのみです。ここにはビジネスロジックを一切記述しません。

    ```python
    # main.py
    import hydra
    from omegaconf import DictConfig
    from .runner import run_task # runnerをインポート

    @hydra.main(config_path="configs", config_name="config.yaml")
    def main(cfg: DictConfig) -> None:
        """
        設定ファイルを読み込み、タスク実行役であるrunnerに処理を委譲する。
        """
        # cfg.taskに応じて、run_trainingやrun_inferenceを呼び分ける
        run_task(cfg)

    if __name__ == "__main__":
        main()
    ```

2.  **タスク実行役 (`runner/`)**: **オーケストレーター**
    `runner`は、`cfg`を受け取り、各コンポーネント（部品）をインスタンス化して組み立て、`Trainer`に引き渡す**指揮者**です。

    ```python
    # runner/train.py (run_taskから呼び出される)
    from omegaconf import DictConfig
    import pytorch_lightning as pl

    # 各コンポーネント（部品）をインポート
    from ..callbacks import build_callbacks
    from ..training.datamodule import ProjectDataModule
    from ..training.module import ProjectLitModule

    def run_training(cfg: DictConfig) -> None:
        """
        cfgから各コンポーネントを構築し、学習を実行する。
        """
        # 1. Callbacks: cfg.callbacksからコールバック群を構築
        callbacks = build_callbacks(cfg.callbacks)

        # 2. DataModule: cfg.dataからデータモジュールを構築
        datamodule = ProjectDataModule(cfg.data)

        # 3. LightningModule: cfgから学習モジュールを構築
        lit_module = ProjectLitModule(cfg)

        # 4. Trainer: cfg.trainingと上記コンポーネントからTrainerを構築
        trainer = pl.Trainer(
            callbacks=callbacks,
            **cfg.training.trainer # max_epochs, gpusなどの設定を展開
        )

        # 5. 学習開始: 全てのコンポーネントをTrainerに渡して学習を開始
        trainer.fit(model=lit_module, datamodule=datamodule)
    ```

---

## 3.2. 各コンポーネントの役割まとめ

上記のフローから、各コンポーネントの役割は以下のように明確化されます。

| コンポーネント               | 役割       | 依存されるコンポーネント         | 説明                                                         |
| :--------------------------- | :--------- | :------------------------------- | :----------------------------------------------------------- |
| **`main.py`**                | **司令塔** | `runner`                         | Hydraを起動し、`cfg`を`runner`に渡す。                       |
| **`runner/`**                | **指揮者** | `callbacks`, `training`, `model` | `cfg`を元に各コンポーネントを組み立て、`Trainer`を起動する。 |
| **`callbacks/`**             | **部品**   | (なし)                           | `cfg.callbacks`からコールバック群を生成するファクトリ。      |
| **`training/datamodule.py`** | **部品**   | (なし)                           | `cfg.data`からデータセットとデータローダーを準備する。       |
| **`training/module.py`**     | **部品**   | `model`                          | `cfg`からモデルや損失関数を準備し、学習ステップを定義する。  |
| **`model/`**                 | **部品**   | (なし)                           | `cfg.model`からモデルの骨格（`nn.Module`）を定義する。       |
| **`configs/`**               | **設計図** | (全て)                           | 全てのコンポーネントの挙動を定義するYAMLファイル群。         |
