# 3. コンポーネントの役割と依存関係

このアーキテクチャの各コンポーネントがどのように連携し、それぞれの役割を果たしているかを、依存関係のフロー（Dependency Flow）を通じて解説します。

---

## 3.1. 依存関係フロー

`main.py`を起点として、各コンポーネントは以下のように連携します。重要なのは、**責務が明確に分離されている**点です。

1.  **エントリーポイント (`main.py`)**: **司令塔**
    `main.py`の責務は、Hydraを起動して設定オブジェクト(`cfg`)を生成し、それを`cfg.task`に応じて適切な`runner`に渡すことのみです。

    ```python
    # main.py
    import hydra
    from omegaconf import DictConfig

    # runnerをインポート
    from .runner.train import TrainRunner
    from .runner.infer import InferRunner

    @hydra.main(config_path="configs", config_name="config.yaml")
    def main(cfg: DictConfig) -> None:
        """
        設定ファイルを読み込み、cfg.taskに応じて適切なタスク実行役(runner)に処理を委譲する。
        """
        if cfg.task == "train":
            TrainRunner(cfg).run()
        elif cfg.task == "infer":
            InferRunner(cfg).run()
        else:
            raise ValueError(f"Unknown task: {cfg.task}")

    if __name__ == "__main__":
        main()
    ```

2.  **タスク実行役 (`runner/`)**: **オーケストレーター**
    `runner`は、`cfg`を受け取り、各コンポーネント（部品）をインスタンス化して組み立て、`Trainer`に引き渡す**指揮者**です。

    ```python
    # runner/train.py
    from omegaconf import DictConfig
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger

    # 各コンポーネント（部品）をインポート
    from ..callbacks import build_callbacks # ファクトリ関数
    from ..training.datamodule import ProjectDataModule
    from ..training.module import ProjectLitModule
    from ..model import build_model # モデル構築用の関数（例）

    class TrainRunner:
        def __init__(self, cfg: DictConfig):
            self.cfg = cfg

        def run(self):
            # 1. Callbacks: cfg.callbacksからコールバック群を構築
            callbacks = build_callbacks(self.cfg.callbacks)

            # 2. DataModule: cfg.dataからデータモジュールを構築
            datamodule = ProjectDataModule(**self.cfg.data)

            # 3. Model: cfg.modelからモデルを構築
            model = build_model(self.cfg.model)

            # 4. LightningModule: モデルとcfgから学習モジュールを構築
            lit_module = ProjectLitModule(model=model, **self.cfg.lit_module)

            # 5. Logger: cfg.loggerからロガーを構築
            logger = TensorBoardLogger(**self.cfg.logger)

            # 6. Trainer: cfg.trainerと上記コンポーネントからTrainerを構築
            trainer = pl.Trainer(
                callbacks=callbacks,
                logger=logger,
                **self.cfg.trainer
            )

            # 7. 学習開始
            trainer.fit(model=lit_module, datamodule=datamodule)
    ```

---

## 3.2. 各コンポーネントの役割まとめ

上記のフローから、各コンポーネントの役割は以下のように明確化されます。

| コンポーネント               | 役割       | 依存されるコンポーネント         | 説明                                                                                               |
| :--------------------------- | :--------- | :------------------------------- | :------------------------------------------------------------------------------------------------- |
| **`main.py`**                | **司令塔** | `runner`                         | Hydraを起動し、`cfg`を`runner`に渡す。                                                             |
| **`runner/`**                | **指揮者** | `callbacks`, `training`, `model` | `cfg`を元に各コンポーネントを組み立て、`Trainer`を起動する。                                       |
| **`callbacks/`**             | **部品**   | (なし)                           | カスタムCallbackの実装や、`cfg`からCallback群を生成するファクトリ (`build_callbacks`) を配置する。 |
| **`training/datamodule.py`** | **部品**   | `training/dataset.py`            | `cfg.data`に基づき、データセットの準備とデータローダーの作成を行う。                               |
| **`training/module.py`**     | **部品**   | `model`, `losses`                | `cfg`に基づきモデルや損失関数を組み合わせ、学習・検証ステップを定義する。                          |
| **`model/`**                 | **部品**   | (なし)                           | `cfg.model`に基づき、モデルの骨格（`nn.Module`）を定義する。                                       |
| **`configs/`**               | **設計図** | (全て)                           | 全てのコンポーネントの挙動を定義するYAMLファイル群。                                               |
