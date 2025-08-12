# filename: development/court_pose/01_vit_heatmap/train.py (Hydra version)
import hydra
from omegaconf import DictConfig, OmegaConf # HydraとOmegaConfをインポート

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import CourtDataModule
from lit_module import CourtLitModule


# @hydra.mainデコレータを使用
@hydra.main(config_path="configs", config_name="vit_heatmap_v1", version_base=None)
def train(config: DictConfig): # 引数でconfigを受け取る
    # Hydraは自動的にDotMapのようなアクセスを提供してくれるので、DotMap変換は不要
    
    # 1. LoggerとCallbackの設定
    logger = TensorBoardLogger("tb_logs", name="vit_heatmap_v1")
    
    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.checkpoint.monitor,
        mode=config.callbacks.checkpoint.mode,
        save_top_k=config.callbacks.checkpoint.save_top_k,
        filename='epoch={epoch}-pck={val/PCK@0.05:.4f}',
        auto_insert_metric_name=False
    )
    early_stop_callback = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        patience=config.callbacks.early_stopping.patience,
        mode=config.callbacks.early_stopping.mode,
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # 2. DataModuleとLightningModuleの初期化
    datamodule = CourtDataModule(config)
    model = CourtLitModule(config)
    
    # 3. Trainerの設定と起動
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor]
    )
    
    trainer.fit(model, datamodule=datamodule)
    
    print("--- Starting Test ---")
    trainer.test(datamodule=datamodule, ckpt_path='best')

if __name__ == '__main__':
    train()