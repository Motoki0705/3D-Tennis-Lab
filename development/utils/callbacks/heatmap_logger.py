from pytorch_lightning import Callback, LightningModule, Trainer
from torchvision.utils import make_grid


class HeatmapImageLogger(Callback):
    def __init__(self, num_samples: int = 3):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True  # ログ記録の準備ができているかを示すフラグ

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        """検証エポックの開始時に呼ばれる"""
        # 準備OKの状態にする
        self.ready = True
        self.images = []
        self.pred_heatmaps = []
        self.target_heatmaps = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch,
        batch_idx: int,
    ):
        """検証の1バッチが終了するごとに呼ばれる"""
        # 最初のバッチのデータのみを保存する
        if self.ready and batch_idx == 0:
            # `validation_step`から返された辞書(outputs)からデータを取得
            self.images = outputs["images"][: self.num_samples]
            self.pred_heatmaps = outputs["pred_heatmaps"][: self.num_samples]
            self.target_heatmaps = outputs["target_heatmaps"][: self.num_samples]
            # 1エポックで一度だけ実行するため、フラグをFalseに
            self.ready = False

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """検証エポックの終了時に呼ばれる"""
        # ロガーが存在しない、または画像が保存されていない場合は何もしない
        if not trainer.logger or not len(self.images):
            return

        writer = trainer.logger.experiment

        # 1. 入力画像をグリッドにして追加
        img_grid = make_grid(self.images, normalize=True)
        writer.add_image("Validation/Input_Images", img_grid, global_step=pl_module.current_epoch)

        # ヒートマップのキーポイント数を取得 (例: 15)
        num_keypoints = self.pred_heatmaps.shape[1]
        # 2. 予測ヒートマップをキーポイントごとにループして保存
        for k in range(num_keypoints):
            # k番目のキーポイントのヒートマップだけを抽出
            # `[:, k:k+1, :, :]` とスライスして次元を(N, 1, H, W)に保つのがポイント
            pred_heatmap_k = self.pred_heatmaps[:, k : k + 1, :, :]
            pred_grid_k = make_grid(pred_heatmap_k, normalize=True)

            # TensorBoardのタグにキーポイント番号を追加
            tag = f"Validation_Pred/Heatmap_KP{k:02d}"
            writer.add_image(tag, pred_grid_k, global_step=pl_module.current_epoch)
