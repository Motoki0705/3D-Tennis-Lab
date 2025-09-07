# RT-DETR Finetuning (Player Detection)

This module scaffolds a finetuning environment for RT-DETR to detect players based on the existing annotation.json used in `hrnet_finetuning`.

- Expects COCO-like annotations where `player` category bboxes are stored as `[x, y, w, h]` in `annotations[].bbox`.
- Uses PyTorch Lightning + Hydra configs.
- The RT-DETR model itself is not bundled. Provide a factory via `cfg.model.builder` that returns an `nn.Module` with `forward(images, targets)` returning a loss dict.

## Quick Start

1. Edit paths in `configs/data/player_detection.yaml` (`images_root`, `labeled_json`).
2. Provide a model builder in `configs/model/rtdetr.yaml`:

   # configs/model/rtdetr.yaml

   num_classes: 1
   builder: "my_pkg.models.rtdetr.build_model"

Where `my_pkg.models.rtdetr.build_model(num_classes: int, cfg) -> nn.Module` returns your RT-DETR model.

3. Launch training:

   python -m development.player_analysis.rt_detr.main task=train

Check logs and checkpoints under `tb_logs/rt_detr_finetune/`.
