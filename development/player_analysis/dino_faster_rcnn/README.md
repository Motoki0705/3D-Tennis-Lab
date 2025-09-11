Overview

- Task: Faster R-CNN object detection with a frozen DINOv3 ViT backbone.
- Layout follows docs/development/development_architecture_guide.md.
- Backbone: DINOv3 ViT (loaded via local `third_party/dinov3`), frozen.

Quickstart

- Training: `python main.py task=train`
- Inference: `python main.py task=infer ckpt_path=PATH/TO.ckpt`

Config pointers (see configs/):

- `model.backbone.{repo_dir, entry, weights}`: local hub path, entry, and weights
- `model.num_classes`: number of foreground classes (incl. background handled internally)
- `data.train/images, data.train/ann`: COCO-style dataset paths
- `trainer.max_epochs, trainer.devices`: training settings

Notes

- This project uses the DINOv3 third_party repo as a local torch.hub to load ViT.
- ViT is frozen; only Faster R-CNN heads are trained.

Resolution Mix Scheduling

- You can reduce early-epoch cost by mixing low/high image resolutions during training.
- Enable by setting `data.image_size_low` and `data.image_size_high` in the data config.
- The `resolution_mix_scheduler` callback linearly increases the probability of sampling
  high-resolution transforms from 0.0 to 1.0 over the training epochs (configurable).
- Dataloaders are reloaded every epoch so workers pick up updated mix ratios.
