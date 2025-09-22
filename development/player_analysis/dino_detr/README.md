# DINO-DETR Player Analysis Experiment

This experiment now uses the shared `development/core` runner. Launch training with:

```bash
python -m development.core.run \
  +experiment_config_dir=development/player_analysis/dino_detr/configs \
  project=player_analysis \
  experiment=dino_detr
```

Override data paths or hyper-parameters via Hydra CLI overrides (for example `data.images_root=...`).
