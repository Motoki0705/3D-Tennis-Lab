"""
Ball heatmap-based detection module with multiscale deep supervision.

Modules:
- dataset.py: COCO-based dataset for ball-only targets with multiscale heatmaps and offsets.
- datamodule.py: DataModule with weighted/balanced sampling and epoch-driven v-mix schedules.
- model/: ViT encoder + simple decoder + multiscale heads for heatmap and offsets.
- lit_module.py: LightningModule that composes multiscale losses and metrics.
- train.py: Hydra entrypoint to train the model.
- infer.py / track.py / evaluate.py: Stubs for inference, tracking, and evaluation.
"""
