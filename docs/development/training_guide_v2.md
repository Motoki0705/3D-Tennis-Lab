# Training Guide v2: Advanced Features

This document outlines how to use the advanced training features including Mixture of Experts (MoE), Generative Adversarial Networks (GAN), and Self-Supervised Learning (SSL).

## Configuration

All advanced features are configured in `configs/ball_heatmap_full_v2.yaml`. They are disabled by default.

### Feature Flags

To enable a feature, set its `enabled` flag to `true`:

- `moe.enabled: true`: Enables the SwitchFFN layers in the ViT encoder.
- `gan.enabled: true`: Enables the trajectory discriminator and the GAN-related losses.
- `ssl.enabled: true`: Enables weak/strong augmentation pairs and consistency loss.

## Key Log Metrics

When features are enabled, new metrics will be logged to TensorBoard under these namespaces:

- `train/g_loss_adv`: Adversarial loss for the generator.
- `train/d_loss`: Total loss for the discriminator.
- `train/aux_moe_*`: Auxiliary losses from MoE layers (load balancing, overflow).
- `train/loss_ssl_consistency`: Consistency loss between weak and strong augmentations.
- `train/loss_velocity_tv`: Temporal variation loss for velocity.
- `train/loss_accel_tv`: Temporal variation loss for acceleration.
- `lambda/*`: Dynamically adjusted lambda values for each loss term.

## Recommended Training Schedule

The `LitModule` and `DynamicLambdaController` implement a 3-phase training schedule based on the current epoch:

1.  **Warmup (`training.warmup_epochs`)**: Base training with heatmap and offset losses. GAN and SSL are disabled.
2.  **Ramp (`training.ramp_epochs`)**: SSL is gradually introduced. The `lambda_self` value is ramped up from 0 to its target value.
3.  **Co-train**: All enabled features are active. The GAN is turned on, and the dynamic lambda controller continues to adjust weights.

## Debugging & Quick Recovery

- **Over-smoothing (blurry heatmaps, low peak values)**:

  - Decrease `lambda_v`, `lambda_a`, `lambda_adv`.
  - Increase `lambda_sharp`, `lambda_pk`.

- **MoE Imbalance/Overflow**:

  - Increase `moe.capacity_factor` (e.g., to 1.5).
  - Increase `training.loss.lambda_lb`.
  - Enable `moe.noisy_gating`.

- **GAN Instability**:
  - Decrease `training.loss.lambda_adv`.
  - Temporarily set `gan.enabled: false` to isolate the issue.
