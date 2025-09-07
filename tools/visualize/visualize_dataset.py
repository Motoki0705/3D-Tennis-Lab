import hydra
from omegaconf import DictConfig
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import make_grid

# Adjust the path to import from the development directory
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from development.ball_tracking.ball_heatmap.training.dataset import BallDataset


def visualize_sample(sample, cfg):
    sup_data = sample.get("sup")
    if not sup_data:
        print("Supervised data not found in this sample.")
        return

    video = sup_data["video"]  # [T, C, H, W]
    targets = sup_data["targets"]
    meta = sup_data["meta"]

    T, C, H, W = video.shape

    print("\n--- Visualizing Sample ---")
    print(f"Clip Info: game_id={meta['game_id']}, clip_id={meta['clip_id']}")
    print(f"Frame Paths: {meta['paths'][0]} ...")

    # --- 1. Video + Coordinates + Speed ---
    # Adjust subplot layout to be more compact
    cols = 4
    rows = (T + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    axes = axes.flatten()

    coords = targets["coords_img"]  # [T, 2]
    speed = targets["speed"]  # [T, 2]
    vis_state = targets["vis_state"]  # [T]

    # Speed is normalized, so un-normalize for visualization
    speed_unnormalized = speed * torch.tensor([W, H], dtype=torch.float32)

    for t in range(T):
        img_t = video[t].permute(1, 2, 0).numpy()  # CHW -> HWC
        ax = axes[t]
        ax.imshow(img_t)
        ax.set_title(f"Frame {t}")
        ax.axis("off")

        # Plot coordinates
        x, y = coords[t]
        print(
            f"Frame {t}: Coord=({x:.1f}, {y:.1f}), Speed=({speed_unnormalized[t, 0]:.1f}, {speed_unnormalized[t, 1]:.1f}), Vis={vis_state[t].item()}"
        )
        if not torch.isnan(x):
            # Color based on visibility: 2-visible (lime), 1-occluded (yellow), 0-not in frame (red)
            color = {2: "lime", 1: "yellow", 0: "red"}.get(vis_state[t].item(), "white")
            circ = patches.Circle(
                (x, y), radius=5, color=color, fill=True, alpha=0.8, label=f"Visibility: {vis_state[t].item()}"
            )
            ax.add_patch(circ)

            # Plot speed vector as an arrow
            vx, vy = speed_unnormalized[t]
            # Scale arrow for better visibility
            arrow_scale = 2.0
            ax.arrow(
                x.item(),
                y.item(),
                vx.item() * arrow_scale,
                vy.item() * arrow_scale,
                head_width=8,
                head_length=8,
                fc="cyan",
                ec="cyan",
                length_includes_head=True,
            )

    # Hide unused subplots
    for i in range(T, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Input Sequence with GT Coordinates and Speed Vectors (Forward Difference)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 2. Heatmaps ---
    heatmaps = targets["hm"]  # List of [T, 1, Hs, Ws]
    for i, hm_scale in enumerate(heatmaps):
        scale_val = cfg.data.scales[i]
        # Create a grid of heatmap images
        hm_grid = make_grid(hm_scale, nrow=T, normalize=True)

        fig, ax = plt.subplots(1, 1, figsize=(15, 2))
        ax.imshow(hm_grid.permute(1, 2, 0).numpy())
        ax.set_title(f"Target Heatmaps (Scale: {scale_val})")
        ax.axis("off")
        plt.show()


@hydra.main(config_path="../development/ball_tracking/ball_heatmap/configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to load dataset and visualize samples."""
    # Instantiate the labeled dataset
    dataset = BallDataset(cfg=cfg.data, labeled=True, semisup=False)

    print(f"Dataset contains {len(dataset)} samples.")

    if len(dataset) == 0:
        print("No samples found in the dataset.")
        return

    # Visualize a few samples
    indices_to_show = [0]
    if len(dataset) > 1:
        indices_to_show.append(len(dataset) // 2)
        indices_to_show.append(len(dataset) - 1)

    print(f"Showing samples for indices: {indices_to_show}")

    for idx in indices_to_show:
        if idx < len(dataset):
            sample = dataset[idx]
            visualize_sample(sample, cfg)


if __name__ == "__main__":
    main()
