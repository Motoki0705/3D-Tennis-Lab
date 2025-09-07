import torch
import torch.nn.functional as F


def warp_points(points: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    """
    Warp a batch of 2D points using a homography matrix.

    Args:
        points (torch.Tensor): Batch of points to transform, shape [B, N, 2] (x, y).
        homography (torch.Tensor): Batch of homography matrices, shape [B, 3, 3].

    Returns:
        torch.Tensor: Batch of warped points, shape [B, N, 2].
    """
    B, N, _ = points.shape
    # Convert points to homogeneous coordinates
    homogeneous_points = F.pad(points, (0, 1), "constant", 1.0)  # [B, N, 3]

    # Apply homography: H * p
    # Unsqueeze homography to [B, 1, 3, 3] for broadcasting over N points
    warped_points_h = torch.bmm(homogeneous_points, homography.transpose(1, 2))

    # Normalize to get 2D coordinates
    warped_points = warped_points_h[..., :2] / (warped_points_h[..., 2:] + 1e-8)

    return warped_points


def get_affine_grid(height: int, width: int, affine_matrix: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Generates a sampling grid for F.affine_grid.

    Args:
        height (int): The height of the target grid.
        width (int): The width of the target grid.
        affine_matrix (torch.Tensor): The [B, 2, 3] affine transformation matrix.
        device (torch.device): The device to create the grid on.

    Returns:
        torch.Tensor: The sampling grid, shape [B, H, W, 2].
    """
    B = affine_matrix.shape[0]
    grid = F.affine_grid(
        affine_matrix,
        (B, 1, height, width),
        align_corners=False,
    ).to(device)
    return grid


def warp_heatmap(heatmap: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Warps a heatmap using a pre-computed sampling grid.

    Args:
        heatmap (torch.Tensor): The heatmap to warp, shape [B, C, H, W].
        grid (torch.Tensor): The sampling grid from get_affine_grid, shape [B, H, W, 2].

    Returns:
        torch.Tensor: The warped heatmap, shape [B, C, H, W].
    """
    return F.grid_sample(heatmap, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
