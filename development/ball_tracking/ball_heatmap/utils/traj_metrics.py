from typing import Tuple

import torch


def calculate_velocity(trajectory: torch.Tensor, mask: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """
    Calculates the velocity of a trajectory, respecting a validity mask.

    Args:
        trajectory (torch.Tensor): The input trajectory, shape [B, T, 2] (x, y).
        mask (torch.Tensor): A validity mask, shape [B, T]. Invalid points (0) are ignored.
        dt (float): Time delta between frames.

    Returns:
        torch.Tensor: The velocity vectors, shape [B, T-1, 2].
    """
    # diff() calculates x_i - x_{i-1}
    velocity = torch.diff(trajectory, dim=1) / dt

    # Mask out invalid transitions
    # A transition t -> t+1 is valid only if both points are valid.
    valid_transitions = (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1)
    return velocity * valid_transitions


def calculate_acceleration(velocity: torch.Tensor, mask: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """
    Calculates the acceleration from a velocity sequence, respecting a validity mask.

    Args:
        velocity (torch.Tensor): The input velocity, shape [B, T-1, 2].
        mask (torch.Tensor): The original validity mask for positions, shape [B, T].
        dt (float): Time delta between frames.

    Returns:
        torch.Tensor: The acceleration vectors, shape [B, T-2, 2].
    """
    acceleration = torch.diff(velocity, dim=1) / dt

    # An acceleration at t+1 (from v_t and v_{t+1}) is valid only if
    # positions t, t+1, and t+2 were all valid.
    valid_transitions = (mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]).unsqueeze(-1)
    return acceleration * valid_transitions


def temporal_regularization_loss(
    trajectory: torch.Tensor, mask: torch.Tensor, dt: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes L1 losses for velocity and acceleration (Temporal Variation).

    Args:
        trajectory (torch.Tensor): The input trajectory, shape [B, T, 2].
        mask (torch.Tensor): A validity mask, shape [B, T].
        dt (float): Time delta between frames.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - The L1 velocity loss (mean over valid transitions).
            - The L1 acceleration loss (mean over valid transitions).
    """
    # Velocity
    velocity = calculate_velocity(trajectory, mask, dt)
    valid_vel_transitions = mask[:, 1:] * mask[:, :-1]
    num_valid_vel = valid_vel_transitions.sum().clamp(min=1.0)
    loss_v = torch.abs(velocity).sum() / num_valid_vel

    # Acceleration
    acceleration = calculate_acceleration(velocity, mask, dt)
    valid_accel_transitions = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
    num_valid_accel = valid_accel_transitions.sum().clamp(min=1.0)
    loss_a = torch.abs(acceleration).sum() / num_valid_accel

    return loss_v, loss_a
