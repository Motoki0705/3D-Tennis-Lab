# Step 3.2: Domain business rules


def is_ready(used: int, rmse: float, min_used: int, rmse_ready: float) -> bool:
    """Determines if the current annotation is good enough to proceed."""
    return used >= min_used and rmse is not None and rmse <= rmse_ready
