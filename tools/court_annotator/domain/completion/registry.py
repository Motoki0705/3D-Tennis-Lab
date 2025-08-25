# Step 4.3: Registry to resolve completion strategies
from .homography import HomographyCompletion


def resolve(strategy_name: str, template_xy):
    """
    Factory function to get a completion strategy instance.
    """
    if strategy_name == "homography":
        return HomographyCompletion(template_xy)
    raise ValueError(f"Unknown completion strategy: {strategy_name}")
