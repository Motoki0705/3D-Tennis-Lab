# Step 4.1: Base class for completion strategies
from ...core.types import FitResult


class CompletionStrategy:
    """Abstract base class for all completion strategies."""

    def fit_and_complete(self, frame_state, court_spec, params) -> FitResult:
        """Fits the model and autocompletes keypoints."""
        raise NotImplementedError
