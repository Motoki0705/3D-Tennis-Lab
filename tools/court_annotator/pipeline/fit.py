# Step 6.2: Example stage implementation (FitStage)
from .runner import Stage
from ..domain.completion.registry import resolve


class FitStage(Stage):
    """A pipeline stage for fitting and auto-completing keypoints."""

    name = "fit"

    def __init__(self, court_spec, strategy_name, params):
        self.spec = court_spec
        self.strategy_name = strategy_name
        self.params = params
        # Resolve once to avoid per-loop instantiation cost
        self.strategy = resolve(self.strategy_name, self.spec.template_xy)

    def process(self, bundle):
        # Get the current frame's state from the bundle
        frame_state = bundle["frame_state"]
        last_fit = getattr(frame_state, "last_fit", None)

        # Skip expensive fit while dragging
        if getattr(frame_state.drag, "active", False):
            if last_fit is not None:
                bundle["fit_result"] = last_fit
            return bundle

        # Only recompute when needed: on frame change, after edits, or forced
        dirty = bool(
            bundle.get("dirty", False) or bundle.get("frame_changed", False) or bundle.get("force_recompute", False)
        )
        if not dirty and last_fit is not None:
            bundle["fit_result"] = last_fit
            return bundle

        # Execute the fit and completion logic once when needed
        fit_result = self.strategy.fit_and_complete(frame_state, self.spec, self.params)
        # Cache and publish
        frame_state.last_fit = fit_result
        bundle["fit_result"] = fit_result
        # Reset transient flags
        bundle["dirty"] = False
        bundle["force_recompute"] = False
        return bundle
