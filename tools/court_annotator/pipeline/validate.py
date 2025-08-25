from .runner import Stage
from ..domain.rules import is_ready


class ValidateStage(Stage):
    name = "validate"

    def __init__(self, params):
        self.params = params

    def process(self, bundle):
        # This will check if the current annotation is READY
        fit_result = bundle.get("fit_result")
        if fit_result:
            bundle["is_ready"] = is_ready(
                fit_result.used,
                fit_result.rmse,
                self.params.get("min_used_points", 4),
                self.params.get("rmse_ready", 2.5),
            )
        else:
            bundle["is_ready"] = False
        return bundle
