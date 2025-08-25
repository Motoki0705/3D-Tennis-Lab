# Step 6.1: Pipeline runner infrastructure


class Stage:
    """Base class for a single step in the processing pipeline."""

    name = "stage"

    def process(self, bundle):
        """Processes the bundle and returns it."""
        return bundle


class PipelineRunner:
    """Runs a sequence of stages."""

    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run_once(self, bundle):
        """Runs all stages once on the bundle."""
        for st in self.stages:
            bundle = st.process(bundle)
        return bundle
